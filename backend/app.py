"""FastAPI backend: serves the UI and exposes /api/segment and /api/inpaint."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import imageops as ops
from .device import pick_device
from .inpainter import LamaInpainter
from .refiners import Progress, make_refiners
from .segmenter import Point, Sam2Segmenter

log = logging.getLogger("objremover")
ROOT = Path(__file__).resolve().parent.parent
MAX_POINTS = 64
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
QUALITIES = ("fast", "sdxl", "flux")


class ModelState:
    def __init__(self, segmenter, inpainter, refiners: dict):
        self.segmenter = segmenter
        self.inpainter = inpainter
        self.refiners = refiners
        self.error: str | None = None
        self.loading = False
        self.progress = Progress()
        self.inpaint_lock = threading.Lock()
        # All model loading is serialized: the transformers/diffusers loaders rely on
        # process-global state (meta-device init, lazy imports), so loading SAM and a
        # refiner in parallel threads leaves one of them half-initialized.
        self.load_lock = threading.Lock()

    def load_all(self):
        self.loading = True
        with self.load_lock:
            for model in (self.inpainter, self.segmenter):
                try:
                    if not model.loaded:
                        model.load()
                except Exception as err:  # noqa: BLE001 - surfaced to the UI
                    log.exception("failed to load %s", model.name)
                    self.error = f"{model.name}: {err}"
        self.loading = False

    def load_refiner(self, name: str) -> None:
        """Load one refiner, unloading the others first (only one fits in memory)."""
        with self.load_lock:
            target = self.refiners[name]
            if target.loaded:
                return
            for other_name, other in self.refiners.items():
                if other_name != name and other.loaded:
                    log.info("unloading %s to make room for %s", other_name, name)
                    other.unload()
            try:
                target.load()
            except Exception:  # noqa: BLE001 - recorded on the refiner
                pass


def create_app(segmenter=None, inpainter=None, refiners=None, *, load_models: bool = True) -> FastAPI:
    device = pick_device()
    state = ModelState(
        segmenter or Sam2Segmenter(device=device),
        inpainter or LamaInpainter(device=device),
        refiners if refiners is not None else make_refiners(device),
    )
    app = FastAPI(title="Obj Remover")
    app.state.models = state
    app.state.device = device

    if load_models:
        threading.Thread(target=state.load_all, name="model-loader", daemon=True).start()

    def foreign_origin(request: Request) -> bool:
        if request.headers.get("sec-fetch-site") == "cross-site":
            return True
        origin = request.headers.get("origin")
        if not origin:
            return False  # same-origin GET, curl, or a non-browser client
        try:
            parts = urlsplit(origin)  # handles IPv6 brackets, userinfo, "null"
            host, _ = parts.hostname, parts.port  # .port raises on a non-numeric port
        except ValueError:
            return True
        # A browser serializes Origin as scheme://host[:port] and nothing else.
        if parts.username is not None or parts.path or parts.query or parts.fragment:
            return True
        return parts.scheme not in ("http", "https") or host not in LOCAL_HOSTS

    @app.middleware("http")
    async def reject_cross_site(request: Request, call_next):
        """The server only listens on loopback, but any web page the user visits could
        still POST to it (a 12 GB download or a 5-minute GPU job). Browsers always
        send Origin on cross-origin POSTs and Sec-Fetch-Site on all fetches, so
        refuse every mutating request, and every /api call, that carries a foreign
        origin. Checking the method rather than only the path means no path
        spelling can route around the check."""
        mutating = request.method not in ("GET", "HEAD", "OPTIONS")
        if (mutating or request.url.path.startswith("/api/")) and foreign_origin(request):
            return JSONResponse(status_code=403, content={"detail": "cross-site requests are not allowed"})
        return await call_next(request)

    async def read_upload(upload: UploadFile) -> bytes:
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        return data

    @app.get("/api/health")
    def health():
        return {
            "device": device,
            "ready": state.segmenter.loaded and state.inpainter.loaded,
            "loading": state.loading,
            "error": state.error,
            "models": {
                "segmenter": {"name": state.segmenter.name, "loaded": state.segmenter.loaded},
                "inpainter": {"name": state.inpainter.name, "loaded": state.inpainter.loaded},
            },
            "refiners": {name: r.status() for name, r in state.refiners.items()},
        }

    @app.get("/api/progress")
    def progress():
        return state.progress.snapshot()

    @app.post("/api/models/{name}/load")
    def load_model(name: str):
        if name not in state.refiners:
            raise HTTPException(404, f"unknown model {name}")
        refiner = state.refiners[name]
        if not refiner.loaded and not refiner.loading:
            refiner.loading = True  # visible immediately, before the thread starts
            threading.Thread(target=state.load_refiner, args=(name,), name=f"load-{name}", daemon=True).start()
        return refiner.status()

    @app.post("/api/segment")
    async def segment(image: UploadFile = File(...), points: str = Form(...)):
        if not state.segmenter.loaded:
            raise HTTPException(503, "segmentation model is still loading")
        try:
            raw = json.loads(points)
            pts = [Point(float(p["x"]), float(p["y"]), int(p.get("label", 1))) for p in raw]
        except (ValueError, TypeError, KeyError) as err:
            raise HTTPException(400, f"invalid points: {err}")
        if not pts or len(pts) > MAX_POINTS:
            raise HTTPException(400, f"points must contain 1..{MAX_POINTS} entries")
        data = await read_upload(image)
        key = hashlib.sha256(data).hexdigest()
        try:
            rgb = ops.decode_rgb(data)
        except Exception as err:  # noqa: BLE001
            raise HTTPException(400, f"could not decode image: {err}")
        h, w = rgb.shape[:2]
        for p in pts:
            if not (0 <= p.x < w and 0 <= p.y < h) or p.label not in (0, 1):
                raise HTTPException(400, "point outside the image or invalid label")
        t0 = time.perf_counter()

        def run_segment():
            m = state.segmenter.segment(key, rgb, pts)
            # Expand the tight segmentation mask so the inpainter also removes the
            # object's edge halo and shadow instead of smearing them into the hole.
            return ops.dilate(m, ops.auto_dilate_radius(w, h))

        # Heavy work runs off the event loop so /api/health and /api/progress stay responsive.
        mask = await run_in_threadpool(run_segment)
        log.info("segment %dx%d %d pts %.2fs", w, h, len(pts), time.perf_counter() - t0)
        return Response(ops.encode_png(mask), media_type="image/png")

    @app.post("/api/inpaint")
    async def inpaint(
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        quality: str = Form("fast"),
        strength: float = Form(0.6),
    ):
        if quality not in QUALITIES:
            raise HTTPException(400, f"quality must be one of {QUALITIES}")
        if not (0.1 <= strength <= 1.0):
            raise HTTPException(400, "strength must be between 0.1 and 1.0")
        if not state.inpainter.loaded:
            raise HTTPException(503, "inpainting model is still loading")
        refiner = None
        if quality != "fast":
            refiner = state.refiners[quality]
            if not refiner.loaded:
                raise HTTPException(503, f"{refiner.label} is not loaded yet")
        try:
            image_bytes, mask_bytes = await read_upload(image), await read_upload(mask)
        except HTTPException:
            raise
        except Exception as err:  # noqa: BLE001
            raise HTTPException(400, f"could not read input: {err}")
        try:
            rgb = ops.decode_rgb(image_bytes)
            m = ops.decode_mask(mask_bytes, size=(rgb.shape[1], rgb.shape[0]))
        except ValueError as err:
            raise HTTPException(400, str(err))
        except Exception as err:  # noqa: BLE001
            raise HTTPException(400, f"could not decode input: {err}")
        if not m.any():
            raise HTTPException(400, "mask is empty")
        if not state.inpaint_lock.acquire(blocking=False):
            raise HTTPException(409, "another removal is still running")

        def run_inpaint():
            try:
                t0 = time.perf_counter()
                state.progress.set("lama")
                out = state.inpainter.inpaint(rgb, m)
                log.info("inpaint %dx%d lama %.2fs", rgb.shape[1], rgb.shape[0], time.perf_counter() - t0)
                if refiner is not None:
                    state.progress.set(refiner.name, 0, 0)
                    out = refiner.refine(
                        out, m, strength=strength,
                        progress=lambda step, total: state.progress.set(refiner.name, step, total),
                    )
                    log.info("inpaint %s total %.2fs", refiner.name, time.perf_counter() - t0)
                return ops.encode_png(out)
            finally:
                state.progress.set("")
                state.inpaint_lock.release()

        png = await run_in_threadpool(run_inpaint)
        return Response(png, media_type="image/png")

    @app.exception_handler(Exception)
    async def unhandled(_: Request, err: Exception):
        log.exception("request failed")
        return JSONResponse(status_code=500, content={"detail": str(err)})

    @app.get("/")
    def index():
        return FileResponse(ROOT / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=ROOT, html=True), name="static")
    return app
