import json
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import imageops as ops
from backend.app import create_app
from backend.refiners import Refiner
from backend.tests.test_api import StubInpainter, StubSegmenter, png_rgb, png_rgba_mask


class StubRefiner(Refiner):
    """Paints the hole solid blue at working size; records what it was given."""

    name = "stub"
    label = "Stub"
    working_side = 256
    multiple = 8

    def __init__(self, device="cpu", fail=False):
        super().__init__(device)
        self.calls = []
        self.fail = fail
        self.loaded = True

    def _load(self):
        if self.fail:
            raise RuntimeError("no weights")

    def repaint(self, image, mask, *, strength, progress):
        self.calls.append((image.size, mask.size, strength))
        if progress:
            progress(1, 2)
            progress(2, 2)
        arr = np.asarray(image).copy()
        arr[np.asarray(mask) > 127] = (0, 0, 255)
        return Image.fromarray(arr)


def test_refine_replaces_only_the_hole_and_resizes_back():
    ref = StubRefiner()
    image = np.full((600, 800, 3), 120, np.uint8)
    mask = np.zeros((600, 800), bool)
    mask[250:350, 350:450] = True
    out = ref.refine(image, mask, strength=0.5)
    assert out.shape == image.shape
    assert tuple(out[300, 400]) == (0, 0, 255)
    assert tuple(out[0, 0]) == (120, 120, 120)
    assert tuple(out[599, 799]) == (120, 120, 120)
    (size, msize, strength), = ref.calls
    assert max(size) == 256 and size == msize and strength == 0.5
    assert size[0] % 8 == 0 and size[1] % 8 == 0


def test_refine_requires_loaded_and_nonempty_mask():
    ref = StubRefiner()
    ref.loaded = False
    with pytest.raises(RuntimeError):
        ref.refine(np.zeros((64, 64, 3), np.uint8), np.ones((64, 64), bool))
    ref.loaded = True
    with pytest.raises(ValueError):
        ref.refine(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64), bool))


def test_load_records_error_and_status():
    ref = StubRefiner(fail=True)
    ref.loaded = False
    with pytest.raises(RuntimeError):
        ref.load()
    s = ref.status()
    assert s["loaded"] is False and s["loading"] is False and "no weights" in s["error"]


@pytest.fixture
def client():
    refiners = {"sdxl": StubRefiner(), "flux": StubRefiner()}
    refiners["sdxl"].name, refiners["flux"].name = "sdxl", "flux"
    refiners["flux"].loaded = False
    app = create_app(StubSegmenter(), StubInpainter(), refiners, load_models=False)
    with TestClient(app) as c:
        c.refiners = refiners
        yield c


def files(w=64, h=48):
    return {
        "image": ("i.png", png_rgb(w, h), "image/png"),
        "mask": ("m.png", png_rgba_mask(w, h, (20, 10, 40, 30)), "image/png"),
    }


def test_health_lists_refiners(client):
    body = client.get("/api/health").json()
    assert body["refiners"]["sdxl"]["loaded"] is True
    assert body["refiners"]["flux"]["loaded"] is False
    assert body["refiners"]["flux"]["label"] == "Stub"


def test_inpaint_quality_routing(client):
    r = client.post("/api/inpaint", files=files(), data={"quality": "fast"})
    assert r.status_code == 200
    assert tuple(ops.decode_rgb(r.content)[20, 30]) == (0, 255, 0)  # stub inpainter only

    r = client.post("/api/inpaint", files=files(), data={"quality": "sdxl", "strength": "0.4"})
    assert r.status_code == 200, r.text
    assert tuple(ops.decode_rgb(r.content)[20, 30]) == (0, 0, 255)  # refined
    assert client.refiners["sdxl"].calls[-1][2] == 0.4

    r = client.post("/api/inpaint", files=files(), data={"quality": "flux"})
    assert r.status_code == 503  # not loaded

    r = client.post("/api/inpaint", files=files(), data={"quality": "nope"})
    assert r.status_code == 400
    r = client.post("/api/inpaint", files=files(), data={"quality": "sdxl", "strength": "5"})
    assert r.status_code == 400


def test_progress_resets_after_request(client):
    client.post("/api/inpaint", files=files(), data={"quality": "sdxl"})
    assert client.get("/api/progress").json() == {"stage": "", "step": 0, "total": 0}


def test_load_endpoint_unloads_the_other_refiner(client):
    r = client.post("/api/models/flux/load")
    assert r.status_code == 200
    for t in threading.enumerate():
        if t.name == "load-flux":
            t.join(timeout=5)
    body = client.get("/api/health").json()
    assert body["refiners"]["flux"]["loaded"] is True
    assert body["refiners"]["sdxl"]["loaded"] is False
    assert client.post("/api/models/nope/load").status_code == 404


def test_refiner_load_waits_for_base_models():
    """Model loading is serialized: transformers/diffusers loaders share process-global
    state, and loading SAM and SDXL in parallel threads corrupts one of them."""
    from backend.app import ModelState

    gate = threading.Event()

    class SlowSegmenter(StubSegmenter):
        loaded = False

        def load(self):
            gate.wait(timeout=5)
            self.loaded = True

    ref = StubRefiner()
    ref.loaded = False
    state = ModelState(SlowSegmenter(), StubInpainter(), {"stub": ref})
    base = threading.Thread(target=state.load_all)
    base.start()
    extra = threading.Thread(target=state.load_refiner, args=("stub",))
    extra.start()
    extra.join(timeout=0.5)
    assert extra.is_alive() and ref.loaded is False, "refiner load must wait for load_all"
    gate.set()
    base.join(timeout=5)
    extra.join(timeout=5)
    assert state.segmenter.loaded and ref.loaded and state.error is None
