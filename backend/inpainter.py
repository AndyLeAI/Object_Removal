"""Object removal with Big-LaMa (TorchScript) on a crop around the mask."""
from __future__ import annotations

import logging
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
import torch

from . import imageops as ops

log = logging.getLogger(__name__)

BIG_LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
BIG_LAMA_MIN_BYTES = 150 * 1024 * 1024
MODELS_DIR = Path(os.environ.get("OBJREMOVER_MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))


class LamaInpainter:
    name = "big-lama"

    def __init__(
        self,
        device: str = "cpu",
        model_path: Path | None = None,
        *,
        max_side: int = 1280,
        max_hole: int = 640,
        feather_frac: float = 0.005,
        dilate_frac: float = 0.004,
        dilate_min: int = 3,
        dilate_max: int = 16,
        margin_min: int = 128,
        margin_frac: float = 0.5,
    ):
        self.device = device
        self.model_path = Path(model_path) if model_path else MODELS_DIR / "big-lama.pt"
        self.max_side = max_side
        self.max_hole = max_hole
        self.feather_frac = feather_frac
        self.dilate_frac = dilate_frac
        self.dilate_min = dilate_min
        self.dilate_max = dilate_max
        self.margin_min = margin_min
        self.margin_frac = margin_frac
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def ensure_weights(self) -> Path:
        if self.model_path.exists() and self.model_path.stat().st_size >= BIG_LAMA_MIN_BYTES:
            return self.model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.model_path.with_suffix(".part")
        log.info("downloading Big-LaMa weights to %s", self.model_path)
        urllib.request.urlretrieve(BIG_LAMA_URL, tmp)
        if tmp.stat().st_size < BIG_LAMA_MIN_BYTES:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("Big-LaMa download looks truncated")
        tmp.replace(self.model_path)
        return self.model_path

    def load(self) -> None:
        path = self.ensure_weights()
        log.info("loading %s on %s", path, self.device)
        self._model = torch.jit.load(str(path), map_location="cpu").to(self.device).eval()
        log.info("inpainter ready")

    def _run(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """image uint8 (H,W,3), mask bool (H,W); both sides multiples of 8. Returns uint8 (H,W,3)."""
        x = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0)[None]
        m = torch.from_numpy(mask.astype(np.float32))[None, None]
        try:
            with torch.no_grad():
                y = self._model(x.to(self.device), m.to(self.device))
        except RuntimeError as err:
            if self.device == "cpu":
                raise
            log.warning("inference failed on %s (%s); falling back to cpu", self.device, err)
            self.device = "cpu"
            self._model = self._model.to("cpu")
            with torch.no_grad():
                y = self._model(x, m)
        y = y[0].clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return y

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self.loaded:
            raise RuntimeError("inpainter not loaded")
        if image.shape[:2] != mask.shape:
            raise ValueError("image and mask sizes differ")
        h, w = mask.shape
        radius = ops.auto_dilate_radius(w, h, frac=self.dilate_frac, lo=self.dilate_min, hi=self.dilate_max)
        mask = ops.dilate(mask, radius)
        bbox = ops.mask_bbox(mask)
        if bbox is None:
            raise ValueError("mask is empty")

        x0, y0, x1, y1 = ops.crop_box_for_mask(
            bbox, w, h, margin_min=self.margin_min, margin_frac=self.margin_frac
        )
        crop = image[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        ch, cw = crop_mask.shape

        hole_w, hole_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        inf_w, inf_h = ops.working_size(cw, ch, hole_w, hole_h, max_side=self.max_side, max_hole=self.max_hole)
        scaled = (inf_w, inf_h) != (cw, ch)
        if scaled:
            work = ops.resize_rgb(crop, (inf_w, inf_h))
            work_mask = ops.resize_mask(crop_mask, (inf_w, inf_h))
        else:
            work, work_mask = crop, crop_mask

        work, (uh, uw) = ops.pad_to_multiple(work, 8)
        work_mask, _ = ops.pad_to_multiple(work_mask, 8)

        with self._lock:
            out = self._run(work, work_mask)[:uh, :uw]

        if scaled:
            out = ops.resize_rgb(out, (cw, ch))

        result = image.copy()
        radius = max(2, int(round(max(cw, ch) * self.feather_frac)))
        result[y0:y1, x0:x1] = ops.paste_feathered(crop, out, crop_mask, radius)
        return result
