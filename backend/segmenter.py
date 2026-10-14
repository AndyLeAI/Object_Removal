"""Click-to-mask segmentation with Segment Anything 2.1."""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from . import imageops as ops

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "facebook/sam2.1-hiera-base-plus"


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    label: int  # 1 = include, 0 = exclude


class Sam2Segmenter:
    """Wraps a SAM 2.1 model and caches image embeddings by image key."""

    name = "sam2.1"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cpu", cache_size: int = 4):
        self.model_id = model_id
        self.device = device
        self.cache_size = cache_size
        self._model = None
        self._processor = None
        self._cache: OrderedDict[str, tuple[list[torch.Tensor], tuple[int, int]]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from transformers import Sam2Model, Sam2Processor

        log.info("loading %s on %s", self.model_id, self.device)
        self._processor = Sam2Processor.from_pretrained(self.model_id)
        self._model = Sam2Model.from_pretrained(self.model_id).to(self.device).eval()
        log.info("segmenter ready")

    def _embed(self, key: str, image: np.ndarray) -> tuple[list[torch.Tensor], tuple[int, int]]:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        pil = Image.fromarray(image)
        inputs = self._processor(images=pil, return_tensors="pt")
        with torch.no_grad():
            emb = self._model.get_image_embeddings(inputs["pixel_values"].to(self.device))
        size = (image.shape[0], image.shape[1])
        self._cache[key] = (emb, size)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return emb, size

    def segment(self, key: str, image: np.ndarray, points: list[Point]) -> np.ndarray:
        """Return a bool mask (H, W) for the object described by `points`."""
        if not self.loaded:
            raise RuntimeError("segmenter not loaded")
        if not points:
            raise ValueError("at least one point is required")
        with self._lock:
            emb, (h, w) = self._embed(key, image)
            coords = [[[[float(p.x), float(p.y)] for p in points]]]
            labels = [[[int(p.label) for p in points]]]
            prompts = self._processor(
                original_sizes=[[h, w]], input_points=coords, input_labels=labels, return_tensors="pt"
            )
            with torch.no_grad():
                out = self._model(
                    image_embeddings=emb,
                    input_points=prompts["input_points"].to(self.device),
                    input_labels=prompts["input_labels"].to(self.device),
                    multimask_output=len(points) == 1,
                )
            masks = self._processor.post_process_masks(out.pred_masks.cpu(), [[h, w]])[0]  # (P, M, H, W)
            scores = out.iou_scores[0, 0].cpu()
            best = int(torch.argmax(scores))
            mask = masks[0, best].numpy().astype(bool)
        return mask
