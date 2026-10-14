"""Pure image helpers shared by the segmenter, the inpainter and the API."""
from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 (x1/y1 exclusive)


def decode_rgb(data: bytes) -> np.ndarray:
    """PNG/JPEG bytes -> uint8 array of shape (H, W, 3)."""
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def decode_mask(data: bytes, size: tuple[int, int] | None = None) -> np.ndarray:
    """Mask image bytes -> bool array (H, W).

    Accepts the RGBA mask canvas the UI exports (alpha > 0 means masked) as
    well as plain grayscale masks (value > 127 means masked).
    """
    with Image.open(io.BytesIO(data)) as im:
        if im.mode in ("RGBA", "LA", "PA"):
            arr = np.asarray(im.convert("RGBA"))[..., 3] > 0
        else:
            arr = np.asarray(im.convert("L")) > 127
    if size is not None and arr.shape != (size[1], size[0]):
        raise ValueError(f"mask size {arr.shape[::-1]} does not match image size {size}")
    return arr


def encode_png(arr: np.ndarray) -> bytes:
    """uint8 (H,W,3) or bool/uint8 (H,W) -> PNG bytes."""
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def mask_bbox(mask: np.ndarray) -> Box | None:
    """Tight bounding box of the true pixels, or None for an empty mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def round_up(value: int, multiple: int) -> int:
    return value if value % multiple == 0 else (value // multiple + 1) * multiple


def crop_box_for_mask(
    bbox: Box,
    width: int,
    height: int,
    *,
    margin_min: int = 128,
    margin_frac: float = 0.5,
    multiple: int = 8,
) -> Box:
    """Expand a mask bbox with context margin, clamped to the image.

    The margin is the larger of ``margin_min`` pixels and ``margin_frac`` of the
    box's longer side, so small objects get plenty of context and large ones
    still fit. The resulting box is grown to a multiple of ``multiple`` when
    the image allows it.
    """
    x0, y0, x1, y1 = bbox
    longer = max(x1 - x0, y1 - y0)
    margin = max(margin_min, int(round(longer * margin_frac)))
    cx0 = max(0, x0 - margin)
    cy0 = max(0, y0 - margin)
    cx1 = min(width, x1 + margin)
    cy1 = min(height, y1 + margin)

    # Grow to multiples of `multiple` where possible, preferring to extend
    # outward rather than shrinking the context.
    want_w = min(width, round_up(cx1 - cx0, multiple))
    want_h = min(height, round_up(cy1 - cy0, multiple))
    cx1 = min(width, cx0 + want_w)
    cx0 = max(0, cx1 - want_w)
    cy1 = min(height, cy0 + want_h)
    cy0 = max(0, cy1 - want_h)
    return cx0, cy0, cx1, cy1


def fit_within(width: int, height: int, limit: int, multiple: int = 8) -> tuple[int, int]:
    """Scale (width, height) down so the longer side <= limit; keep multiples of 8."""
    scale = min(1.0, limit / max(width, height))
    w = max(multiple, int(round(width * scale / multiple)) * multiple)
    h = max(multiple, int(round(height * scale / multiple)) * multiple)
    return w, h


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a (2r+1)^2 square structuring element."""
    if radius <= 0:
        return mask.astype(bool)
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    k = 2 * radius + 1
    # A square structuring element is separable: two 1-D max filters.
    out = F.max_pool2d(t, kernel_size=(1, k), stride=1, padding=(0, radius))
    out = F.max_pool2d(out, kernel_size=(k, 1), stride=1, padding=(radius, 0))
    return out[0, 0].numpy() > 0.5


def pad_to_multiple(arr: np.ndarray, multiple: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad the bottom/right of (H,W[,C]) so H and W are multiples of `multiple`."""
    h, w = arr.shape[:2]
    ph, pw = round_up(h, multiple) - h, round_up(w, multiple) - w
    if ph == 0 and pw == 0:
        return arr, (h, w)
    pad = [(0, ph), (0, pw)] + [(0, 0)] * (arr.ndim - 2)
    mode = "reflect" if (ph < h and pw < w) else "edge"
    return np.pad(arr, pad, mode=mode), (h, w)


def resize_rgb(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize(size, Image.LANCZOS))


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    im = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)
    return np.asarray(im) > 127


def auto_dilate_radius(width: int, height: int, *, frac: float = 0.01, lo: int = 6, hi: int = 40) -> int:
    """Dilation radius as a fraction of the longer image side, clamped to [lo, hi].

    Segmentation masks hug the object's pixels; LaMa needs the mask to also cover
    the anti-aliased edge, shadow and colour bleed around it, otherwise it fills the
    hole with those boundary colours and leaves a faded silhouette.
    """
    return int(min(hi, max(lo, round(max(width, height) * frac))))


def working_size(
    width: int, height: int, hole_w: int, hole_h: int, *, max_side: int, max_hole: int, multiple: int = 8
) -> tuple[int, int]:
    """Inference size for a crop: longer side <= max_side and the hole's longer side <= max_hole."""
    scale = min(1.0, max_side / max(width, height), max_hole / max(1, hole_w, hole_h))
    w = max(multiple, int(round(width * scale / multiple)) * multiple)
    h = max(multiple, int(round(height * scale / multiple)) * multiple)
    return w, h


def feather_alpha(mask: np.ndarray, radius: int) -> np.ndarray:
    """Soft (float32, 0..1) version of a binary mask: 1 inside, falling to 0 over ~radius px outside."""
    m = mask.astype(np.float32)
    if radius <= 0:
        return m
    t = torch.from_numpy(m)[None, None]
    k = 2 * radius + 1
    for _ in range(2):  # two box blurs ~ a triangle kernel
        t = F.avg_pool2d(t, kernel_size=k, stride=1, padding=radius, count_include_pad=True)
    blurred = t[0, 0].numpy()
    return np.maximum(m, blurred)  # keep full strength inside the mask


def paste_feathered(base: np.ndarray, fill: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Blend `fill` into `base` where `mask` is set, with a feathered seam. All (H,W[,3])."""
    alpha = feather_alpha(mask, radius)[..., None]
    out = base.astype(np.float32) * (1 - alpha) + fill.astype(np.float32) * alpha
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def fit_to_side(width: int, height: int, side: int, multiple: int = 8) -> tuple[int, int]:
    """Resize (up or down) so the longer side == side, keeping aspect, multiples of `multiple`."""
    scale = side / max(width, height)
    w = max(multiple, int(round(width * scale / multiple)) * multiple)
    h = max(multiple, int(round(height * scale / multiple)) * multiple)
    return w, h
