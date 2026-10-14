import numpy as np
import pytest
from PIL import Image
import io

from backend import imageops as ops


def test_mask_bbox_empty_and_simple():
    assert ops.mask_bbox(np.zeros((10, 10), bool)) is None
    m = np.zeros((10, 12), bool)
    m[2:5, 3:9] = True
    assert ops.mask_bbox(m) == (3, 2, 9, 5)


def test_crop_box_adds_margin_and_clamps():
    box = ops.crop_box_for_mask((500, 500, 520, 520), 2000, 2000)
    x0, y0, x1, y1 = box
    assert x0 <= 500 - 128 and y0 <= 500 - 128
    assert x1 >= 520 + 128 and y1 >= 520 + 128
    assert (x1 - x0) % 8 == 0 and (y1 - y0) % 8 == 0

    # near the corner: clamps to the image and still covers the mask
    box = ops.crop_box_for_mask((0, 0, 10, 10), 300, 200)
    assert box[0] == 0 and box[1] == 0
    assert box[2] <= 300 and box[3] <= 200
    assert box[2] >= 10 and box[3] >= 10


def test_crop_box_uses_fractional_margin_for_big_masks():
    box = ops.crop_box_for_mask((1000, 1000, 1600, 1400), 4000, 4000)
    # longer side is 600 -> margin 300 > 128
    assert box[0] <= 700 and box[2] >= 1900


def test_crop_box_never_exceeds_image():
    box = ops.crop_box_for_mask((0, 0, 50, 50), 50, 50)
    assert box == (0, 0, 50, 50)


def test_fit_within():
    assert ops.fit_within(800, 600, 1280) == (800, 600)
    w, h = ops.fit_within(2560, 1920, 1280)
    assert max(w, h) <= 1280 and w % 8 == 0 and h % 8 == 0
    assert abs(w / h - 2560 / 1920) < 0.02


def test_dilate_grows_square():
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    d = ops.dilate(m, 2)
    assert d.sum() == 25
    assert d[2, 2] and not d[1, 1]
    assert ops.dilate(m, 0).sum() == 1


def test_pad_to_multiple():
    a = np.zeros((13, 21, 3), np.uint8)
    p, (h, w) = ops.pad_to_multiple(a, 8)
    assert p.shape == (16, 24, 3) and (h, w) == (13, 21)
    b = np.zeros((16, 16), np.uint8)
    p, _ = ops.pad_to_multiple(b, 8)
    assert p is b


def test_mask_round_trip_rgba_and_gray():
    rgba = np.zeros((4, 6, 4), np.uint8)
    rgba[1:3, 2:5] = (138, 63, 252, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    m = ops.decode_mask(buf.getvalue(), size=(6, 4))
    assert m.sum() == 6 and m[1, 2] and not m[0, 0]

    gray = ops.encode_png(m)
    m2 = ops.decode_mask(gray)
    assert np.array_equal(m, m2)

    with pytest.raises(ValueError):
        ops.decode_mask(gray, size=(5, 4))


def test_rgb_round_trip():
    rgb = (np.random.default_rng(0).random((5, 7, 3)) * 255).astype(np.uint8)
    assert np.array_equal(ops.decode_rgb(ops.encode_png(rgb)), rgb)


def test_auto_dilate_radius_scales_with_image_and_clamps():
    # ~1% of the longer side, clamped to [lo, hi]
    assert ops.auto_dilate_radius(2724, 2494, frac=0.01, lo=6, hi=40) == 27
    assert ops.auto_dilate_radius(300, 200, frac=0.01, lo=6, hi=40) == 6
    assert ops.auto_dilate_radius(8000, 8000, frac=0.01, lo=6, hi=40) == 40


def test_working_size_limits_crop_and_hole():
    assert ops.working_size(800, 600, 100, 100, max_side=1280, max_hole=512) == (800, 600)
    w, h = ops.working_size(2000, 1000, 1200, 300, max_side=1280, max_hole=512)
    assert max(w, h) <= 1280 and w % 8 == 0
    assert w * 1200 / 2000 <= 512 + 8  # hole scaled down to the limit
    w, h = ops.working_size(1200, 1200, 200, 200, max_side=1280, max_hole=512)
    assert (w, h) == (1200, 1200)


def test_feather_alpha_full_inside_soft_outside():
    m = np.zeros((40, 40), bool)
    m[10:30, 10:30] = True
    a = ops.feather_alpha(m, 3)
    assert a.dtype == np.float32
    assert a[20, 20] == 1.0 and a[10, 10] == 1.0  # inside stays fully masked
    assert 0 < a[9, 20] < 1 and 0 < a[7, 20] < a[9, 20]  # fades outside
    assert a[0, 0] == 0.0
    assert np.array_equal(ops.feather_alpha(m, 0), m.astype(np.float32))


def test_paste_feathered_changes_only_near_mask():
    base = np.full((30, 30, 3), 100, np.uint8)
    fill = np.full((30, 30, 3), 200, np.uint8)
    m = np.zeros((30, 30), bool)
    m[10:20, 10:20] = True
    out = ops.paste_feathered(base, fill, m, 2)
    assert tuple(out[15, 15]) == (200, 200, 200)
    assert 100 < out[9, 15, 0] < 200
    assert tuple(out[0, 0]) == (100, 100, 100)


def test_fit_to_side_scales_up_and_down():
    assert ops.fit_to_side(512, 256, 1024) == (1024, 512)
    w, h = ops.fit_to_side(3000, 2000, 1024, multiple=16)
    assert max(w, h) == 1024 and w % 16 == 0 and h % 16 == 0
