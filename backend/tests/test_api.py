import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import imageops as ops
from backend.app import create_app
from backend.inpainter import LamaInpainter
from backend.segmenter import Point


class StubSegmenter:
    name = "stub-seg"
    loaded = True

    def __init__(self):
        self.calls = []

    def segment(self, key, image, points):
        self.calls.append((key, points))
        h, w = image.shape[:2]
        m = np.zeros((h, w), bool)
        p = points[0]
        m[int(p.y), int(p.x)] = True
        return m


class StubInpainter:
    name = "stub-inp"
    loaded = True

    def inpaint(self, image, mask):
        out = image.copy()
        out[mask] = (0, 255, 0)
        return out


class ConstantModel:
    """Stands in for the TorchScript LaMa: returns the image with the hole filled grey."""

    def __call__(self, x, m):
        return x * (1 - m) + 0.5 * m

    def to(self, device):
        return self

    def eval(self):
        return self


def png_rgb(w, h, value=10):
    arr = np.full((h, w, 3), value, np.uint8)
    return ops.encode_png(arr)


def png_rgba_mask(w, h, box):
    arr = np.zeros((h, w, 4), np.uint8)
    x0, y0, x1, y1 = box
    arr[y0:y1, x0:x1] = (138, 63, 252, 255)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    seg, inp = StubSegmenter(), StubInpainter()
    app = create_app(seg, inp, load_models=False)
    with TestClient(app) as c:
        c.seg, c.inp = seg, inp
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["models"]["segmenter"]["name"] == "stub-seg"


def test_segment_returns_mask_png(client):
    r = client.post(
        "/api/segment",
        files={"image": ("i.png", png_rgb(20, 10), "image/png")},
        data={"points": json.dumps([{"x": 3, "y": 4, "label": 1}, {"x": 5, "y": 5, "label": 0}])},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    mask = ops.decode_mask(r.content)
    assert mask.shape == (10, 20) and mask[4, 3] and mask[0, 0] is not None
    assert mask.sum() > 1  # expanded around the stub's single pixel, clamped by the image edges
    key, pts = client.seg.calls[0]
    assert pts == [Point(3, 4, 1), Point(5, 5, 0)]
    assert len(key) == 64


def test_segment_rejects_bad_points(client):
    img = ("i.png", png_rgb(20, 10), "image/png")
    r = client.post("/api/segment", files={"image": img}, data={"points": "[]"})
    assert r.status_code == 400
    r = client.post("/api/segment", files={"image": img}, data={"points": json.dumps([{"x": 99, "y": 1}])})
    assert r.status_code == 400
    r = client.post("/api/segment", files={"image": img}, data={"points": "nope"})
    assert r.status_code == 400


def test_inpaint_replaces_masked_pixels(client):
    r = client.post(
        "/api/inpaint",
        files={
            "image": ("i.png", png_rgb(20, 10), "image/png"),
            "mask": ("m.png", png_rgba_mask(20, 10, (2, 2, 6, 5)), "image/png"),
        },
    )
    assert r.status_code == 200, r.text
    out = ops.decode_rgb(r.content)
    assert tuple(out[3, 3]) == (0, 255, 0)
    assert tuple(out[0, 0]) == (10, 10, 10)


def test_inpaint_rejects_empty_or_mismatched_mask(client):
    img = ("i.png", png_rgb(20, 10), "image/png")
    r = client.post("/api/inpaint", files={"image": img, "mask": ("m.png", png_rgba_mask(20, 10, (0, 0, 0, 0)), "image/png")})
    assert r.status_code == 400
    r = client.post("/api/inpaint", files={"image": img, "mask": ("m.png", png_rgba_mask(21, 10, (0, 0, 2, 2)), "image/png")})
    assert r.status_code == 400


def test_not_ready_returns_503():
    class NotLoaded(StubSegmenter):
        loaded = False

    app = create_app(NotLoaded(), StubInpainter(), load_models=False)
    with TestClient(app) as c:
        r = c.post("/api/segment", files={"image": ("i.png", png_rgb(4, 4), "image/png")}, data={"points": "[{\"x\":1,\"y\":1}]"})
        assert r.status_code == 503


def test_lama_inpainter_crop_pipeline(tmp_path):
    inp = LamaInpainter(device="cpu", model_path=tmp_path / "x.pt", max_side=64)
    inp._model = ConstantModel()
    rng = np.random.default_rng(1)
    image = (rng.random((300, 400, 3)) * 255).astype(np.uint8)
    mask = np.zeros((300, 400), bool)
    mask[100:140, 150:200] = True
    out = inp.inpaint(image, mask)
    assert out.shape == image.shape
    # masked pixels were replaced (grey-ish after resize), unmasked untouched
    assert np.array_equal(out[0, 0], image[0, 0])
    assert np.array_equal(out[290, 390], image[290, 390])
    assert abs(int(out[120, 175].astype(int).mean()) - 128) < 8
    # pixels just outside the mask but within the dilation band are also replaced
    assert abs(int(out[140, 175].astype(int).mean()) - 128) < 8
    # beyond the dilation band the image is untouched
    assert np.array_equal(out[150, 175], image[150, 175])


def test_lama_inpainter_empty_mask_raises(tmp_path):
    inp = LamaInpainter(device="cpu", model_path=tmp_path / "x.pt")
    inp._model = ConstantModel()
    with pytest.raises(ValueError):
        inp.inpaint(np.zeros((16, 16, 3), np.uint8), np.zeros((16, 16), bool))


def test_segment_mask_is_expanded_so_lama_covers_object_edges(client):
    # The stub marks a single pixel; the API must expand it so the inpainter
    # also covers the object's anti-aliased edge and shadow.
    r = client.post(
        "/api/segment",
        files={"image": ("i.png", png_rgb(2000, 1000), "image/png")},
        data={"points": json.dumps([{"x": 500, "y": 500, "label": 1}])},
    )
    assert r.status_code == 200, r.text
    mask = ops.decode_mask(r.content)
    radius = ops.auto_dilate_radius(2000, 1000)
    assert radius >= 6
    assert mask.sum() == (2 * radius + 1) ** 2
    assert mask[500, 500] and mask[500 - radius, 500 - radius] and not mask[500 - radius - 1, 500]


def test_lama_inpainter_dilation_scales_with_image(tmp_path):
    inp = LamaInpainter(device="cpu", model_path=tmp_path / "x.pt")
    inp._model = ConstantModel()
    image = np.full((2000, 2000, 3), 10, np.uint8)
    mask = np.zeros((2000, 2000), bool)
    mask[1000:1010, 1000:1010] = True
    out = inp.inpaint(image, mask)
    radius = ops.auto_dilate_radius(2000, 2000, frac=inp.dilate_frac, lo=inp.dilate_min, hi=inp.dilate_max)
    assert radius > 3
    assert tuple(out[1010 + radius - 1, 1005]) != (10, 10, 10)
    assert tuple(out[1010 + radius + 1, 1005]) == (10, 10, 10)


def test_cross_site_requests_are_rejected(client):
    """A web page in the user's browser must not be able to drive the local API."""
    img = png_rgb(8, 8)
    pts = json.dumps([{"x": 2, "y": 2}])
    foreign = (
        {"Origin": "https://evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Origin": "null"},
        {"Origin": "http://localhost.evil.example"},
        {"Origin": "http://127.0.0.1.evil.example:5177"},
        {"Origin": "http://localhost:evil"},  # naive "strip the port" parsing let this through
        {"Origin": "http://localhost:5177.evil.example"},
        {"Origin": "http://evil.example@localhost"},
    )
    for headers in foreign:
        r = client.post("/api/segment", files={"image": ("a.png", img, "image/png")}, data={"points": pts}, headers=headers)
        assert r.status_code == 403, headers
        assert client.post("/api/models/sdxl/load", headers=headers).status_code == 403, headers
        # Any mutating request is checked, whatever the path spelling.
        assert client.post("//api/segment", headers=headers).status_code == 403, headers
    # Same-origin and no-Origin (curl) requests still work.
    for headers in ({"Origin": "http://127.0.0.1:5177"}, {"Origin": "http://localhost:5177"}, {"Origin": "http://[::1]:5177"}, {}):
        r = client.post("/api/segment", files={"image": ("a.png", img, "image/png")}, data={"points": pts}, headers=headers)
        assert r.status_code == 200, headers
    assert client.get("/api/health", headers={"Origin": "https://evil.example"}).status_code == 403


def test_oversized_upload_is_rejected(client, monkeypatch):
    from backend import app as app_module

    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 100)
    img = png_rgb(64, 64)
    assert len(img) > 100
    r = client.post("/api/segment", files={"image": ("a.png", img, "image/png")}, data={"points": json.dumps([{"x": 2, "y": 2}])})
    assert r.status_code == 413
    r = client.post("/api/inpaint", files={"image": ("a.png", img, "image/png"), "mask": ("m.png", png_rgba_mask(64, 64, (8, 8, 24, 24)), "image/png")})
    assert r.status_code == 413
