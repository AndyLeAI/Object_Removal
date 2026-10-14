# Zoom, smart selection, stronger inpainting, brush fix — design

Date: 2026-10-13

## Goals (from the request)

1. Zoom into the image while painting.
2. Smart selection: click an object and an AI model produces the mask.
3. Replace the inpainting model with a more powerful one that an Apple M4 can run.
4. Fix the mismatch between the brush-size preview and the painted stroke.

## Current state

Static HTML/JS app served by `server.py`. A web worker runs LaMa fp16 ONNX
(ONNX Runtime Web, WASM). The whole image is resized to 512x512, inpainted,
and the masked pixels are upscaled back. The brush preview is a fixed DOM
circle sized in CSS pixels while the stroke is sized in image pixels.

## Decisions

### Architecture: local Python backend, same UI

Browser WASM cannot run a segmentation model plus full-resolution inpainting
at usable speed. A FastAPI backend (`backend/`) runs PyTorch on Metal (MPS)
with CPU fallback and serves the static UI. Everything stays on localhost.

Endpoints (all multipart/JSON, binary PNG responses):

- `GET  /api/health` -> `{device, models:{segmenter, inpainter}, ready}`
- `POST /api/segment` image.png + `points=[{x,y,label}]` -> mask.png (L, 0/255)
- `POST /api/inpaint` image.png + mask.png -> result.png (RGB)

The backend caches SAM image embeddings keyed by the SHA-256 of the uploaded
image bytes (small LRU), so repeated clicks on the same image are fast.

### Segmentation model

Segment Anything 2.1 (`facebook/sam2.1-hiera-base-plus`) via `transformers`.
Positive click = new object; shift-click = negative point refining the most
recent object. The returned mask is dilated by a few pixels so LaMa has margin.

### Inpainting model

Big-LaMa (TorchScript, fp32) at native resolution with a crop strategy:
bounding box of the mask, padded by max(128 px, 50% of the box), clamped to
the image, sides rounded to multiples of 8. If the crop's longer side exceeds
1280 px it is downscaled for inference and the result is upscaled back; only
masked pixels are replaced. This replaces the 512 px whole-image downscale.

Rationale: LaMa remains the most reliable pure object-removal model that runs
in seconds on an M4; diffusion inpainters are larger, slower, and tend to
hallucinate new objects. The inpainter is behind a small interface so a
diffusion model can be added later.

### Zoom and pan

Canvases live in a `#canvasWrap` element inside a fixed-size viewport and are
positioned with `translate(tx, ty) scale(s)`. Wheel/pinch zooms around the
cursor; space+drag or middle-drag pans; toolbar has fit, 100%, +, -. Pointer
to image mapping continues to use `getBoundingClientRect()` which already
reflects the transform.

### Brush preview

Preview diameter = brush size in image px * (on-screen canvas width / image
width). Same rule for the cursor circle at any zoom level.

### Frontend structure

`app.js` is split into ES modules: `viewport.js` (zoom/pan math), `api.js`
(backend client), `app.js` (UI wiring). Pure math lives in `viewport.js` and
is tested with `node --test`.

### Removed

`worker.js`, the ONNX Runtime WASM bundle and the fp16 ONNX requirement.

## Testing

- Python: pytest for image ops (crop box, dilation, mask round-trip) and API
  tests with stub models.
- JS: `node --test` for viewport math and brush scale.
- Manual: run the server, load an image, zoom, smart-select, remove.

## Addendum (2026-10-13): quality tiers

Request: make the fill "content aware" with SDXL inpainting and FLUX.1 Fill as
options, plus the LaMa quick wins.

### Quality tiers (`quality` form field on `/api/inpaint`)

| Tier | Pipeline | Working size | Typical time on M4 Pro |
| --- | --- | --- | --- |
| `fast` | Big-LaMa only | crop <= 1280 px, hole <= 512 px | seconds |
| `sdxl` | LaMa, then SDXL inpainting refines the hole (strength 0.6) | crop resized to 1024 px | ~30-60 s |
| `flux` | LaMa, then FLUX.1 Fill regenerates the hole | crop resized to <= 1024 px | minutes |

LaMa always runs first so the diffusion models start from a coherent fill
(SDXL uses it as the init image; FLUX Fill is conditioned on the masked image
and is told "empty background" by the prompt). Only the masked pixels are
pasted back, through a feathered alpha so the seam is invisible.

### Models

- SDXL: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, fp16 (~7 GB).
- FLUX.1 Fill dev: transformer as GGUF Q4_K_S from
  `YarvixPA/FLUX.1-Fill-dev-GGUF` (6.8 GB); T5-XXL encoder as GGUF Q5_K_M from
  `city96/t5-v1_1-xxl-encoder-gguf` (3.4 GB); CLIP-L from
  `openai/clip-vit-large-patch14`; VAE from `ostris/Flex.1-alpha` (the FLUX
  VAE, Apache-2.0). All ungated. The prompt is constant, so its T5/CLIP
  embeddings are computed once and cached to `models/flux_prompt_embeds.pt`;
  the text encoders are then freed.
- FLUX.1 [dev] weights are under the FLUX.1 [dev] Non-Commercial License.

### Memory

24 GB unified memory. Only one diffusion refiner is resident at a time;
selecting the other unloads it and calls `torch.mps.empty_cache()`.

### Loading UX

Refiners load lazily. The UI's Quality control calls
`POST /api/models/{name}/load`; `/api/health` reports
`refiners[name] = {loaded, loading, error}` and the Remove button waits until
the chosen tier is ready. First use downloads the weights.

### LaMa quick wins (all tiers)

- Working resolution: scale the crop so its longer side <= 1280 px and the
  mask's longer side <= 512 px (LaMa gets mushy on large holes at full res).
- Feathered paste: masked pixels are blended back through a blurred alpha
  (radius ~0.5 % of the crop) instead of a hard edge.
