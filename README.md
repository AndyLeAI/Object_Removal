# ObjRemover

**Author:** Andy N Le

ObjRemover is a local AI object-removal tool. Click an object to select it
(Segment Anything 2.1), or paint over it with the brush, then let Big-LaMa
rebuild the area. Everything runs on your own machine; images never leave it.

## What's new in V20

- **Smart select**: click an object and the mask is generated for you.
  Shift+click to exclude an area, click again on a new object to add it.
- **Zoom and pan**: scroll or pinch to zoom around the cursor, hold Space
  (or use the middle mouse button) and drag to pan, `0` fits, `1` is 100%.
- **Stronger inpainting**: Big-LaMa runs at full resolution on a crop around
  the mask instead of downscaling the whole image to 512 px. Smart-select
  masks are expanded by about 1% of the image size so the object's edge
  halo and shadow are removed too, instead of leaving a faded silhouette.
- **Quality tiers**: *Fast* (Big-LaMa, seconds), *HQ* (LaMa + SDXL
  inpainting refinement, about a minute) and *Ultra* (LaMa + FLUX.1 Fill,
  several minutes). HQ and Ultra download their weights on first use.
- **Brush fix**: the brush preview now matches the painted stroke at every
  zoom level.
- Inference moved from browser WASM to a local Python backend using PyTorch
  on Apple Metal (MPS), CUDA, or CPU.

## Requirements

- Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/) (`brew install uv`
  on macOS, or `curl -LsSf https://astral.sh/uv/install.sh | sh`).
- About 1 GB of disk for the models (downloaded on first start into `models/`
  and the Hugging Face cache).
- Apple Silicon Mac (tested on M4 Pro), or any machine with a CUDA GPU or a
  reasonably fast CPU.

## How to run

### macOS / Linux

```bash
./start_mac.sh
```

### Windows

Double-click `start_windows.bat`.

### Manual start

```bash
uv sync
uv run python server.py
```

Then open <http://localhost:5177>. The first start downloads the models
(~500 MB); the status line in the app shows progress.

Environment variables: `OBJREMOVER_PORT` (default 5177), `OBJREMOVER_DEVICE`
(`mps`, `cuda` or `cpu`; auto-detected), `OBJREMOVER_MODELS_DIR`,
`OBJREMOVER_NO_BROWSER=1` to skip opening a browser tab.

## Usage

1. Open or drag and drop an image into the app.
2. **Smart select** (default): click the object you want gone. Shift+click
   on areas that should stay. Click another object to add it to the mask.
3. **Brush** (`B`): paint the mask by hand; `[` and `]` change the size.
4. Zoom with the scroll wheel or pinch, pan with Space+drag.
5. Click **Remove Object**. Undo with `⌘Z` / `Ctrl+Z`.
6. Download the result.

## Models

| Task | Model | Source |
| --- | --- | --- |
| Segmentation | SAM 2.1 Hiera base-plus | `facebook/sam2.1-hiera-base-plus` on Hugging Face |
| Inpainting (Fast) | Big-LaMa (TorchScript, fp32) | github.com/Sanster/models release `add_big_lama` |
| HQ refinement | SDXL inpainting 0.1, fp16 (~7 GB) | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` |
| Ultra refinement | FLUX.1 Fill dev, GGUF Q4_K_S (~7 GB) + T5-XXL GGUF (~3.4 GB) + CLIP-L + VAE | `YarvixPA/FLUX.1-Fill-dev-GGUF`, `city96/t5-v1_1-xxl-encoder-gguf`, `openai/clip-vit-large-patch14`, `ostris/Flex.1-alpha` |

The HQ and Ultra tiers always run Big-LaMa first and then repaint the hole
with the diffusion model, so the result stays coherent while gaining detail.
Only one diffusion model is kept in memory at a time; switching tiers unloads
the other. FLUX.1 [dev] weights are released under the FLUX.1 [dev]
Non-Commercial License. Make sure you have about 20 GB of free disk space
before enabling both tiers.

Both load automatically on first start. The segmentation model can be
swapped by editing `DEFAULT_MODEL_ID` in `backend/segmenter.py` (for example
`facebook/sam2.1-hiera-large` for the best masks, or `-small` for speed).

## Development

```bash
uv sync --group dev
uv run pytest            # backend tests
node --test js/tests/*.test.js   # viewport math tests
```

## Notes

- Images are processed on localhost only; the browser talks to the local
  backend on the same machine.
- Large images are capped at 3072 px on the longer side when loaded.
- If the browser shows an older version, hard refresh (`Cmd+Shift+R` /
  `Ctrl+F5`).

## Support

For questions or support, please contact:

**Andy N Le**
Phone: **+84 868 231 181**
Email: **me@andyle.one**

## GitHub

If this project helps you, please give it a ⭐ star on GitHub.

Thank you for supporting the project.
