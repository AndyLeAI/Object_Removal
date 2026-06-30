# ObjRemover

**Author:** Andy N Le

ObjRemover is a simple local AI object-removal tool powered by LaMa ONNX. It runs directly in your browser and is designed for local use, with no image upload required.

## What’s New in V12

This version fixes the brush opacity display issue:

- Painted mask strokes no longer become visually solid when brush strokes overlap.
- The internal mask remains fully opaque so the AI model can correctly detect the removal area.
- The visible mask canvas is displayed at exactly **40% opacity**.
- The brush preview is synchronized with the painted mask at **40% opacity**.

## Required Files

Before running the app, place the following files inside the `assets/` folder:

```text
assets/lama_fp16.onnx
assets/ort-wasm-simd-threaded.asyncify.wasm
```
## Download lama_fp16.onnx here: [Lama FP16](https://huggingface.co/g-ronimo/lama/resolve/main/lama_fp16.onnx)

## How to Run

### Windows

Double-click:

```text
start_windows.bat
```

Then open:

```text
http://localhost:5177
```

### Manual Start

Run:

```bash
python server.py
```

Then open:

```text
http://localhost:5177
```

## Usage

1. Open or drag and drop an image into the app.
2. Adjust the brush size.
3. Paint over the object or area you want to remove.
4. Click **Remove Object**.
5. Download the result when processing is complete.

## Notes

- The app runs locally in your browser.
- Your images are not uploaded to any server.
- The FP16 LaMa model must be placed locally in the `assets/` folder.
- If the browser still shows an older version, press **Ctrl + F5** to hard refresh.

## Support

For questions or support, please contact:

**Andy N Le**  
Phone: **+84 868 231 181**  
Email: **me@andyle.one**

## GitHub

If this project helps you, please give it a ⭐ star on GitHub.

Thank you for supporting the project.
