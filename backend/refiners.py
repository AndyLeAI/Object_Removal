"""Diffusion refiners that repaint the hole after LaMa: SDXL inpainting and FLUX.1 Fill."""
from __future__ import annotations

import gc
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from . import imageops as ops

log = logging.getLogger(__name__)

PROMPT = (
    "empty background, seamless continuation of the surrounding scene, nothing there, "
    "photorealistic, highly detailed, sharp focus, natural lighting"
)
NEGATIVE_PROMPT = "object, person, animal, text, watermark, logo, blurry, artifacts, distorted, deformed"

SDXL_MODEL_ID = os.environ.get("OBJREMOVER_SDXL_MODEL", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
FLUX_GGUF_REPO = os.environ.get("OBJREMOVER_FLUX_GGUF_REPO", "YarvixPA/FLUX.1-Fill-dev-GGUF")
FLUX_GGUF_FILE = os.environ.get("OBJREMOVER_FLUX_GGUF_FILE", "flux1-fill-dev-Q4_K_S.gguf")
FLUX_T5_REPO = "city96/t5-v1_1-xxl-encoder-gguf"
FLUX_T5_FILE = os.environ.get("OBJREMOVER_FLUX_T5_FILE", "t5-v1_1-xxl-encoder-Q5_K_M.gguf")
FLUX_T5_TOKENIZER = "google/t5-v1_1-xxl"
FLUX_CLIP_REPO = "openai/clip-vit-large-patch14"
FLUX_VAE_REPO = "ostris/Flex.1-alpha"  # Apache-2.0 copy of the FLUX VAE
FLUX_TRANSFORMER_CONFIG = Path(__file__).resolve().parent / "configs" / "flux-fill-transformer"
MODELS_DIR = Path(os.environ.get("OBJREMOVER_MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))

ProgressFn = Callable[[int, int], None]


@dataclass
class Progress:
    """Shared progress of the running removal, polled by the UI."""

    stage: str = ""
    step: int = 0
    total: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, stage: str, step: int = 0, total: int = 0) -> None:
        with self._lock:
            self.stage, self.step, self.total = stage, step, total

    def snapshot(self) -> dict:
        with self._lock:
            return {"stage": self.stage, "step": self.step, "total": self.total}


def free_memory(device: str) -> None:
    gc.collect()
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


class Refiner:
    """Base class: repaint the masked area of an image crop at `working_side` resolution."""

    name = "refiner"
    label = "Refiner"
    download_gb = 0.0
    working_side = 1024
    multiple = 8
    margin_frac = 0.75
    margin_min = 96

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.loaded = False
        self.loading = False
        self.error: str | None = None
        self._lock = threading.Lock()

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        if self.loaded:
            return
        self.loading, self.error = True, None
        try:
            self._load()
            self.loaded = True
        except Exception as err:  # noqa: BLE001 - surfaced to the UI
            log.exception("failed to load %s", self.name)
            self.error = str(err)
            self._unload()
            raise
        finally:
            self.loading = False

    def unload(self) -> None:
        if not self.loaded:
            return
        self._unload()
        self.loaded = False
        free_memory(self.device)

    def status(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "download_gb": self.download_gb,
            "loaded": self.loaded,
            "loading": self.loading,
            "error": self.error,
        }

    def _load(self) -> None:  # pragma: no cover - heavy
        raise NotImplementedError

    def _unload(self) -> None:  # pragma: no cover
        pass

    # --- inference -------------------------------------------------------
    def repaint(self, image: Image.Image, mask: Image.Image, *, strength: float, progress: ProgressFn | None) -> Image.Image:
        """Repaint the white area of `mask` in `image` (both working-size PIL). Returns RGB PIL."""
        raise NotImplementedError

    def refine(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        strength: float = 0.6,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        """Full-size uint8 image (already LaMa-filled) + bool mask -> full-size uint8 image."""
        if not self.loaded:
            raise RuntimeError(f"{self.name} is not loaded")
        h, w = mask.shape
        bbox = ops.mask_bbox(mask)
        if bbox is None:
            raise ValueError("mask is empty")
        x0, y0, x1, y1 = ops.crop_box_for_mask(
            bbox, w, h, margin_min=self.margin_min, margin_frac=self.margin_frac, multiple=self.multiple
        )
        crop = image[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        ch, cw = crop_mask.shape
        ww, wh = ops.fit_to_side(cw, ch, self.working_side, self.multiple)
        pil_img = Image.fromarray(crop).resize((ww, wh), Image.LANCZOS)
        pil_mask = Image.fromarray(crop_mask.astype(np.uint8) * 255).resize((ww, wh), Image.NEAREST)
        with self._lock:
            out = self.repaint(pil_img, pil_mask, strength=strength, progress=progress)
        out = np.asarray(out.convert("RGB").resize((cw, ch), Image.LANCZOS))
        radius = max(2, int(round(max(cw, ch) * 0.006)))
        result = image.copy()
        result[y0:y1, x0:x1] = ops.paste_feathered(crop, out, crop_mask, radius)
        return result


class SdxlRefiner(Refiner):
    name = "sdxl"
    label = "HQ · SDXL inpainting"
    download_gb = 7.0
    working_side = 1024
    multiple = 8
    steps = 30
    guidance = 6.0

    def __init__(self, device: str = "cpu", model_id: str = SDXL_MODEL_ID):
        super().__init__(device)
        self.model_id = model_id
        self._pipe = None

    def _load(self) -> None:
        from diffusers import AutoPipelineForInpainting

        dtype = torch.float16 if self.device != "cpu" else torch.float32
        log.info("loading %s (%s) on %s", self.model_id, dtype, self.device)
        pipe = AutoPipelineForInpainting.from_pretrained(self.model_id, torch_dtype=dtype, variant="fp16")
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe.to(self.device)
        log.info("sdxl refiner ready")

    def _unload(self) -> None:
        self._pipe = None

    def repaint(self, image, mask, *, strength, progress):
        steps = self.steps
        total = max(1, int(round(steps * strength)))

        def on_step(pipe, i, t, kwargs):
            if progress:
                progress(i + 1, total)
            return kwargs

        gen = torch.Generator("cpu").manual_seed(0)
        out = self._pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=image,
            mask_image=mask,
            width=image.width,
            height=image.height,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=self.guidance,
            generator=gen,
            callback_on_step_end=on_step,
        )
        return out.images[0]


class FluxFillRefiner(Refiner):
    name = "flux"
    label = "Ultra · FLUX.1 Fill"
    download_gb = 12.0
    # GGUF weights are dequantized on every step, so on an M4 Pro a step costs
    # ~15 s at 768 px and ~22 s at 1024 px. 768 px / 20 steps keeps a removal near 5 minutes.
    working_side = int(os.environ.get("OBJREMOVER_FLUX_SIDE", "768"))
    multiple = 16
    steps = int(os.environ.get("OBJREMOVER_FLUX_STEPS", "20"))
    guidance = 30.0

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._pipe = None
        self.embeds_path = MODELS_DIR / "flux_prompt_embeds.pt"
        self._embeds: tuple[torch.Tensor, torch.Tensor] | None = None

    def _prompt_embeds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """T5 + CLIP embeddings of the constant prompt, computed once and cached on disk."""
        if self.embeds_path.exists():
            data = torch.load(self.embeds_path, map_location="cpu")
            return data["prompt_embeds"], data["pooled_prompt_embeds"]
        from diffusers import FluxFillPipeline
        from huggingface_hub import hf_hub_download
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

        log.info("encoding the FLUX prompt once (loads T5-XXL, this takes a minute)")
        t5_path = hf_hub_download(FLUX_T5_REPO, FLUX_T5_FILE)
        text_encoder = CLIPTextModel.from_pretrained(FLUX_CLIP_REPO, torch_dtype=torch.bfloat16)
        tokenizer = CLIPTokenizer.from_pretrained(FLUX_CLIP_REPO)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            FLUX_T5_REPO, gguf_file=FLUX_T5_FILE, torch_dtype=torch.bfloat16
        )
        tokenizer_2 = T5TokenizerFast.from_pretrained(FLUX_T5_TOKENIZER)
        encoder = FluxFillPipeline(
            scheduler=None, vae=None, text_encoder=text_encoder, tokenizer=tokenizer,
            text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2, transformer=None,
        )
        with torch.no_grad():
            prompt_embeds, pooled, _ = encoder.encode_prompt(
                prompt=PROMPT, prompt_2=None, device="cpu", max_sequence_length=512
            )
        del encoder, text_encoder, text_encoder_2
        gc.collect()
        self.embeds_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled, "prompt": PROMPT}, self.embeds_path)
        log.info("cached prompt embeddings to %s (t5 file %s)", self.embeds_path, t5_path)
        return prompt_embeds, pooled

    def _load(self) -> None:
        from diffusers import AutoencoderKL, FluxFillPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
        from diffusers import FlowMatchEulerDiscreteScheduler
        from huggingface_hub import hf_hub_download

        self._embeds = self._prompt_embeds()
        gguf_path = hf_hub_download(FLUX_GGUF_REPO, FLUX_GGUF_FILE)
        log.info("loading FLUX.1 Fill transformer from %s on %s", gguf_path, self.device)
        transformer = FluxTransformer2DModel.from_single_file(
            gguf_path,
            config=str(FLUX_TRANSFORMER_CONFIG),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        vae = AutoencoderKL.from_pretrained(FLUX_VAE_REPO, subfolder="vae", torch_dtype=torch.bfloat16)
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
            base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096,
        )
        pipe = FluxFillPipeline(
            scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None,
            text_encoder_2=None, tokenizer_2=None, transformer=transformer,
        )
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe.to(self.device)
        log.info("flux refiner ready")

    def _unload(self) -> None:
        self._pipe = None

    def repaint(self, image, mask, *, strength, progress):
        prompt_embeds, pooled = self._embeds
        dtype = self._pipe.transformer.dtype

        def on_step(pipe, i, t, kwargs):
            if progress:
                progress(i + 1, self.steps)
            return kwargs

        gen = torch.Generator("cpu").manual_seed(0)
        out = self._pipe(
            prompt_embeds=prompt_embeds.to(self.device, dtype),
            pooled_prompt_embeds=pooled.to(self.device, dtype),
            image=image,
            mask_image=mask,
            width=image.width,
            height=image.height,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=gen,
            max_sequence_length=512,
            callback_on_step_end=on_step,
        )
        return out.images[0]


def make_refiners(device: str) -> dict[str, Refiner]:
    return {r.name: r for r in (SdxlRefiner(device), FluxFillRefiner(device))}
