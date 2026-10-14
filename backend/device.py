import os

import torch


def pick_device() -> str:
    """MPS on Apple Silicon, CUDA when present, CPU otherwise. OBJREMOVER_DEVICE overrides."""
    forced = os.environ.get("OBJREMOVER_DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
