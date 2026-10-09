"""Pick the torch device SAM 2 and ViTMatte run on."""

import os

# SAM 2 uses a few ops Apple's MPS backend lacks; let those fall back to the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("no CUDA or Apple MPS GPU found; pass --device cpu to run on the CPU (expect it to be very slow)")


def free_memory(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
