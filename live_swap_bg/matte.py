"""Refine each SAM 2 mask into a soft alpha with ViTMatte."""

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .clip import HEIGHT, WIDTH, Clip
from .device import pick_device

VITMATTE_DIR = "vitmatte-base-distinctions-646"
TRIMAP_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
THIN_CORE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def trimap(mask: np.ndarray, *, thin: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (trimap, sure foreground, support). Eroding by 9 px wipes out parts thinner than that (a pump
    nozzle, a spray head, a brush tip); thin=True keeps their core as sure foreground."""
    mask = mask.astype(np.uint8) * 255
    foreground = cv2.erode(mask, TRIMAP_KERNEL) == 255
    if thin:
        thin_parts = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.morphologyEx(mask, cv2.MORPH_OPEN, TRIMAP_KERNEL)))
        foreground |= cv2.erode(thin_parts, THIN_CORE_KERNEL) > 0
    support = cv2.dilate(mask, TRIMAP_KERNEL) > 0
    tri = np.zeros(mask.shape, np.uint8)
    tri[support] = 128
    tri[foreground] = 255
    return tri, foreground, support


def matte(clip: Clip, models: Path, *, thin: bool = False, device: str | None = None) -> dict:
    import torch
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    if (clip.root / "matting.json").exists():
        raise FileExistsError(f"{clip.root}/matting.json exists; delete it and alpha/ to matte again")
    count = clip.info["frames"]
    device = pick_device(device)
    processor = VitMatteImageProcessor.from_pretrained(models / VITMATTE_DIR, local_files_only=True)
    model = VitMatteForImageMatting.from_pretrained(models / VITMATTE_DIR, local_files_only=True).eval().to(device)
    clip.alpha_dir.mkdir(exist_ok=True)
    started = time.time()
    with torch.inference_mode():
        for index in range(count):
            image = Image.open(clip.frames_dir / f"{index:04d}.png").convert("RGB")
            tri, foreground, support = trimap(clip.mask(index), thin=thin)
            inputs = processor(images=image, trimaps=Image.fromarray(tri), return_tensors="pt").to(device)
            alpha = model(**inputs).alphas[0, 0, :HEIGHT, :WIDTH].float().cpu().numpy().clip(0, 1)
            if alpha.shape != (HEIGHT, WIDTH) or not np.isfinite(alpha).all():
                raise RuntimeError(f"frame {index}: ViTMatte returned an invalid alpha of shape {alpha.shape}")
            alpha[foreground] = 1
            alpha[~support] = 0
            Image.fromarray(np.rint(alpha * 65535).astype(np.uint16)).save(clip.alpha_dir / f"{index:04d}.png")
            if index % 20 == 0:
                print(f"matte frame {index} {time.time() - started:.0f}s", flush=True)
    result = {"frames": count, "thin": thin, "seconds": round(time.time() - started, 1)}
    clip.write_json("matting.json", result)
    return result
