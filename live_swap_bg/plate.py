"""Prepare the generated scene as the background plate render uses.

- A light disc blur, like a phone's portrait mode, so the plate sits behind a sharp product.
- Only a ring around the foreground edge is matched to the original wall colour. Whatever the matte leaves at
  the edge (a stray pixel, a translucent rim) shows the original wall, and it disappears when the plate right
  there is close in colour. First the whole plate gets an exposure/white-balance nudge (no patch shows), then
  the ring closes the rest. Outside the ring the scene stays as generated.
- Matching only works when the scene is roughly as bright as the original wall. A bright seaside room behind a
  clip shot against a dark grey wall would be dragged down to grey, so a large gap is refused; pick a scene
  closer in brightness, or pass match=False to keep the scene as is (render's edge repair still cleans the rim).
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .clip import HEIGHT, WIDTH, Clip
from .motion import plate_to_frame

FULL_MATCH_PX, FADE_PX = 20, 120  # matched within 20 px of the edge, fading out over 120 px (shorter shows a patch)
MAX_LIGHTNESS_GAP = 20  # CIELAB L between the scene near the edge and the original wall


def wall_colour(clip: Clip) -> np.ndarray:
    """Median CIELAB of a 5-20 px ring outside the foreground, every sixth frame."""
    samples = []
    for k in range(0, clip.info["frames"], 6):
        source = clip.rgb(k) / 255
        foreground = (clip.alpha(k) > 0.05).astype(np.uint8)
        ring = (cv2.dilate(foreground, np.ones((41, 41), np.uint8)) > 0) & ~(cv2.dilate(foreground, np.ones((11, 11), np.uint8)) > 0)
        samples.append(cv2.cvtColor(source.astype(np.float32), cv2.COLOR_RGB2LAB)[ring])
    return np.median(np.concatenate(samples), axis=0).astype(np.float32)


def prepare_plate(clip: Clip, background: Path, *, blur: int = 3, frame_space: bool = False, match: bool = True) -> dict:
    """frame_space=True: the scene was composed as the viewer sees the frame, so map it back through the middle
    frame's zoom and shift. Use it when the camera moved a lot and props near the edge would be pushed out."""
    motion = json.loads(clip.require(clip.root / "motion.json", "run `live-swap-bg motion` first").read_text())
    count = clip.info["frames"]
    target = wall_colour(clip)
    raw = ImageOps.fit(Image.open(background).convert("RGB"), (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)
    rgb = np.asarray(raw, np.float32) / 255
    if frame_space:
        rgb = cv2.warpAffine(rgb, plate_to_frame(motion, count // 2), (WIDTH, HEIGHT),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * blur + 1, 2 * blur + 1)).astype(np.float32)
    blurred = cv2.filter2D(rgb, -1, disk / disk.sum(), borderType=cv2.BORDER_REFLECT)

    # The ring has to continue past the frame edge: a hand entering from the side would otherwise leave a
    # straight colour seam along the frame border that slides into view as the camera moves.
    pad = FULL_MATCH_PX + FADE_PX
    weight = np.zeros((HEIGHT, WIDTH), np.float32)
    for k in range(0, count, 2):
        foreground = (clip.alpha(k) > 0.05).astype(np.uint8)
        foreground = cv2.copyMakeBorder(foreground, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        distance = cv2.distanceTransform(1 - foreground, cv2.DIST_L2, 5)
        band = np.clip((FULL_MATCH_PX + FADE_PX - distance) / FADE_PX, 0, 1)
        to_padded = plate_to_frame(motion, k)
        to_padded[:, 2] += pad
        weight = np.maximum(weight, cv2.warpAffine(band, to_padded, (WIDTH, HEIGHT),
                                                   flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                                                   borderMode=cv2.BORDER_REPLICATE))
    if weight.sum() == 0:
        raise ValueError("the foreground never comes near the plate; check alpha/")
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    band_mean = (lab * weight[:, :, None]).sum(axis=(0, 1)) / weight.sum()
    gap = float(target[0] - band_mean[0])
    if match and abs(gap) > MAX_LIGHTNESS_GAP:
        raise ValueError(f"the scene around the foreground (L={band_mean[0]:.0f}) is {abs(gap):.0f} "
                         f"{'darker' if gap > 0 else 'brighter'} than the original wall (L={target[0]:.0f}); matching "
                         f"would recolour the whole scene. Generate a scene closer in brightness, or rerun with "
                         f"--no-match to only blur it (render's edge repair still cleans the rim)")
    if match:
        lab = lab + (target - band_mean) * np.array([1.0, 0.5, 0.5], np.float32)  # half the tint: keep the scene's mood
        local = np.dstack([cv2.GaussianBlur(lab[:, :, c], (0, 0), 30) for c in range(3)])
        lab = lab + weight[:, :, None] * np.array([1.0, 0.3, 0.3], np.float32) * (target - local)
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    Image.fromarray(np.uint8(out * 255 + 0.5)).save(clip.root / "plate.png")
    Image.fromarray(np.uint8(weight * 255)).save(clip.root / "plate-match-weight.png")
    result = {"background": str(Path(background).resolve()), "blur_radius": blur, "frame_space": frame_space,
              "matched": match, "wall_lab": np.round(target, 2).tolist(),
              "ring_lab_before": np.round(band_mean, 2).tolist(), "lightness_gap": round(gap, 1)}
    clip.write_json("plate.json", result)
    return result
