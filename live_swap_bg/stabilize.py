"""Temporal clean-up of masks and alphas. Every operation keeps the untouched input in `<dir>-before-<op>/`
and reruns from there, so running one twice gives the same result instead of compounding.

When to use which (each has a failure mode; look at the result before moving on):
- vote_masks: a mask edge flickers on and off frame to frame (a nail tip, a cap highlight). Run it between
  track and matte. It smears a fast-moving thin part (a nail during a quick hand move) into a fork: compare
  masks-before-vote/ with masks/ on the fastest frames.
- fill_alpha_dropouts: a translucent body or foam drops out for one or two frames. Safe: it only raises alpha.
- median_alpha: soft-edge alpha still shimmers after vote_masks. It drags a fast-moving finger gap closed and
  can hollow out a nail; the first and last frames keep their raw alpha because the window is one-sided there.
"""

import shutil
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .clip import Clip

WINDOW = 5
DROPOUT = 0.4


def flicker_pixels(binary_frames: list[np.ndarray]) -> int:
    """Pixels that switch on/off in at least 10% of frame transitions (at least 4 times), opened to drop specks."""
    switches = np.zeros(binary_frames[0].shape, np.int32)
    for a, b in pairwise(binary_frames):
        switches += a ^ b
    hot = (switches >= max(4, int((len(binary_frames) - 1) * 0.10))).astype(np.uint8)
    return int(cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).sum())


def _source_dir(directory: Path, operation: str) -> Path:
    backup = directory.with_name(f"{directory.name}-before-{operation}")
    if not backup.exists():
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} is missing")
        shutil.copytree(directory, backup)
    return backup


def vote_masks(clip: Clip) -> dict:
    source = _source_dir(clip.masks_dir, "vote")
    count = clip.info["frames"]
    masks = [np.asarray(Image.open(source / f"{k:04d}.png")) > 127 for k in range(count)]
    half = WINDOW // 2
    voted = []
    for i in range(count):
        window = np.stack(masks[max(0, i - half):i + half + 1]).astype(np.int16)
        voted.append(window.sum(0) > window.shape[0] // 2)
    for k, mask in enumerate(voted):
        Image.fromarray(mask.astype(np.uint8) * 255).save(clip.masks_dir / f"{k:04d}.png")
    return {"flicker_pixels_before": flicker_pixels(masks), "flicker_pixels_after": flicker_pixels(voted)}


def fill_alpha_dropouts(clip: Clip) -> dict:
    source = _source_dir(clip.alpha_dir, "dropfill")
    count = clip.info["frames"]
    alpha = np.stack([np.asarray(Image.open(source / f"{k:04d}.png"), np.float32) / 65535 for k in range(count)])
    filled = alpha.copy()
    for t in range(1, count - 1):  # a one-frame dip: raise it to the lower of its neighbours
        filled[t] = np.maximum(filled[t], np.minimum(alpha[t - 1], alpha[t + 1]))
    for t in range(1, count - 2):  # a two-frame dip
        floor = np.minimum(alpha[t - 1], alpha[t + 2])
        filled[t] = np.maximum(filled[t], floor)
        filled[t + 1] = np.maximum(filled[t + 1], floor)

    def dips(x):
        return int(((x[1:-1] - x[:-2] < -DROPOUT) & (x[1:-1] - x[2:] < -DROPOUT)).sum())

    for k in range(count):
        Image.fromarray(np.rint(filled[k] * 65535).astype(np.uint16)).save(clip.alpha_dir / f"{k:04d}.png")
    return {"one_frame_dips_before": dips(alpha), "one_frame_dips_after": dips(filled),
            "changed_pixels": int((np.abs(filled - alpha) > 0.05).sum())}


def median_alpha(clip: Clip) -> dict:
    source = _source_dir(clip.alpha_dir, "median")
    count = clip.info["frames"]
    alpha = np.stack([np.asarray(Image.open(source / f"{k:04d}.png"), np.float32) for k in range(count)])
    half = WINDOW // 2
    out = np.empty_like(alpha)
    for i in range(count):
        out[i] = np.median(alpha[max(0, i - half):i + half + 1], axis=0)
    for i in range(count):  # one-sided windows at the ends lag the edge: keep raw, blend one frame in
        if i < 2 or i >= count - 3:
            out[i] = alpha[i]
        elif i == 2 or i == count - 4:
            out[i] = 0.5 * (alpha[i] + out[i])
    for k in range(count):
        Image.fromarray(out[k].astype(np.uint16)).save(clip.alpha_dir / f"{k:04d}.png")
    return {"flicker_pixels_before": flicker_pixels([a / 65535 > 0.5 for a in alpha]),
            "flicker_pixels_after": flicker_pixels([a / 65535 > 0.5 for a in out])}
