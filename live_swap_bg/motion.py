"""Estimate how the camera moved from the original background, so the new background can move the same way.

Only translation is recovered: no rotation, scale, perspective or parallax. A product that moves while the
background behind it stays frozen reads as fake at a glance; a background that follows the camera does not.
"""

from pathlib import Path

import cv2
import numpy as np

from .clip import HEIGHT, WIDTH, Clip

# On a near-featureless wall LK locks onto video noise and mains-light flicker and the plate jitters by several
# pixels every few frames without failing outright. Good clips stay under ~1.2 px of per-frame residual.
MAX_RESIDUAL_PX = 1.5
STATIC_ZOOM = 1.02


def estimate_background_translation(frames: list[Path], masks: list[Path]) -> dict:
    if len(frames) < 2 or len(frames) != len(masks):
        raise ValueError(f"camera translation needs matching frames/masks: {len(frames)}/{len(masks)}")
    gray, backgrounds = [], []
    for frame_path, mask_path in zip(frames, masks, strict=True):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None or frame.shape != (HEIGHT, WIDTH) or mask.shape != frame.shape:
            raise ValueError(f"camera input must be a readable 720x960 frame/mask: {frame_path}, {mask_path}")
        if not np.any(mask) or not np.any(mask == 0):
            raise ValueError(f"foreground mask needs both foreground and background: {mask_path}")
        background = (cv2.dilate(mask, np.ones((51, 51), np.uint8)) == 0).astype("uint8") * 255
        background[:18] = background[-18:] = 0
        background[:, :18] = background[:, -18:] = 0
        gray.append(frame)
        backgrounds.append(background)
    positions = [np.zeros(2)]
    quality = []
    for index in range(1, len(frames)):
        points = cv2.goodFeaturesToTrack(gray[index - 1], 1000, 0.002, 8, mask=backgrounds[index - 1])
        if points is None or len(points) < 6:
            raise ValueError(f"insufficient background points: frame={index}")
        forward, status, _ = cv2.calcOpticalFlowPyrLK(gray[index - 1], gray[index], points, None, winSize=(35, 35), maxLevel=4)
        if forward is None or status is None:
            raise ValueError(f"background forward tracking failed: frame={index}")
        reverse, reverse_status, _ = cv2.calcOpticalFlowPyrLK(gray[index], gray[index - 1], forward, None,
                                                              winSize=(35, 35), maxLevel=4)
        if reverse is None or reverse_status is None:
            raise ValueError(f"background reverse tracking failed: frame={index}")
        destinations = forward.reshape(-1, 2)
        inside = ((destinations[:, 0] >= 0) & (destinations[:, 0] < WIDTH)
                  & (destinations[:, 1] >= 0) & (destinations[:, 1] < HEIGHT))
        good = ((status.ravel() > 0) & (reverse_status.ravel() > 0)
                & (np.linalg.norm((reverse - points).reshape(-1, 2), axis=1) < 0.8) & inside)
        indices = np.where(good)[0]
        indices = indices[backgrounds[index][destinations[indices, 1].astype(int), destinations[indices, 0].astype(int)] > 0]
        if len(indices) < 6:
            raise ValueError(f"background track lost: frame={index} points={len(indices)}")
        origins = points[indices].reshape(-1, 2)
        deltas = destinations[indices] - origins
        cells = {}
        for point, delta in zip(origins, deltas, strict=True):
            cells.setdefault((int(point[0] // 120), int(point[1] // 120)), []).append(delta)
        shifts = [np.median(values, axis=0) for values in cells.values() if len(values) >= 2]
        if len(shifts) < 3:
            raise ValueError(f"insufficient background grid cells: frame={index} cells={len(shifts)}")
        translation = np.median(shifts, axis=0)
        positions.append(positions[-1] + translation)
        quality.append({"frame": index, "tracks": len(indices), "grid_cells": len(shifts),
                        "median_translation_residual": float(np.median(np.linalg.norm(deltas - translation, axis=1)))})
    positions = np.array(positions)
    positions -= (positions.min(axis=0) + positions.max(axis=0)) / 2
    zoom = float(np.ceil((1 + max(np.abs(positions[:, 0]).max() / ((WIDTH - 1) / 2),
                                 np.abs(positions[:, 1]).max() / ((HEIGHT - 1) / 2)) + 0.015) * 100) / 100)
    return {"method": "spatially balanced background translation; no rotation/scale/perspective reconstruction",
            "positions": positions.tolist(), "range_pixels": np.ptp(positions, axis=0).tolist(),
            "background_zoom": zoom, "quality": quality}


def static_motion(count: int, reason: str) -> dict:
    return {"method": "static background", "reason": reason, "positions": [[0.0, 0.0]] * count,
            "range_pixels": [0.0, 0.0], "background_zoom": STATIC_ZOOM, "quality": []}


def estimate_motion(clip: Clip, *, static: bool = False) -> dict:
    count = clip.info["frames"]
    if static:
        result = static_motion(count, "requested with --static")
    else:
        frames = clip.frame_paths()
        masks = [clip.require(clip.masks_dir / p.name, "run `live-swap-bg track` first") for p in frames]
        try:
            estimate = estimate_background_translation(frames, masks)
        except ValueError as error:
            raise ValueError(f"{error}. The original background has too little texture to follow the camera; "
                             "rerun with --static to keep the new background still") from error
        residual = float(np.median([q["median_translation_residual"] for q in estimate["quality"]]))
        if residual > MAX_RESIDUAL_PX:
            clip.write_json("motion-lk.json", estimate)
            result = static_motion(count, f"median LK residual {residual:.2f}px > {MAX_RESIDUAL_PX}px: the estimate "
                                          "follows noise, not the camera (kept in motion-lk.json)")
        else:
            result = estimate
        result["median_residual_px"] = round(residual, 3)
    clip.write_json("motion.json", result)
    return result


def plate_to_frame(motion: dict, index: int) -> np.ndarray:
    """Affine that maps the plate onto output frame `index`: zoom about the centre, then the camera shift."""
    zoom = motion["background_zoom"]
    dx, dy = motion["positions"][index]
    cx, cy = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
    return np.array([[zoom, 0, cx * (1 - zoom) + dx], [0, zoom, cy * (1 - zoom) + dy]], np.float32)
