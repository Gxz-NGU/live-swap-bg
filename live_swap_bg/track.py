"""Track the seed mask through every frame with SAM 2's video predictor."""

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .clip import Clip
from .device import free_memory, pick_device
from .seed import SAM2_CHECKPOINT, SAM2_CONFIG


def track(clip: Clip, models: Path, *, multi: bool = False, device: str | None = None) -> dict:
    """multi=True tracks each prompt group as its own object and unions them per frame. Use it when a single
    object loses a translucent or frosted bottle body a few frames after the seed."""
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    if (clip.root / "tracking.json").exists():
        raise FileExistsError(f"{clip.root}/tracking.json exists; delete it and masks/ to track again")
    info = clip.info
    count, seed_frame = info["frames"], info["seed_frame"]
    if multi:
        seeds = [np.asarray(Image.open(p)) > 127 for p in sorted((clip.root / "seed-groups").glob("*.png"))]
        if not seeds:
            raise FileNotFoundError(f"{clip.root}/seed-groups/ is empty; run `live-swap-bg seed` first")
    else:
        seeds = [np.asarray(Image.open(clip.require(clip.root / "seed-mask.png", "run `live-swap-bg seed` first"))) > 127]
    device = pick_device(device)
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(models / SAM2_CHECKPOINT), device=device)
    predictor.fill_hole_area = 0
    clip.masks_dir.mkdir(exist_ok=True)
    started = time.time()
    masks = {}
    with torch.inference_mode():
        state = predictor.init_state(str(clip.jpg_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for object_id, seed in enumerate(seeds, 1):
            predictor.add_new_mask(state, frame_idx=seed_frame, obj_id=object_id, mask=seed)
        for reverse in (False, True):
            for index, ids, logits in predictor.propagate_in_video(state, start_frame_idx=seed_frame, reverse=reverse):
                if sorted(ids) != list(range(1, len(seeds) + 1)):
                    raise RuntimeError(f"frame {index}: SAM 2 returned objects {ids}, expected 1..{len(seeds)}")
                union = (logits[:, 0].float().cpu().numpy() > 0).any(axis=0).astype(np.uint8) * 255
                if multi:
                    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
                masks[index] = union
                if index % 10 == 0:
                    free_memory(device)
                    print(f"track frame {index} {time.time() - started:.0f}s", flush=True)
    if len(masks) != count:
        raise RuntimeError(f"tracking covered {len(masks)} of {count} frames")
    for index, mask in masks.items():
        Image.fromarray(mask).save(clip.masks_dir / f"{index:04d}.png")
    areas = [float((m > 0).mean() * 100) for _, m in sorted(masks.items())]
    result = {"frames": count, "objects": len(seeds), "seconds": round(time.time() - started, 1),
              "area_percent_min": round(min(areas), 2), "area_percent_max": round(max(areas), 2),
              "largest_area_jump": round(float(np.abs(np.diff(areas)).max()), 2)}
    clip.write_json("tracking.json", result)
    return result
