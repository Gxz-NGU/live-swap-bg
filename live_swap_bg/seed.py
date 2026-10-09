"""Seed mask: SAM 2 image prediction from hand-placed prompt groups, then clean-up and a review image.

prompts.json:
    {
      "groups": [                       one group per part: the product, the hand, a thin nail...
        {"name": "bottle", "box": [x0, y0, x1, y1], "pos": [[x, y], ...], "neg": [[x, y], ...]}
      ],
      "add_polygons": [[[x, y], ...]],   optional: paint regions SAM missed (a bottle shoulder against a white wall)
      "clear_boxes": [[x0, y0, x1, y1]]  optional: wipe regions SAM wrongly took
    }

Coordinates are pixels on seed-frame.png (720x960); seed-grid.jpg shows them. Put negative points on things that
must stay out (a second bottle on the table, the gap between two fingers): a negative point inside a gap keeps
that gap open even where clean-up would close it.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .clip import HEIGHT, WIDTH, Clip
from .device import free_memory, pick_device

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = "sam2.1_hiera_large.pt"
MAX_GROUP_AREA = 0.55  # a group covering more than this of the frame grabbed the background
ENCLOSED_HOLE_MAX_PX = 15000
NEGATIVE_GAP_WINDOW_PX = 30


def load_prompts(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; write the prompt groups first (see README)")
    prompts = json.loads(path.read_text())
    groups = prompts.get("groups") if isinstance(prompts, dict) else None
    if not groups:
        raise ValueError(f"{path} needs a non-empty \"groups\" list")
    for group in groups:
        missing = {"name", "box", "pos", "neg"} - set(group)
        if missing:
            raise ValueError(f"{path}: group {group.get('name', '?')} lacks {sorted(missing)}")
        x0, y0, x1, y1 = group["box"]
        if not (0 <= x0 < x1 <= WIDTH and 0 <= y0 < y1 <= HEIGHT):
            raise ValueError(f"{path}: group {group['name']} box {group['box']} is not inside the 720x960 frame")
        for x, y in group["pos"] + group["neg"]:
            if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
                raise ValueError(f"{path}: group {group['name']} point ({x}, {y}) is outside the frame")
    return prompts


def best_candidate(candidates: np.ndarray, box: list[int]) -> np.ndarray | None:
    """Pick by how much of the mask sits inside the box and how much of the box it fills, not by SAM's own score:
    SAM scores how clean a mask is, not whether it is the right object."""
    x0, y0, x1, y1 = box
    best, best_score = None, -1.0
    for candidate in candidates:
        mask = candidate.astype(bool)
        if not mask.any():
            continue
        inside = mask[y0:y1, x0:x1].sum()
        score = inside / mask.sum() * min(inside / max((y1 - y0) * (x1 - x0), 1) * 3, 1.0)
        if score > best_score:
            best_score, best = score, mask
    return best


def clean_union(groups: list[dict], group_masks: list[np.ndarray]) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    for group_mask in group_masks:
        mask |= group_mask.astype(np.uint8) * 255
    before_close = mask.copy()
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    negatives = [(x, y) for group in groups for x, y in group["neg"]]
    # Closing fills slits narrower than 7 px (between fingers, a long nail and the bottle). Hand back the
    # background SAM found around each negative point that lies in such a slit.
    _, background_parts = cv2.connectedComponents((before_close == 0).astype(np.uint8))
    for x, y in negatives:
        part = background_parts[y, x]
        if part:
            window = np.zeros_like(mask, bool)
            window[max(y - NEGATIVE_GAP_WINDOW_PX, 0):y + NEGATIVE_GAP_WINDOW_PX + 1,
                   max(x - NEGATIVE_GAP_WINDOW_PX, 0):x + NEGATIVE_GAP_WINDOW_PX + 1] = True
            mask[(background_parts == part) & window] = 0
    # Fill small holes fully enclosed by the outline, unless a negative point says the hole is a real gap.
    outer = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 40:
            cv2.drawContours(outer, [contour], -1, 255, -1)
    count, holes, stats, _ = cv2.connectedComponentsWithStats(((outer > 0) & (mask == 0)).astype(np.uint8))
    for label in range(1, count):
        hole = holes == label
        if stats[label, cv2.CC_STAT_AREA] < ENCLOSED_HOLE_MAX_PX and not any(hole[y, x] for x, y in negatives):
            mask[hole] = 255
    mask[outer == 0] = 0
    # Keep only the pieces a positive point landed on.
    _, parts = cv2.connectedComponents((mask > 0).astype(np.uint8))
    keep = {int(parts[y, x]) for group in groups for x, y in group["pos"] if parts[y, x] > 0}
    return np.where(np.isin(parts, list(keep)), 255, 0).astype(np.uint8)


def make_seed(clip: Clip, models: Path, *, prompts_path: Path | None = None, device: str | None = None) -> dict:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    prompts = load_prompts(prompts_path or clip.root / "prompts.json")
    groups = prompts["groups"]
    device = pick_device(device)
    rgb = np.asarray(Image.open(clip.root / "seed-frame.png").convert("RGB"))
    predictor = SAM2ImagePredictor(build_sam2(SAM2_CONFIG, str(models / SAM2_CHECKPOINT), device=device))
    group_masks, report = [], []
    with torch.inference_mode():
        predictor.set_image(rgb)
        for group in groups:
            candidates, _, _ = predictor.predict(
                box=np.asarray(group["box"]), point_coords=np.asarray(group["pos"] + group["neg"]),
                point_labels=np.asarray([1] * len(group["pos"]) + [0] * len(group["neg"])), multimask_output=True)
            best = best_candidate(candidates, group["box"])
            if best is None:
                raise RuntimeError(f"group {group['name']}: SAM 2 returned no mask; move the box or add positive points")
            if best.mean() > MAX_GROUP_AREA:
                raise RuntimeError(f"group {group['name']} covers {best.mean():.0%} of the frame; it grabbed the "
                                   "background, tighten the box and add negative points")
            group_masks.append(best)
            report.append({"name": group["name"], "area_percent": round(float(best.mean() * 100), 2)})
    free_memory(device)

    mask = clean_union(groups, group_masks)
    for x0, y0, x1, y1 in prompts.get("clear_boxes", []):
        mask[y0:y1, x0:x1] = 0
    for polygon in prompts.get("add_polygons", []):
        cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 255)
    if not mask.any():
        raise RuntimeError("the seed mask came out empty; check that the positive points sit on the product and hand")

    groups_dir = clip.root / "seed-groups"
    groups_dir.mkdir(exist_ok=True)
    for index, (group, group_mask) in enumerate(zip(groups, group_masks, strict=True)):
        Image.fromarray(group_mask.astype(np.uint8) * 255).save(groups_dir / f"{index:02d}-{group['name']}.png")
    Image.fromarray(mask).save(clip.root / "seed-mask.png")
    write_review(rgb, mask, groups, clip.root / "seed-review.jpg")
    result = {"groups": report, "foreground_percent": round(float((mask > 0).mean() * 100), 2)}
    clip.write_json("seed.json", result)
    return result


def write_review(rgb: np.ndarray, mask: np.ndarray, groups: list[dict], out: Path) -> None:
    """Background dimmed, outline in red, positive points green and negative points blue."""
    review = rgb.copy()
    review[mask == 0] = (review[mask == 0] * 0.28).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(review, contours, -1, (255, 60, 60), 2)
    image = Image.fromarray(review)
    draw = ImageDraw.Draw(image)
    for group in groups:
        for x, y in group["pos"]:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(60, 255, 60))
        for x, y in group["neg"]:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(60, 120, 255))
    image.save(out, quality=93)
