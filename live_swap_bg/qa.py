"""Checks for the things that go wrong. They point you at frames to look at; none of them is a pass on its own."""

import json
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .clip import HEIGHT, WIDTH, Clip
from .edges import repair_edges
from .render import decode


def source_score(video: Path) -> dict:
    """Before any work: can the camera motion be read off this clip's background? Finds the moving foreground by
    frame differencing, then counts trackable points per 120 px cell in what is left. Same bar as `motion`:
    at least 6 points in 3 cells on every frame pair (checked on all pairs: failures sit mid-clip)."""
    capture = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w / h > WIDTH / HEIGHT:
            nw = round(h * WIDTH / HEIGHT)
            frame = frame[:, (w - nw) // 2:(w - nw) // 2 + nw]
        else:
            nh = round(w * HEIGHT / WIDTH)
            frame = frame[(h - nh) // 2:(h - nh) // 2 + nh]
        frames.append(cv2.cvtColor(cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    capture.release()
    if len(frames) < 4:
        raise ValueError(f"{video} decodes to {len(frames)} frames; need at least 4 to score")
    motion = np.mean([cv2.absdiff(a, b).astype(np.float32) for a, b in pairwise(frames)], axis=0)
    moving = (motion > max(float(np.percentile(motion, 75)), 2.0)).astype(np.uint8)
    moving = cv2.dilate(cv2.morphologyEx(moving, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)), np.ones((51, 51), np.uint8))
    background = (moving == 0).astype(np.uint8) * 255
    background[:18] = background[-18:] = 0
    background[:, :18] = background[:, -18:] = 0
    cells, points = [], []
    for a, b in pairwise(frames):
        p0 = cv2.goodFeaturesToTrack(a, maxCorners=400, qualityLevel=0.01, minDistance=8, mask=background)
        if p0 is None:
            cells.append(0)
            points.append(0)
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0 - back, axis=2).ravel() < 0.8)
        kept = p0.reshape(-1, 2)[good]
        points.append(len(kept))
        cells.append(len({(int(x // 120), int(y // 120)) for x, y in kept}))
    moving_share = float((moving > 0).mean())
    return {"worst_cells": int(min(cells)), "worst_points": int(min(points)),
            "trackable": min(cells) >= 3 and min(points) >= 6,
            "moving_area_percent": round(moving_share * 100, 1)}


def flicker_scan(clip: Clip, box: tuple[int, int, int, int], *, top: int = 5, threshold: int = 40) -> dict:
    """Frames where something inside `box` (output pixels) flashes for one frame and is gone in the next:
    per frame, pixels where |v_k - (v_k-1 + v_k+1)/2| > threshold. Writes out/flicker-<x0>-<y0>.jpg with
    rows of k-1 | k | k+1 | repaired alpha at k, enlarged 3x. A finger gap that shows the old wall on one frame
    and the new one on the next is the classic case; stills hide it, playback does not."""
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= WIDTH and 0 <= y0 < y1 <= HEIGHT):
        raise ValueError(f"box {box} is not inside the 720x960 frame")
    render_info = json.loads(clip.require(clip.root / "render.json", "run `live-swap-bg render` first").read_text())
    frames = [f[y0:y1, x0:x1].astype(np.float32) for f in decode(Path(render_info["output"]))]
    scores = sorted(((int((np.abs(frames[k] - (frames[k - 1] + frames[k + 1]) / 2).max(axis=2) > threshold).sum()), k)
                     for k in range(1, len(frames) - 1)), reverse=True)
    picked = sorted(k for _, k in scores[:top])
    scale = 3
    w, h = (x1 - x0) * scale, (y1 - y0) * scale
    sheet = Image.new("RGB", (w * 4 + 30, (h + 6) * len(picked)), (255, 255, 255))
    start = render_info["first_source_frame"]
    for row, k in enumerate(picked):
        alpha, _ = repair_edges(clip.alpha(k + start), clip.rgb(k + start), render_info["max_gap_pixels"])
        tiles = [frames[k - 1], frames[k], frames[k + 1], np.dstack([alpha[y0:y1, x0:x1] * 255] * 3)]
        for col, tile in enumerate(tiles):
            image = Image.fromarray(np.uint8(np.clip(tile, 0, 255))).resize((w, h), Image.Resampling.NEAREST)
            sheet.paste(image, (col * (w + 10), row * (h + 6)))
        ImageDraw.Draw(sheet).text((2, row * (h + 6) + 2), f"f{k - 1} | f{k} | f{k + 1} | alpha f{k}", fill=(255, 0, 0))
    out = clip.out_dir / f"flicker-{x0}-{y0}.jpg"
    sheet.save(out, quality=90)
    return {"spike_pixels_by_frame": [{"frame": k, "pixels": n} for n, k in scores[:top * 2]], "sheet": str(out)}


def background_jitter(clip: Clip) -> dict:
    """How the new background moves in the rendered video, measured away from the foreground. The high-frequency
    part (path minus its 7-frame moving average) is what a viewer sees as a shaking background."""
    render_info = json.loads(clip.require(clip.root / "render.json", "run `live-swap-bg render` first").read_text())
    start, count = render_info["first_source_frame"], render_info["frames"]
    union = np.zeros((HEIGHT, WIDTH), bool)
    for k in range(start, start + count):
        union |= clip.alpha(k) > 0.01
    keep = ~cv2.dilate(union.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    keep[:20], keep[-20:], keep[:, :20], keep[:, -20:] = False, False, False, False
    if keep.sum() < 2000:
        raise ValueError("the foreground covers nearly the whole frame; too little background left to measure")
    mask = keep.astype(np.uint8) * 255
    frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in decode(Path(render_info["output"]))]
    steps = []
    for prev, cur in pairwise(frames):
        points = cv2.goodFeaturesToTrack(prev, 600, 0.01, 8, mask=mask)
        if points is None:
            raise ValueError("no trackable texture in the rendered background; jitter cannot be measured")
        moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, points, None, winSize=(31, 31), maxLevel=3)
        steps.append(np.median((moved - points).reshape(-1, 2)[status.ravel() > 0], axis=0))
    steps = np.array(steps)
    path = np.vstack([[0, 0], np.cumsum(steps, axis=0)])
    smooth = np.array([np.convolve(path[:, i], np.ones(7) / 7, mode="same") for i in (0, 1)]).T
    inner = (path - smooth)[7:-7] if len(path) > 14 else path - smooth
    return {"largest_step_px": np.abs(steps).max(axis=0).round(2).tolist(),
            "range_px": np.ptp(path, axis=0).round(1).tolist(),
            "high_frequency_std_px": inner.std(axis=0).round(2).tolist()}


def edge_crops(clip: Clip) -> dict:
    """The frames where edges go wrong most often, cropped around the foreground at full resolution: the first 3,
    the last 4 (one-sided temporal windows), and the frame where the foreground moves most."""
    render_info = json.loads(clip.require(clip.root / "render.json", "run `live-swap-bg render` first").read_text())
    start, count = render_info["first_source_frame"], render_info["frames"]
    frames = decode(Path(render_info["output"]))
    areas = [clip.alpha(start + k) > 0.5 for k in range(count)]
    change = [int((areas[k] ^ areas[k - 1]).sum()) for k in range(1, count)]
    fastest = int(np.argmax(change)) + 1
    picks = sorted({0, 1, 2, count - 4, count - 3, count - 2, count - 1, fastest} & set(range(count)))
    tiles = []
    for k in picks:
        ys, xs = np.nonzero(areas[k])
        if not len(ys):
            continue
        pad = 24
        bx0, by0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
        bx1, by1 = min(xs.max() + pad, WIDTH), min(ys.max() + pad, HEIGHT)
        crop = Image.fromarray(frames[k][by0:by1, bx0:bx1])
        ImageDraw.Draw(crop).text((4, 4), f"frame {start + k}" + (" (fastest)" if k == fastest else ""), fill=(255, 0, 0))
        tiles.append(crop)
    height = max(t.height for t in tiles)
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 8 * len(tiles), height), (30, 30, 30))
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width + 8
    out = clip.out_dir / "edge-crops.jpg"
    sheet.save(out, quality=92)
    return {"frames": [start + k for k in picks], "fastest_frame": start + fastest, "sheet": str(out)}
