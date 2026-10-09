"""Composite the real hand and product over the moving plate and encode the result."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .clip import FPS, HEIGHT, WIDTH, Clip
from .edges import repair_edges
from .light import key_light_direction, plate_floor_lab, product_light_span, relight
from .motion import plate_to_frame


def pick_encoder(ffmpeg: str) -> list[str]:
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True).stdout
    if "h264_videotoolbox" in encoders:
        return ["-c:v", "h264_videotoolbox", "-allow_sw", "1", "-b:v", "10M"]
    if "libx264" in encoders:
        return ["-c:v", "libx264", "-preset", "slow", "-crf", "16"]
    raise RuntimeError(f"{ffmpeg} has neither h264_videotoolbox nor libx264; install an ffmpeg build with H.264")


def render(clip: Clip, *, background: Path | None = None, edge_repair: bool = True, max_gap_pixels: int = 0,
           rests_on_surface: bool = False, start: int = 0, count: int | None = None, ffmpeg: str = "ffmpeg") -> dict:
    """rests_on_surface: the product stands on a surface in the plate (not held); adds a contact shadow.
    max_gap_pixels: fill background gaps fully enclosed by the foreground up to this size, only after checking
    they show the old wall and not a real slit or shadow."""
    info = clip.info
    count = info["frames"] - start if count is None else count
    if start < 0 or count < 2 or start + count > info["frames"]:
        raise ValueError(f"frames {start}..{start + count - 1} are outside the clip's 0..{info['frames'] - 1}")
    plate_path = background or clip.root / "plate.png"
    clip.require(plate_path, "run `live-swap-bg plate <workdir> <scene.png>` or pass --background")
    motion = json.loads(clip.require(clip.root / "motion.json", "run `live-swap-bg motion` first").read_text())
    plate = np.asarray(ImageOps.fit(Image.open(plate_path).convert("RGB"), (WIDTH, HEIGHT),
                                    method=Image.Resampling.LANCZOS), np.float32)
    # A product cut from its own shot carries its own light and casts nothing; give it the plate's.
    light = key_light_direction(plate)
    floor_lab = plate_floor_lab(plate)
    span = product_light_span(clip.alpha(info["seed_frame"]), light)

    clip.out_dir.mkdir(exist_ok=True)
    output = clip.out_dir / f"{clip.name}.mp4"
    partial = output.with_suffix(".rendering.mp4")
    encoder = subprocess.Popen([ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
                                "-r", str(FPS), "-i", "-", "-an", *pick_encoder(ffmpeg), "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", str(partial)], stdin=subprocess.PIPE)
    for index in range(start, start + count):
        rgb, alpha = clip.rgb(index), clip.alpha(index)
        if edge_repair:
            alpha, rgb = repair_edges(alpha, rgb, max_gap_pixels)
        to_frame = plate_to_frame(motion, index)
        if cv2.warpAffine(np.ones((HEIGHT, WIDTH), np.float32), to_frame, (WIDTH, HEIGHT)).min() < 0.999:
            raise RuntimeError(f"frame {index}: the plate does not cover the frame after the camera shift")
        moved = cv2.warpAffine(plate, to_frame, (WIDTH, HEIGHT))
        lit, moved = relight(np.uint8(rgb), alpha, moved, light=light, span=span, floor_lab=floor_lab,
                             rests_on_surface=rests_on_surface)
        frame = np.clip(lit * alpha[:, :, None] + moved * (1 - alpha[:, :, None]), 0, 255).astype(np.uint8)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while encoding {partial}")
    frames = decode(partial)
    if len(frames) != count:
        raise RuntimeError(f"{partial} decodes to {len(frames)} frames, expected {count}")
    partial.replace(output)
    sheets = write_contact_sheets(frames, start, clip.out_dir)
    result = {"output": str(output), "frames": count, "first_source_frame": start, "fps": FPS,
              "width": WIDTH, "height": HEIGHT, "plate": str(Path(plate_path).resolve()),
              "edge_repair": edge_repair, "max_gap_pixels": max_gap_pixels, "rests_on_surface": rests_on_surface,
              "review_sheets": [str(p) for p in sheets],
              "visual_review": "pending: metrics cannot judge looks; check the sheets and play the video"}
    clip.write_json("render.json", result)
    return result


def decode(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def write_contact_sheets(frames: list[np.ndarray], start: int, out_dir: Path, per_sheet: int = 24) -> list[Path]:
    """Every encoded frame as a small tile, 24 per sheet: review material, not a pass."""
    font = ImageFont.load_default()
    paths = []
    for offset in range(0, len(frames), per_sheet):
        sheet = Image.new("RGB", (6 * 240, 4 * 345), (25, 25, 25))
        draw = ImageDraw.Draw(sheet)
        for j, frame in enumerate(frames[offset:offset + per_sheet]):
            x, y = (j % 6) * 240, (j // 6) * 345
            sheet.paste(Image.fromarray(frame).resize((240, 320)), (x, y + 25))
            draw.text((x + 6, y + 3), f"frame {start + offset + j}", font=font, fill="white")
        path = out_dir / f"review-{offset:04d}.jpg"
        sheet.save(path, quality=92)
        paths.append(path)
    return paths
