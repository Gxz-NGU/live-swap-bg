"""Turn an input video into the frames every later stage works on."""

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from .clip import FPS, HEIGHT, WIDTH, Clip

# SAM 2 keeps every frame's features while it tracks; a Live Photo is about 90 frames.
MAX_FRAMES = 150


def init_clip(source: Path, root: Path, *, start: float | None = None, end: float | None = None,
              ffmpeg: str = "ffmpeg") -> Clip:
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input video not found: {source}")
    clip = Clip(root)
    if clip.root.exists() and any(clip.root.iterdir()):
        raise FileExistsError(f"{clip.root} is not empty; pick a new working directory")
    clip.frames_dir.mkdir(parents=True)
    clip.jpg_dir.mkdir()
    shutil.copy2(source, clip.root / ("source" + source.suffix.lower()))

    trim = ""
    if start is not None or end is not None:
        bounds = ":".join(part for part in (f"start={start}" if start is not None else "",
                                            f"end={end}" if end is not None else "") if part)
        trim = f"trim={bounds},setpts=PTS-STARTPTS,"
    # Centre-crop to 3:4 portrait, never stretch.
    crop = f"crop=min(iw\\,ih*3/4):min(ih\\,iw*4/3),scale={WIDTH}:{HEIGHT}:flags=lanczos,fps={FPS}"
    subprocess.run([ffmpeg, "-v", "error", "-i", str(source), "-map", "0:v:0", "-an", "-vf", trim + crop,
                    "-start_number", "0", str(clip.frames_dir / "%04d.png")], check=True)
    frames = sorted(clip.frames_dir.glob("*.png"))
    if len(frames) < 2:
        raise ValueError(f"{source} gave {len(frames)} frame(s) after trimming; a clip needs at least 2")
    if len(frames) > MAX_FRAMES:
        raise ValueError(f"{source} gave {len(frames)} frames, more than {MAX_FRAMES}; pass --start/--end to cut "
                         f"a shorter stretch (at {FPS} fps that is {MAX_FRAMES / FPS:.0f} seconds)")
    for frame in frames:
        Image.open(frame).convert("RGB").save(clip.jpg_dir / (frame.stem + ".jpg"), quality=95)

    seed = len(frames) // 2
    shutil.copy2(frames[seed], clip.root / "seed-frame.png")
    write_seed_grid(clip.root / "seed-frame.png", clip.root / "seed-grid.jpg")
    clip.write_json("clip.json", {"source": str(source), "source_copy": "source" + source.suffix.lower(),
                                  "frames": len(frames), "seed_frame": seed, "start": start, "end": end,
                                  "width": WIDTH, "height": HEIGHT, "fps": FPS})
    return clip


def write_seed_grid(frame: Path, out: Path, step: int = 60) -> None:
    """The seed frame with a labelled pixel grid, for reading off box and point coordinates."""
    image = Image.open(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x in range(0, WIDTH, step):
        draw.line([(x, 0), (x, HEIGHT)], fill=(255, 0, 0), width=1)
        draw.text((x + 2, 2), str(x), fill=(255, 255, 0))
    for y in range(0, HEIGHT, step):
        draw.line([(0, y), (WIDTH, y)], fill=(255, 0, 0), width=1)
        draw.text((2, y + 2), str(y), fill=(255, 255, 0))
    image.save(out, quality=92)
