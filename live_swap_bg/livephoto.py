"""Export the rendered clip as a Live Photo pair (JPEG + MOV) to import into Photos or AirDrop to an iPhone.

The input has to be an iPhone Live Photo movie: its timed metadata track (the still-image-time marker) is what
makes Photos treat a movie as a Live Photo, and it is copied onto the new video. macOS only (AVFoundation).
"""

import json
import subprocess
import uuid
from pathlib import Path

from .clip import Clip

SWIFT_SOURCE = Path(__file__).with_name("swift") / "LivePhotoPair.swift"
TOOL_BINARY = Path.home() / ".cache" / "live-swap-bg" / "live-photo-pair"


def tool() -> Path:
    if not TOOL_BINARY.is_file() or TOOL_BINARY.stat().st_mtime < SWIFT_SOURCE.stat().st_mtime:
        TOOL_BINARY.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(["xcrun", "swiftc", "-O", str(SWIFT_SOURCE), "-o", str(TOOL_BINARY)],
                                   capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"compiling {SWIFT_SOURCE.name} failed (needs Xcode command line tools): "
                               f"{completed.stderr.strip()[-500:]}")
    return TOOL_BINARY


def run_tool(*args: str) -> str:
    completed = subprocess.run([str(tool()), *args], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"live-photo-pair {args[0]} failed: {completed.stderr.strip()}")
    return completed.stdout


def export_live_photo(clip: Clip, *, ffmpeg: str = "ffmpeg") -> dict:
    info = clip.info
    template = clip.root / info["source_copy"]
    if template.suffix != ".mov":
        raise ValueError(f"the input was {template.name}; Live Photo export needs the original iPhone Live Photo .MOV")
    source_meta = json.loads(run_tool("inspect", str(template)))
    if not source_meta["still_image_times"]:
        raise ValueError(f"{template.name} has no still-image-time marker; it is not a Live Photo movie")
    rendered = clip.require(clip.out_dir / f"{clip.name}.mp4", "run `live-swap-bg render` first")
    render_info = json.loads((clip.root / "render.json").read_text())

    out = clip.out_dir / "livephoto"
    out.mkdir(exist_ok=True)
    as_mov = out / "video-only.mov"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(rendered), "-c", "copy", "-f", "mov", str(as_mov)], check=True)
    identifier = str(uuid.uuid4()).upper()
    movie, photo = out / f"{clip.name}.MOV", out / f"{clip.name}.JPG"
    run_tool("pair", str(as_mov), str(template), str(movie), identifier)

    # Key photo: the frame at the original still-image time, shifted by what init trimmed and render skipped.
    trimmed = (info["start"] or 0) + render_info["first_source_frame"] / info["fps"]
    still = min(max(source_meta["still_image_times"][0] - trimmed, 0.0), render_info["frames"] / info["fps"] - 0.05)
    unstamped = out / "photo-unstamped.jpg"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{still:.3f}", "-i", str(rendered), "-frames:v", "1",
                    "-q:v", "2", str(unstamped)], check=True)
    run_tool("stamp", str(unstamped), identifier, str(photo))
    as_mov.unlink()
    unstamped.unlink()

    movie_meta = json.loads(run_tool("inspect", str(movie)))
    photo_meta = json.loads(run_tool("inspect-photo", str(photo)))
    if not (movie_meta["content_identifier"] == photo_meta["content_identifier"] == identifier):
        raise RuntimeError(f"identifiers do not match: movie={movie_meta['content_identifier']} "
                           f"photo={photo_meta['content_identifier']} expected={identifier}")
    if not movie_meta["still_image_times"]:
        raise RuntimeError(f"{movie} lost the still-image-time marker")
    result = {"photo": str(photo), "movie": str(movie), "content_identifier": identifier,
              "still_image_seconds": round(still, 3), "movie_metadata_keys": movie_meta["metadata_keys"]}
    clip.write_json("livephoto.json", result)
    return result
