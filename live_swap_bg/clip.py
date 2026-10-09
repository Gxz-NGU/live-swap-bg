"""A clip's working directory. Every stage reads its inputs from here and writes its outputs here.

    <workdir>/
      clip.json          what `init` recorded: source, frame count, seed frame, trim
      source.mov         copy of the input video (the Live Photo template when it is a Live Photo)
      frames/            720x960 RGB frames at 30 fps
      sam-jpg/           the same frames as JPEG, the input format SAM 2 wants
      seed-frame.png     the frame the prompts are drawn on; seed-grid.jpg has a coordinate grid
      prompts.json       boxes and points you write by hand (see README)
      seed-mask.png      union of the prompt groups; seed-groups/ keeps each group
      masks/             SAM 2 mask per frame
      alpha/             ViTMatte alpha per frame, 16-bit PNG
      motion.json        background translation per frame
      background.png     your generated scene; plate.png is the prepared version render uses
      out/               rendered MP4, review sheets, Live Photo pair
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT, FPS = 720, 960, 30


class Clip:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.frames_dir = self.root / "frames"
        self.jpg_dir = self.root / "sam-jpg"
        self.masks_dir = self.root / "masks"
        self.alpha_dir = self.root / "alpha"
        self.out_dir = self.root / "out"

    @property
    def info(self) -> dict:
        path = self.root / "clip.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; run `live-swap-bg init <video> {self.root}` first")
        return json.loads(path.read_text())

    @property
    def name(self) -> str:
        return self.root.name

    def frame_paths(self) -> list[Path]:
        expected = self.info["frames"]
        paths = sorted(self.frames_dir.glob("*.png"))
        if [p.name for p in paths] != [f"{i:04d}.png" for i in range(expected)]:
            raise ValueError(f"{self.frames_dir} should hold frames 0000-{expected - 1:04d}.png, found {len(paths)} files")
        return paths

    def require(self, path: Path, hint: str) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; {hint}")
        return path

    def mask(self, index: int) -> np.ndarray:
        path = self.require(self.masks_dir / f"{index:04d}.png", "run `live-swap-bg track` first")
        return np.asarray(Image.open(path)) > 127

    def alpha(self, index: int) -> np.ndarray:
        path = self.require(self.alpha_dir / f"{index:04d}.png", "run `live-swap-bg matte` first")
        return np.asarray(Image.open(path), np.float32) / 65535

    def rgb(self, index: int) -> np.ndarray:
        return np.asarray(Image.open(self.frames_dir / f"{index:04d}.png").convert("RGB"), np.float32)

    def write_json(self, name: str, value: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        return path
