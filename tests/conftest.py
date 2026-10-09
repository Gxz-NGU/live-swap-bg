import json
import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

from live_swap_bg.clip import Clip


def make_clip(root, count=8, *, textured=True):
    """A synthetic clip: a textured wall the camera pans across, and a 'product' that moves faster on its own.
    Writes frames, masks and alpha the way track and matte would."""
    cv2.setNumThreads(1)
    clip = Clip(root)
    for folder in (clip.frames_dir, clip.jpg_dir, clip.masks_dir, clip.alpha_dir):
        folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    wall = cv2.GaussianBlur(rng.integers(20, 230, (960, 720, 3), dtype=np.uint8), (5, 5), 0)
    if not textured:
        wall[:] = 80
    for index in range(count):
        frame = cv2.warpAffine(wall, np.float32([[1, 0, index * 2], [0, 1, -index]]), (720, 960))
        mask = np.zeros((960, 720), np.uint8)
        mask[560:960, 250 + index * 8:390 + index * 8] = 255
        frame[mask > 0] = (220, 180, 100)
        Image.fromarray(frame).save(clip.frames_dir / f"{index:04d}.png")
        Image.fromarray(mask).save(clip.masks_dir / f"{index:04d}.png")
        Image.fromarray(mask.astype(np.uint16) * 257).save(clip.alpha_dir / f"{index:04d}.png")
    shutil.copy2(clip.frames_dir / "0000.png", clip.root / "seed-frame.png")
    (clip.root / "clip.json").write_text(json.dumps({"source": "synthetic", "source_copy": "source.mp4",
                                                     "frames": count, "seed_frame": count // 2, "start": None,
                                                     "end": None, "width": 720, "height": 960, "fps": 30}))
    return clip, wall


@pytest.fixture
def synthetic_clip(tmp_path):
    return make_clip(tmp_path / "clip")
