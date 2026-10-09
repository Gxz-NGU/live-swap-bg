import json
import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

from live_swap_bg.motion import estimate_motion
from live_swap_bg.plate import prepare_plate
from live_swap_bg.qa import background_jitter, edge_crops
from live_swap_bg.render import decode, render

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _scene(path):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(60, 200, (960, 720, 3), dtype=np.uint8)).save(path)


def test_render_puts_the_real_product_on_the_new_scene(synthetic_clip, tmp_path):
    clip, _ = synthetic_clip
    estimate_motion(clip)
    _scene(tmp_path / "scene.png")
    prepare_plate(clip, tmp_path / "scene.png")
    result = render(clip)
    frames = decode(clip.out_dir / f"{clip.name}.mp4")
    assert len(frames) == result["frames"] == 8
    # Relighting changes the product's brightness on purpose; its colour (Lab a/b) must stay put.
    product = cv2.cvtColor(frames[0][700:900, 270:370], cv2.COLOR_RGB2LAB).reshape(-1, 3).mean(0)
    original = cv2.cvtColor(np.uint8([[[220, 180, 100]]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    assert np.linalg.norm(product[1:] - original[1:]) < 6
    assert json.loads((clip.root / "render.json").read_text())["visual_review"].startswith("pending")
    assert (clip.out_dir / "review-0000.jpg").exists()


def test_render_refuses_without_a_plate(synthetic_clip):
    clip, _ = synthetic_clip
    estimate_motion(clip)
    with pytest.raises(FileNotFoundError, match="plate"):
        render(clip)


def test_render_refuses_frames_outside_the_clip(synthetic_clip, tmp_path):
    clip, _ = synthetic_clip
    estimate_motion(clip)
    _scene(tmp_path / "scene.png")
    with pytest.raises(ValueError, match="outside"):
        render(clip, background=tmp_path / "scene.png", start=6, count=5)


def test_checks_run_on_a_rendered_clip(synthetic_clip, tmp_path):
    clip, _ = synthetic_clip
    estimate_motion(clip)
    _scene(tmp_path / "scene.png")
    render(clip, background=tmp_path / "scene.png")
    jitter = background_jitter(clip)
    assert len(jitter["high_frequency_std_px"]) == 2
    crops = edge_crops(clip)
    assert (clip.out_dir / "edge-crops.jpg").exists() and crops["frames"][0] == 0


def test_plate_refuses_a_scene_far_brighter_than_the_old_wall(synthetic_clip, tmp_path):
    clip, _ = synthetic_clip
    estimate_motion(clip)
    Image.new("RGB", (720, 960), (250, 250, 250)).save(tmp_path / "white.png")
    with pytest.raises(ValueError, match="--no-match"):
        prepare_plate(clip, tmp_path / "white.png")
    result = prepare_plate(clip, tmp_path / "white.png", match=False)
    assert result["matched"] is False
    plate = np.asarray(Image.open(clip.root / "plate.png"))
    assert plate.mean() > 245  # left as bright as generated
