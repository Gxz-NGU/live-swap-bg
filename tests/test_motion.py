import json

import numpy as np
import pytest
from conftest import make_clip

from live_swap_bg.motion import estimate_background_translation, estimate_motion


def test_camera_follows_the_background_not_the_faster_foreground(synthetic_clip):
    clip, _ = synthetic_clip
    frames = clip.frame_paths()
    result = estimate_background_translation(frames, [clip.masks_dir / p.name for p in frames])
    steps = np.diff(np.array(result["positions"]), axis=0)
    # The wall moves +2, -1 per frame; the product moves +8 and must not pull the estimate.
    assert np.allclose(steps, [2, -1], atol=0.3)
    assert result["background_zoom"] > 1


def test_featureless_wall_stops_and_points_at_static(tmp_path):
    clip, _ = make_clip(tmp_path / "flat", textured=False)
    with pytest.raises(ValueError, match="--static"):
        estimate_motion(clip)


def test_static_keeps_the_plate_still(synthetic_clip):
    clip, _ = synthetic_clip
    result = estimate_motion(clip, static=True)
    assert result["positions"] == [[0.0, 0.0]] * 8
    assert json.loads((clip.root / "motion.json").read_text())["method"] == "static background"


def test_a_noisy_estimate_falls_back_to_a_still_plate(synthetic_clip, monkeypatch):
    clip, _ = synthetic_clip
    noisy = {"positions": [[0.0, 0.0]] * 8, "background_zoom": 1.05, "range_pixels": [9, 9],
             "quality": [{"median_translation_residual": 4.7}] * 7}
    monkeypatch.setattr("live_swap_bg.motion.estimate_background_translation", lambda *_: noisy)
    result = estimate_motion(clip)
    assert result["method"] == "static background"
    assert (clip.root / "motion-lk.json").exists()


def test_wrong_sized_frames_are_refused(synthetic_clip):
    clip, _ = synthetic_clip
    from PIL import Image
    Image.new("RGB", (640, 480)).save(clip.frames_dir / "0003.png")
    frames = clip.frame_paths()
    with pytest.raises(ValueError, match="720x960"):
        estimate_background_translation(frames, [clip.masks_dir / p.name for p in frames])
