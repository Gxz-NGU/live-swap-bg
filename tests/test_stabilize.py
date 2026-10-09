import numpy as np
from PIL import Image

from live_swap_bg import stabilize


def _write_alpha(clip, index, value):
    alpha = np.zeros((960, 720), np.float32)
    alpha[100:200, 100:200] = value
    Image.fromarray(np.rint(alpha * 65535).astype(np.uint16)).save(clip.alpha_dir / f"{index:04d}.png")


def test_a_one_frame_alpha_dropout_is_filled(synthetic_clip):
    clip, _ = synthetic_clip
    for k in range(8):
        _write_alpha(clip, k, 0.1 if k == 4 else 1.0)
    report = stabilize.fill_alpha_dropouts(clip)
    assert report["one_frame_dips_before"] > 0 and report["one_frame_dips_after"] == 0
    assert clip.alpha(4)[150, 150] > 0.99


def test_dropout_fill_never_lowers_alpha(synthetic_clip):
    clip, _ = synthetic_clip
    for k in range(8):
        _write_alpha(clip, k, 1.0 if k == 4 else 0.2)  # a one-frame spike up must stay
    stabilize.fill_alpha_dropouts(clip)
    assert clip.alpha(4)[150, 150] > 0.99


def test_rerunning_starts_from_the_untouched_input(synthetic_clip):
    clip, _ = synthetic_clip
    for k in range(8):
        _write_alpha(clip, k, 0.1 if k == 4 else 1.0)
    stabilize.fill_alpha_dropouts(clip)
    _write_alpha(clip, 4, 0.5)  # someone edits the output; a rerun must still read the backup
    stabilize.fill_alpha_dropouts(clip)
    assert clip.alpha(4)[150, 150] > 0.99
    assert (clip.root / "alpha-before-dropfill").is_dir()


def test_mask_vote_removes_a_one_frame_flicker(synthetic_clip):
    clip, _ = synthetic_clip
    for k in range(8):
        mask = np.zeros((960, 720), np.uint8)
        mask[300:400, 300:400] = 255
        if k == 3:
            mask[300:400, 400:420] = 255  # a sliver that appears for one frame only
        Image.fromarray(mask).save(clip.masks_dir / f"{k:04d}.png")
    stabilize.vote_masks(clip)
    assert not clip.mask(3)[350, 410]
    assert clip.mask(3)[350, 350]
