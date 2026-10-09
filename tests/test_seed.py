import json

import numpy as np
import pytest

from live_swap_bg.seed import best_candidate, clean_union, load_prompts


def _group(**extra):
    return {"name": "bottle", "box": [100, 100, 300, 500], "pos": [[200, 300]], "neg": [], **extra}


def test_prompts_need_groups(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"groups": []}))
    with pytest.raises(ValueError, match="groups"):
        load_prompts(path)


def test_a_point_outside_the_frame_is_refused(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"groups": [_group(pos=[[800, 10]])]}))
    with pytest.raises(ValueError, match="outside the frame"):
        load_prompts(path)


def test_candidate_choice_prefers_the_mask_that_fills_the_box():
    box = [100, 100, 300, 500]
    tight = np.zeros((960, 720), bool)
    tight[110:490, 110:290] = True
    spill = np.ones((960, 720), bool)  # SAM often scores the whole frame as a clean mask
    assert np.array_equal(best_candidate(np.stack([spill, tight]), box), tight)


def test_small_enclosed_hole_is_filled_but_a_marked_gap_stays_open():
    mask = np.zeros((960, 720), bool)
    mask[100:500, 100:300] = True
    mask[200:220, 150:170] = False  # specular-highlight hole
    mask[300:320, 150:170] = False  # real gap, marked with a negative point
    out = clean_union([_group(neg=[[160, 310]])], [mask])
    assert out[210, 160] == 255
    assert out[310, 160] == 0


def test_pieces_without_a_positive_point_are_dropped():
    mask = np.zeros((960, 720), bool)
    mask[100:500, 100:300] = True
    mask[600:700, 500:600] = True  # a second bottle standing on the table
    out = clean_union([_group()], [mask])
    assert out[300, 200] == 255
    assert out[650, 550] == 0
