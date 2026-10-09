import cv2
import numpy as np
import pytest

from live_swap_bg.edges import (
    cap_dark_fringe_alpha,
    decontaminate_foreground,
    fill_enclosed_gaps,
    repair_edges,
    tighten_alpha,
)

SIZE = (96, 96)


def _soft_square(background_rgb, foreground_rgb=(240, 240, 240), edge_sigma=2.0):
    """A square whose soft edge mixes the two colours the way a camera does."""
    hard = np.zeros(SIZE, np.float32)
    hard[30:66, 30:66] = 1.0
    alpha = cv2.GaussianBlur(hard, (0, 0), edge_sigma)
    foreground = np.array(foreground_rgb, np.float32)
    background = np.array(background_rgb, np.float32)
    return alpha, alpha[..., None] * foreground + (1 - alpha[..., None]) * background


def _soft_edge_pixels(alpha: np.ndarray) -> int:
    return int(((alpha > 0.1) & (alpha < 0.9)).sum())


def _footprint(columns: slice, rows: slice) -> np.ndarray:
    union = np.zeros((960, 720), np.uint8)
    union[rows, columns] = 1
    return union


def test_tighten_alpha_drops_the_faint_rim_and_closes_the_inner_rim():
    alpha = np.array([0.0, 0.05, 0.5, 0.95, 1.0], np.float32)
    tightened = tighten_alpha(alpha)
    assert tightened[0] == tightened[1] == 0.0
    assert tightened[3] == tightened[4] == 1.0
    assert 0.4 < tightened[2] < 0.6


def test_tighten_alpha_refuses_a_floor_that_is_not_below_the_ceiling():
    with pytest.raises(ValueError, match="floor 0.5 and ceiling 0.5"):
        tighten_alpha(np.zeros(SIZE, np.float32), floor=0.5, ceiling=0.5)


def test_edge_pixels_lose_the_old_background_colour():
    alpha, rgb = _soft_square(background_rgb=(200, 0, 0))
    edge = (alpha > 0.2) & (alpha < 0.8)
    assert rgb[..., 1][edge].mean() < 190  # a red halo: green is far below the foreground's 240
    clean = decontaminate_foreground(rgb, alpha)
    assert clean[..., 1][edge].mean() > 215


def test_pixels_that_are_fully_inside_or_outside_keep_their_colour():
    alpha, rgb = _soft_square(background_rgb=(200, 0, 0))
    clean = decontaminate_foreground(rgb, alpha)
    assert np.abs(clean[48, 48] - rgb[48, 48]).max() < 1
    assert np.abs(clean[5, 5] - rgb[5, 5]).max() < 1


def test_translucent_body_far_from_the_old_background_keeps_its_colour():
    alpha = np.zeros((120, 120), np.float32)
    alpha[20:100, 20:100] = 1.0
    alpha[45:75, 45:75] = 0.6  # a glass window in the middle of the product
    rgb = np.full((120, 120, 3), 200, np.float32)
    rgb[45:75, 45:75] = (90, 140, 180)
    clean = decontaminate_foreground(rgb, alpha)
    assert np.abs(clean[60, 60] - rgb[60, 60]).max() < 1


def test_only_pixels_within_the_edge_band_are_repainted():
    alpha = np.zeros((120, 120), np.float32)
    alpha[20:100, 20:100] = 1.0
    translucent_rows = (25, 27, 29, 31, 40)  # 6, 8, 10, 12 and 21 px from the outside
    rgb = np.full((120, 120, 3), 200, np.float32)
    for row in translucent_rows:
        alpha[row, 30:90] = 0.6
        rgb[row, 30:90] = (90, 140, 180)
    clean = decontaminate_foreground(rgb, alpha)
    repainted_rows = set(np.flatnonzero((np.abs(clean - rgb) > 0.5).any(axis=(1, 2))).tolist())
    assert {25, 27} <= repainted_rows
    assert not repainted_rows & {29, 31, 40}


def test_pixels_just_outside_the_matte_lose_the_old_background_colour_too():
    alpha, rgb = _soft_square(background_rgb=(200, 0, 0))
    fractional = ((alpha > 0.01) & (alpha < 0.99)).astype(np.uint8)
    ring = (alpha <= 0.01) & (cv2.dilate(fractional, np.ones((5, 5), np.uint8)) > 0)
    assert ring.any() and rgb[..., 1][ring].mean() < 5  # pure wall: scaling would blend this red into the edge
    assert decontaminate_foreground(rgb, alpha)[..., 1][ring].mean() > 215


def test_repair_estimates_colour_with_the_capped_alpha_before_tightening_it():
    alpha, rgb = _soft_square(background_rgb=(200, 0, 0), edge_sigma=1.0)
    edge = (alpha > 0.93) & (alpha < 0.99)  # opaque after tightening, but still tinted by the old wall
    repaired_alpha, repaired_rgb = repair_edges(alpha, rgb)
    assert edge.any()
    assert (repaired_alpha[edge] == 1.0).all()
    assert repaired_rgb[..., 1][edge].min() > rgb[..., 1][edge].max()


def test_a_clean_soft_edge_on_a_dark_wall_keeps_its_anti_aliasing():
    alpha, rgb = _soft_square(background_rgb=(30, 30, 30), edge_sigma=1.0)
    repaired_alpha, _ = repair_edges(alpha, rgb)
    assert _soft_edge_pixels(repaired_alpha) >= 0.8 * _soft_edge_pixels(alpha)  # squeezed to a hard edge it stair-steps


def test_a_matte_that_swallowed_a_dark_fringe_is_pulled_back_to_the_object():
    hard = np.zeros(SIZE, np.float32)
    hard[30:66, 30:66] = 1.0
    swallowing = np.zeros(SIZE, np.float32)
    swallowing[27:69, 27:69] = 1.0  # the matte reaches 3 px into the wall
    true_alpha = cv2.GaussianBlur(hard, (0, 0), 1.0)
    foreground = np.float32((240, 240, 240))
    rgb = true_alpha[..., None] * foreground + (1 - true_alpha[..., None]) * np.float32((30, 30, 30))
    matte = cv2.GaussianBlur(swallowing, (0, 0), 1.0)
    fringe = (matte > 0.8) & (true_alpha < 0.1)
    assert fringe.any()
    repaired_alpha, repaired_rgb = repair_edges(matte, rgb)
    assert repaired_alpha[fringe].max() < 0.05
    new_wall = np.float32((80, 100, 140))  # the colours must also be the object's, not the old wall's
    ideal = tighten_alpha(true_alpha)[..., None] * foreground + (1 - tighten_alpha(true_alpha)[..., None]) * new_wall
    shown = repaired_alpha[..., None] * repaired_rgb + (1 - repaired_alpha[..., None]) * new_wall
    transition = (true_alpha > 0.05) & (true_alpha < 0.95)
    assert np.abs(shown - ideal).max(axis=2)[transition].mean() < 8


def test_small_enclosed_gap_is_filled_with_the_surrounding_colour():
    alpha = np.ones(SIZE, np.float32)
    alpha[40:46, 40:46] = 0
    rgb = np.full((*SIZE, 3), (200, 120, 80), np.float32)
    rgb[40:46, 40:46] = (90, 90, 90)  # the old background seen through the gap
    filled_alpha, filled_rgb = fill_enclosed_gaps(alpha, rgb, max_pixels=100)
    assert filled_alpha[43, 43] == 1.0
    assert np.abs(filled_rgb[43, 43] - (200, 120, 80)).max() < 12


def test_gap_larger_than_the_limit_is_left_open():
    alpha = np.ones(SIZE, np.float32)
    alpha[30:60, 30:60] = 0
    rgb = np.zeros((*SIZE, 3), np.float32)
    kept_alpha, kept_rgb = fill_enclosed_gaps(alpha, rgb, max_pixels=100)
    assert kept_alpha[45, 45] == 0.0
    assert kept_rgb is rgb


def test_background_touching_the_frame_edge_is_not_a_gap():
    alpha = np.ones(SIZE, np.float32)
    alpha[0:6, 40:46] = 0
    kept_alpha, _ = fill_enclosed_gaps(alpha, np.zeros((*SIZE, 3), np.float32), max_pixels=100)
    assert kept_alpha[2, 43] == 0.0


def test_filling_a_gap_next_to_the_outline_never_grows_the_silhouette():
    alpha = np.zeros(SIZE, np.float32)
    alpha[10:60, 10:60] = 1.0
    alpha[11, 30] = 0  # one-pixel gap, one pixel inside the top outline
    filled, _ = fill_enclosed_gaps(alpha, np.full((*SIZE, 3), 150, np.float32), max_pixels=1)
    assert filled[11, 30] == 1.0
    assert filled[9, 30] == 0.0
    assert filled.sum() == alpha.sum() + 1


def test_a_neighbouring_gap_over_the_limit_keeps_all_of_its_pixels():
    alpha = np.ones(SIZE, np.float32)
    alpha[30, 30] = 0  # fillable, one pixel
    alpha[30:36, 32:38] = 0  # too big, two pixels away
    filled, _ = fill_enclosed_gaps(alpha, np.full((*SIZE, 3), 150, np.float32), max_pixels=1)
    assert filled[30, 30] == 1.0
    assert (filled[30:36, 32:38] == 0.0).all()


def test_gap_colour_never_comes_from_the_old_background_outside_the_silhouette():
    alpha = np.zeros(SIZE, np.float32)
    alpha[10:60, 10:60] = 1.0
    alpha[11, 30] = 0  # one pixel below the top outline
    rgb = np.full((*SIZE, 3), (200, 0, 0), np.float32)  # the wall
    rgb[alpha > 0.5] = (240, 240, 240)
    _, filled_rgb = fill_enclosed_gaps(alpha, rgb, max_pixels=1)
    assert np.abs(filled_rgb[11, 30] - 240).max() < 12  # the gap
    assert np.abs(filled_rgb[10, 30] - 240).max() < 12  # the outline pixel next to it


def test_gap_fill_refuses_when_no_foreground_colour_is_left_to_sample():
    alpha = np.zeros(SIZE, np.float32)
    alpha[30:33, 30:33] = 1.0
    alpha[31, 31] = 0  # the whole foreground is the gap's own rim
    rgb = np.full((*SIZE, 3), (0, 0, 255), np.float32)
    rgb[alpha > 0.5] = (210, 130, 70)
    with pytest.raises(ValueError, match="no foreground pixel is left.*1 gap pixels"):
        fill_enclosed_gaps(alpha, rgb, max_pixels=1)


def test_dark_fringe_on_a_dark_old_background_is_capped():
    rgb = np.full((*SIZE, 3), 10, np.float32)
    rgb[27:69, 27:69] = 12  # fringe the matte swallowed
    rgb[30:66, 30:66] = 200
    alpha = np.zeros(SIZE, np.float32)
    alpha[27:69, 27:69] = 1.0
    capped = cap_dark_fringe_alpha(alpha, rgb)
    assert capped[28, 48] < 0.1  # fringe row
    assert capped[48, 48] == 1.0  # object interior
    assert (capped <= alpha).all()  # the cap only ever lowers alpha


def test_a_detail_darker_than_the_old_background_keeps_its_alpha():
    rgb = np.full((*SIZE, 3), 60, np.float32)  # dark wall
    rgb[30:66, 30:66] = 200
    rgb[36:60, 33] = 20  # a nail's dark side, darker than the wall, so no mix of nail and wall
    alpha = np.zeros(SIZE, np.float32)
    alpha[30:66, 30:66] = 1.0
    capped = cap_dark_fringe_alpha(alpha, rgb)
    assert (capped[36:60, 33] == 1.0).all()  # no hole, so no loose sliver at columns 30-32


def test_dark_object_on_a_light_background_is_untouched():
    rgb = np.full((*SIZE, 3), 220, np.float32)
    rgb[30:66, 30:66] = 40
    alpha = np.zeros(SIZE, np.float32)
    alpha[28:68, 28:68] = 1.0
    assert np.array_equal(cap_dark_fringe_alpha(alpha, rgb), alpha)


def test_low_contrast_dark_object_on_a_dark_background_is_untouched():
    rgb = np.full((*SIZE, 3), 20, np.float32)
    rgb[30:66, 30:66] = 40
    alpha = np.zeros(SIZE, np.float32)
    alpha[28:68, 28:68] = 1.0
    assert np.array_equal(cap_dark_fringe_alpha(alpha, rgb), alpha)


def test_repair_fills_enclosed_gaps_only_when_asked():
    alpha = np.ones(SIZE, np.float32)
    alpha[40:46, 40:46] = 0
    rgb = np.full((*SIZE, 3), (200, 120, 80), np.float32)
    left_open, _ = repair_edges(alpha, rgb)
    assert left_open[43, 43] == 0.0
    filled, _ = repair_edges(alpha, rgb, max_gap_pixels=100)
    assert filled[43, 43] > 0.9


def test_repair_keeps_alpha_and_colour_in_range():
    alpha, rgb = _soft_square(background_rgb=(10, 10, 10))
    repaired_alpha, repaired_rgb = repair_edges(alpha, rgb)
    assert repaired_alpha.shape == SIZE and repaired_rgb.shape == (*SIZE, 3)
    assert np.isfinite(repaired_alpha).all() and np.isfinite(repaired_rgb).all()
    assert repaired_alpha.min() >= 0.0 and repaired_alpha.max() <= 1.0
    assert repaired_rgb.min() >= 0.0 and repaired_rgb.max() <= 255.0


def test_repair_refuses_non_finite_input_with_the_index():
    alpha, rgb = _soft_square(background_rgb=(10, 10, 10))
    alpha[3, 4] = np.nan
    with pytest.raises(ValueError, match=r"alpha has a non-finite value at index \(3, 4\)"):
        repair_edges(alpha, rgb)


def test_repair_refuses_alpha_and_rgb_of_different_frames():
    alpha, rgb = _soft_square(background_rgb=(10, 10, 10))
    with pytest.raises(ValueError, match="must describe the same frame"):
        repair_edges(alpha[:-1], rgb)

