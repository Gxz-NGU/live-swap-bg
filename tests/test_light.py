import numpy as np
import pytest

from live_swap_bg.light import (
    key_light_direction,
    plate_floor_lab,
    product_light_span,
    relight,
)


def _plate_lit_from_upper_left() -> np.ndarray:
    plate = np.full((960, 720, 3), 60, np.uint8)
    plate[:300, :250] = 240
    return plate


def _centre_product() -> tuple[np.ndarray, np.ndarray]:
    product = np.full((960, 720, 3), 170, np.uint8)
    alpha = np.zeros((960, 720), np.float32)
    alpha[400:700, 260:460] = 1.0
    return product, alpha


def test_key_light_points_at_the_bright_corner():
    light = key_light_direction(_plate_lit_from_upper_left())
    assert light[0] < -0.5 and light[1] < -0.2
    assert pytest.approx(1.0, abs=1e-3) == float(np.linalg.norm(light))


def test_light_span_measures_the_product_not_the_frame():
    _, alpha = _centre_product()
    light = np.array([-1.0, 0.0], np.float32)
    # the product is 200 px wide, so half of it is 100 either side of its centre
    assert 80 < product_light_span(alpha, light) < 105


def test_empty_alpha_fails_instead_of_returning_a_span():
    with pytest.raises(ValueError, match="empty alpha"):
        product_light_span(np.zeros((960, 720), np.float32), np.array([-1.0, 0.0], np.float32))


def test_the_side_facing_the_light_ends_up_brighter_than_the_far_side():
    plate = _plate_lit_from_upper_left()
    product, alpha = _centre_product()
    light = key_light_direction(plate)
    lit, _ = relight(
        product, alpha, plate.astype(np.float32),
        light=light, span=product_light_span(alpha, light), floor_lab=plate_floor_lab(plate),
    )
    near = lit[500:600, 270:310].mean()
    far = lit[500:600, 410:450].mean()
    assert near > far * 1.15


def test_the_product_casts_a_shadow_away_from_the_light():
    plate = _plate_lit_from_upper_left()
    product, alpha = _centre_product()
    light = key_light_direction(plate)
    _, shaded = relight(
        product, alpha, plate.astype(np.float32),
        light=light, span=product_light_span(alpha, light), floor_lab=plate_floor_lab(plate),
    )
    original = plate.astype(np.float32)
    away = (original - shaded)[550:650, 470:560].mean()
    toward = (original - shaded)[400:500, 160:250].mean()
    assert away > 5 and away > toward


def test_a_resting_product_darkens_the_surface_along_its_lower_outline():
    plate = _plate_lit_from_upper_left()
    product, alpha = _centre_product()
    light = key_light_direction(plate)
    span = product_light_span(alpha, light)
    floor = plate_floor_lab(plate)
    _, free = relight(product, alpha, plate.astype(np.float32), light=light, span=span, floor_lab=floor)
    _, resting = relight(
        product, alpha, plate.astype(np.float32), light=light, span=span, floor_lab=floor,
        rests_on_surface=True,
    )
    # The band hugs the outline and is only a few pixels wide, so measure its
    # peak: a mean over any window big enough to find it is mostly product.
    added = (free - resting).mean(2)
    base = added[600:710, 240:262].max()
    top = added[400:470, 240:262].max()
    assert base > 3 and base > top


def test_the_colour_shift_stays_under_the_just_noticeable_difference():
    plate = _plate_lit_from_upper_left()
    product = np.zeros((960, 720, 3), np.uint8)
    product[:, :] = (150, 140, 210)  # a lavender bottle against a warm plate
    alpha = np.zeros((960, 720), np.float32)
    alpha[400:700, 260:460] = 1.0
    import cv2

    light = key_light_direction(plate)
    lit, _ = relight(
        product, alpha, plate.astype(np.float32),
        light=light, span=product_light_span(alpha, light), floor_lab=plate_floor_lab(plate),
    )
    solid = alpha > 0.9
    before = cv2.cvtColor(product, cv2.COLOR_RGB2LAB).astype(np.float32)[solid][:, 1:].mean(0)
    after = cv2.cvtColor(np.uint8(lit), cv2.COLOR_RGB2LAB).astype(np.float32)[solid][:, 1:].mean(0)
    assert float(np.linalg.norm(after - before)) < 2.3


def test_a_soft_edge_on_the_shadow_side_is_not_brighter_than_either_side():
    import cv2

    plate = np.full((960, 720, 3), 200, np.float32)  # a bright wall
    product = np.full((960, 720, 3), 120, np.uint8)
    hard = np.zeros((960, 720), np.float32)
    hard[400:700, 260:460] = 1.0
    alpha = cv2.GaussianBlur(hard, (0, 0), 2.0)
    light = np.array([-1.0, 0.0], np.float32)  # lit from the left, so the shadow falls on the right
    lit, shaded = relight(
        product, alpha, plate, light=light, span=product_light_span(alpha, light), floor_lab=plate_floor_lab(np.uint8(plate)),
    )
    shown = (lit * alpha[..., None] + shaded * (1 - alpha[..., None])).mean(2)
    edge = shown[550, 450:472]  # across the right-hand outline, inside to outside
    # the plate behind a half-covered pixel is in the same shadow as the plate beside it, so no pale rim
    assert edge.max() <= max(edge[0], edge[-1]) + 1

