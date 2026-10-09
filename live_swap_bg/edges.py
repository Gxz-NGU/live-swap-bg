"""Edge repair for the compositor.

The renderer pastes the source RGB through the matte, so a semi-transparent edge pixel still carries the old
background colour (a halo), and a gap the matte filled in shows the old background.
Each function works on one 720x960 frame: alpha is float 0..1, rgb is float 0..255.
"""

import cv2
import numpy as np

ALPHA_FLOOR = 0.08
ALPHA_CEILING = 0.92
DARK_FRINGE_BAND_PX = 6
DARKER_THAN_BACKGROUND = 0.15  # in units of the foreground-to-background luma gap
DECONTAMINATE_BAND_PX = 8
DARK_BACKGROUND_LUMA = 110.0
DARK_BACKGROUND_MIN_CONTRAST = 40.0


def _require_frame(alpha: np.ndarray, rgb: np.ndarray) -> None:
    if alpha.ndim != 2 or rgb.shape != (*alpha.shape, 3):
        raise ValueError(f"alpha {alpha.shape} and rgb {rgb.shape} must describe the same frame")
    for name, values in (("alpha", alpha), ("rgb", rgb)):
        if not np.isfinite(values).all():
            first = tuple(int(i) for i in np.argwhere(~np.isfinite(values))[0])
            raise ValueError(f"{name} has a non-finite value at index {first}")


def tighten_alpha(alpha: np.ndarray, floor: float = ALPHA_FLOOR, ceiling: float = ALPHA_CEILING) -> np.ndarray:
    """Drop the faint outer rim of the matte and make the inner rim opaque."""
    if not 0.0 <= floor < ceiling <= 1.0:
        raise ValueError(f"alpha floor {floor} and ceiling {ceiling} must satisfy 0 <= floor < ceiling <= 1")
    return np.clip((alpha - floor) / (ceiling - floor), 0.0, 1.0)


def _blur_fusion_step(image, foreground, background, alpha, radius):
    weight = alpha[..., None]
    blurred_alpha = cv2.blur(alpha, (radius, radius))[..., None]
    blurred_foreground = cv2.blur(foreground * weight, (radius, radius)) / (blurred_alpha + 1e-5)
    blurred_background = cv2.blur(background * (1 - weight), (radius, radius)) / ((1 - blurred_alpha) + 1e-5)
    estimate = blurred_foreground + weight * (image - weight * blurred_foreground - (1 - weight) * blurred_background)
    return np.clip(estimate, 0.0, 1.0), blurred_background


def decontaminate_foreground(rgb: np.ndarray, alpha: np.ndarray, radius: int = 61) -> np.ndarray:
    """Replace the colour of fractional-alpha pixels next to the old background with the estimated foreground colour.

    Blur fusion; only pixels within DECONTAMINATE_BAND_PX of the outside are touched, so a translucent product body
    keeps its own colour. Limit: translucent glass or frosted parts inside that band read as old-background
    halo and change colour. The two-pixel ring around those pixels is repainted too: resampling the straight
    RGB would blend the old wall colour just outside the matte back into the edge.
    """
    image = rgb.astype(np.float32) / 255.0
    foreground, background = _blur_fusion_step(image, image, image, alpha, radius)
    foreground = _blur_fusion_step(image, foreground, background, alpha, 7)[0]
    distance_to_outside = cv2.distanceTransform((alpha >= 0.02).astype(np.uint8), cv2.DIST_L2, 3)
    fractional = ((alpha > 0.01) & (alpha < 0.99)).astype(np.uint8)
    repaint = ((cv2.dilate(fractional, np.ones((5, 5), np.uint8)) > 0) & (distance_to_outside <= DECONTAMINATE_BAND_PX))[..., None]
    return np.where(repaint, foreground, image) * 255.0


def fill_enclosed_gaps(alpha: np.ndarray, rgb: np.ndarray, max_pixels: int):
    """Fill background gaps fully enclosed by the foreground (thumb/bottle crevice) with inpainted colour.

    Only for gaps a person has confirmed show the old background and not a real shadow or slit. Alpha changes
    only inside the selected gaps; their foreground rim gets the inpainted colour but keeps its alpha. The
    colour comes from the product's own pixels only, never from the old background around the silhouette.
    """
    height, width = alpha.shape
    background = (alpha < 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4)
    gaps = np.zeros((height, width), bool)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area <= max_pixels and x > 0 and y > 0 and x + w < width and y + h < height:
            gaps |= labels == label
    if not gaps.any():
        return alpha, rgb
    rim = cv2.dilate(gaps.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    repaint = gaps | (rim & (alpha >= 0.5))
    colour_unavailable = repaint | (alpha < 0.5)
    if colour_unavailable.all():
        raise ValueError(f"no foreground pixel is left to take a fill colour from for the {int(gaps.sum())} gap pixels")
    filled = cv2.inpaint(np.clip(rgb, 0, 255).astype(np.uint8), colour_unavailable.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return np.where(gaps, 1.0, alpha).astype(alpha.dtype), np.where(repaint[..., None], filled, rgb).astype(rgb.dtype)


def cap_dark_fringe_alpha(alpha: np.ndarray, rgb: np.ndarray, band_px: int = DARK_FRINGE_BAND_PX) -> np.ndarray:
    """Cap the alpha of edge pixels at the foreground coverage their colour shows, when the old background is dark.

    Where a pixel sits between the local old-background luma and the local foreground luma says how much
    foreground it holds: a dark fringe the matte swallowed gets a low alpha, while an edge whose alpha already
    agrees with its colour keeps its own soft profile, so edges stay anti-aliased. Light backgrounds, dark
    foregrounds on dark backgrounds, and dark objects on light backgrounds are left alone, and so is anything
    darker than the old background, which no mix of the two can produce. Limits: a genuinely dark outline on a bright
    object shot against a dark background looks the same as the fringe and is capped with it; and the
    old-background luma is a local mean, so a dark rim hugging the object with a much brighter wall a few
    pixels beyond it reads too bright and the edge there is squeezed harder than it should be.
    """
    luma = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    inside = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    band = (alpha > 0.003) & (inside <= band_px)
    interior = ((inside >= 3) & (alpha > 0.98)).astype(np.float32)
    exterior = (alpha < 0.02).astype(np.float32)
    interior_weight = cv2.GaussianBlur(interior, (0, 0), 5)
    exterior_weight = cv2.GaussianBlur(exterior, (0, 0), 6)
    foreground_luma = cv2.GaussianBlur(luma * interior, (0, 0), 5) / np.maximum(interior_weight, 1e-6)
    background_luma = cv2.GaussianBlur(luma * exterior, (0, 0), 6) / np.maximum(exterior_weight, 1e-6)
    applies = (
        band
        & (interior_weight > 0.05)
        & (exterior_weight > 0.05)
        & (background_luma < DARK_BACKGROUND_LUMA)
        & (foreground_luma - background_luma > DARK_BACKGROUND_MIN_CONTRAST)
    )
    position = (luma - background_luma) / np.maximum(foreground_luma - background_luma, 1e-3)
    # A mix of foreground and old background is never darker than the background. A pixel well below it is a
    # dark detail or a shadow on the foreground (a nail's dark side); capping it would cut a hole that leaves a
    # loose sliver flickering beside the edge.
    mixed = position > -DARKER_THAN_BACKGROUND
    return np.where(applies & mixed, np.minimum(alpha, np.clip(position, 0.0, 1.0)), alpha)


def repair_edges(alpha: np.ndarray, rgb: np.ndarray, max_gap_pixels: int = 0):
    """Run the whole repair on one frame and return the new (alpha, rgb); gaps are only filled when max_gap_pixels > 0."""
    _require_frame(alpha, rgb)
    if max_gap_pixels > 0:
        alpha, rgb = fill_enclosed_gaps(alpha, rgb, max_gap_pixels)
    capped_alpha = cap_dark_fringe_alpha(alpha, rgb)  # how much foreground a pixel really holds
    return tighten_alpha(capped_alpha), decontaminate_foreground(rgb, capped_alpha)

