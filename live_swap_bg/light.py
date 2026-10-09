"""Make a product cut from its own shot sit under the background plate's light.

The product carries the light of the shot it came from and casts nothing, so on a plate with a real key light
it reads as pasted on: the side facing the plate's light gets brighter, the far side darker, and the product
casts a soft shadow away from the light.
"""

from __future__ import annotations

import cv2
import numpy as np

SHADE = 0.30  # how far the lit and unlit sides of the product part
SHADOW = 0.55  # how dark the cast shadow gets at its core
CONTACT = 0.55  # how dark the surface goes right against a resting product
CHROMA_MATCH = 0.25  # how much of the product-to-plate colour gap to close
# The product has to stay its real colour for the buyer, so the warm/cool nudge is
# hard-clamped below the CIE76 just-noticeable difference. A share
# alone does not bound it: 0.25 of a lavender bottle against a neutral plate is 7.3.
CHROMA_SHIFT_LIMIT = 2.0
EXPOSURE_MATCH = 0.75


def key_light_direction(plate: np.ndarray) -> np.ndarray:
    """Unit vector from the frame centre toward where the plate's light comes from."""
    grey = cv2.cvtColor(np.uint8(plate), cv2.COLOR_RGB2GRAY).astype(np.float32)
    rows, columns = grey.shape
    bright_y, bright_x = np.nonzero(grey >= np.percentile(grey, 90))
    direction = np.array([bright_x.mean() - columns / 2, bright_y.mean() - rows / 2], np.float32)
    return direction / (np.linalg.norm(direction) + 1e-6)


def product_light_span(alpha: np.ndarray, light: np.ndarray) -> float:
    """How far the product reaches along the light direction, in pixels.

    The ramp has to be normalised across the product, not the frame: a product
    covering a third of the picture would otherwise only ever see a third of it.
    """
    ys, xs = np.nonzero(alpha > 0.5)
    if not len(ys):
        raise ValueError("cannot measure a product light span on an empty alpha")
    reach = (xs - xs.mean()) * light[0] + (ys - ys.mean()) * light[1]
    return max(float(np.percentile(np.abs(reach), 98)), 1.0)


def plate_floor_lab(plate: np.ndarray) -> np.ndarray:
    """Median Lab of the plate's lower half, where a held or resting product sits."""
    lab = cv2.cvtColor(np.uint8(plate), cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.median(lab[lab.shape[0] // 2 :].reshape(-1, 3), 0)


def relight(
    product: np.ndarray,
    alpha: np.ndarray,
    plate: np.ndarray,
    *,
    light: np.ndarray,
    span: float,
    floor_lab: np.ndarray,
    rests_on_surface: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the product under the plate's light, and the plate under its shadow."""
    height, width = alpha.shape
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    ys, xs = np.nonzero(alpha > 0.5)
    if not len(ys):
        raise ValueError("cannot relight an empty alpha")
    facing = np.clip(((columns - xs.mean()) * light[0] + (rows - ys.mean()) * light[1]) / span, -1, 1)

    lab = cv2.cvtColor(np.uint8(product), cv2.COLOR_RGB2LAB).astype(np.float32)
    here = np.median(lab[alpha > 0.9], 0)
    lab[:, :, 0] *= np.clip(floor_lab[0] / max(here[0], 1), 0.82, 1.12) ** EXPOSURE_MATCH
    shift = (floor_lab[1:] - here[1:]) * CHROMA_MATCH * EXPOSURE_MATCH
    reach = float(np.linalg.norm(shift))
    if reach > CHROMA_SHIFT_LIMIT:
        shift = shift * (CHROMA_SHIFT_LIMIT / reach)
    lab[:, :, 1] += shift[0]
    lab[:, :, 2] += shift[1]
    lit = cv2.cvtColor(np.clip(lab, 0, 255).astype("uint8"), cv2.COLOR_LAB2RGB).astype(np.float32)
    lit = np.clip(lit * (1 + SHADE * facing)[:, :, None], 0, 255)

    shift = -light * span * 0.55
    cast = cv2.warpAffine(alpha, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (width, height), borderValue=0)
    cast = cv2.GaussianBlur(cast, (0, 0), span * 0.22)
    # The shadow belongs to the plate. Keeping it out from under the product with (1 - alpha) left the plate
    # behind a half-covered edge pixel half as dark as the plate beside it: a pale rim on the shadow side.
    darkening = SHADOW * cast
    if rests_on_surface:
        # It is the bottom of the OUTLINE that touches, not the bottom row: a
        # package whose front flap runs off the frame has every column bottoming
        # out at the last row, so a downward shadow would fall off-screen while
        # its base corners stay cut out.
        top, base = ys.min(), ys.max()
        lowness = np.clip((rows - (top + 0.65 * (base - top))) / max(0.35 * (base - top), 1), 0, 1)
        hug = cv2.GaussianBlur(alpha, (0, 0), span * 0.05) * lowness
        darkening = np.maximum(darkening, CONTACT * hug)
    return lit, plate * np.clip(1 - darkening, 0, 1)[:, :, None]
