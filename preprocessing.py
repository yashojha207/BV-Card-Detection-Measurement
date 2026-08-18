from __future__ import annotations

import cv2
import numpy as np

from config import (
    MASK_THRESHOLD,
    MORPH_KERNEL_SIZE,
    MORPH_OPEN_ITERATIONS,
    MORPH_CLOSE_ITERATIONS,
    MIN_FOREGROUND_AREA_RATIO,
    MAX_FOREGROUND_AREA_RATIO,
)


# ============================================================
# THRESHOLDING
# ============================================================

def threshold_mask(
    mask_gray: np.ndarray,
) -> np.ndarray:
    """
    Convert a soft grayscale segmentation mask into a binary mask.

    Output values:
        0   = background
        255 = foreground
    """

    if mask_gray.ndim != 2:
        raise ValueError(
            "Expected a single-channel grayscale mask."
        )

    if mask_gray.dtype != np.uint8:
        mask_gray = np.clip(
            mask_gray,
            0,
            255,
        ).astype(np.uint8)

    _, binary_mask = cv2.threshold(
        mask_gray,
        MASK_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )

    return binary_mask


# ============================================================
# MORPHOLOGICAL CLEANUP
# ============================================================

def create_morphology_kernel() -> np.ndarray:
    """
    Create the morphology kernel used for opening and closing.
    """

    if MORPH_KERNEL_SIZE < 1:
        raise ValueError(
            "MORPH_KERNEL_SIZE must be at least 1."
        )

    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE,
        ),
    )


def apply_morphological_cleanup(
    binary_mask: np.ndarray,
) -> np.ndarray:
    """
    Remove small foreground noise and fill small gaps.

    Opening:
        removes isolated white regions.

    Closing:
        fills small black holes and breaks in the card mask.
    """

    kernel = create_morphology_kernel()

    cleaned_mask = binary_mask.copy()

    if MORPH_OPEN_ITERATIONS > 0:
        cleaned_mask = cv2.morphologyEx(
            cleaned_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=MORPH_OPEN_ITERATIONS,
        )

    if MORPH_CLOSE_ITERATIONS > 0:
        cleaned_mask = cv2.morphologyEx(
            cleaned_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=MORPH_CLOSE_ITERATIONS,
        )

    return cleaned_mask


# ============================================================
# CONTOUR EXTRACTION
# ============================================================

def find_external_contours(
    binary_mask: np.ndarray,
) -> list[np.ndarray]:
    """
    Extract all external foreground contours from a binary mask.
    """

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    return list(contours)


def select_largest_contour(
    contours: list[np.ndarray],
) -> np.ndarray:
    """
    Select the contour with the greatest enclosed area.
    """

    if not contours:
        raise RuntimeError(
            "No contours were found in the segmentation mask."
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    largest_area = float(
        cv2.contourArea(largest_contour)
    )

    if largest_area <= 0:
        raise RuntimeError(
            "The largest contour has zero area."
        )

    return largest_contour


def validate_largest_contour(
    contour: np.ndarray,
    image_shape: tuple[int, int],
) -> None:
    """
    Confirm that the detected foreground region occupies a
    reasonable fraction of the image.
    """

    image_height, image_width = image_shape

    image_area = float(
        image_height * image_width
    )

    contour_area = float(
        cv2.contourArea(contour)
    )

    area_ratio = contour_area / image_area

    if area_ratio < MIN_FOREGROUND_AREA_RATIO:
        raise RuntimeError(
            "The detected foreground object is too small. "
            f"Area ratio: {area_ratio:.4f}"
        )

    if area_ratio > MAX_FOREGROUND_AREA_RATIO:
        raise RuntimeError(
            "The detected foreground region covers almost the "
            "entire image. Ensure that visible background surrounds "
            "the credit card."
        )


# ============================================================
# COMPONENT MASK
# ============================================================

def create_filled_contour_mask(
    image_shape: tuple[int, int],
    contour: np.ndarray,
) -> np.ndarray:
    """
    Create a clean binary image containing only the selected contour.
    """

    image_height, image_width = image_shape

    component_mask = np.zeros(
        (
            image_height,
            image_width,
        ),
        dtype=np.uint8,
    )

    cv2.drawContours(
        component_mask,
        [contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    return component_mask


# ============================================================
# HOLE FILLING
# ============================================================

def fill_internal_holes(
    binary_mask: np.ndarray,
) -> np.ndarray:
    """
    Fill enclosed black holes inside the foreground object.

    This uses flood filling from the image border, then inverts
    the flooded background to identify internal holes.
    """

    flood_fill_image = binary_mask.copy()

    image_height, image_width = flood_fill_image.shape

    flood_mask = np.zeros(
        (
            image_height + 2,
            image_width + 2,
        ),
        dtype=np.uint8,
    )

    cv2.floodFill(
        flood_fill_image,
        flood_mask,
        seedPoint=(0, 0),
        newVal=255,
    )

    inverted_flood = cv2.bitwise_not(
        flood_fill_image
    )

    filled_mask = cv2.bitwise_or(
        binary_mask,
        inverted_flood,
    )

    return filled_mask


# ============================================================
# CONTOUR SMOOTHING
# ============================================================

def smooth_component_mask(
    component_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply a mild blur and re-thresholding step.

    This removes tiny staircase-like segmentation artefacts while
    preserving the card's general boundary.
    """

    blurred = cv2.GaussianBlur(
        component_mask,
        (5, 5),
        sigmaX=0,
    )

    _, smoothed_mask = cv2.threshold(
        blurred,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return smoothed_mask


def reextract_largest_contour(
    clean_mask: np.ndarray,
) -> np.ndarray:
    """
    Re-extract the largest detailed contour after cleanup.
    """

    contours = find_external_contours(
        clean_mask
    )

    return select_largest_contour(
        contours
    )


# ============================================================
# CONVEX HULL
# ============================================================

def compute_convex_hull(
    contour: np.ndarray,
) -> np.ndarray:
    """
    Compute the contour's convex hull.

    This helps suppress small inward segmentation defects while
    preserving the outer card boundary.
    """

    hull = cv2.convexHull(
        contour,
        returnPoints=True,
    )

    if hull is None or len(hull) < 4:
        raise RuntimeError(
            "Could not compute a valid convex hull."
        )

    return hull


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def clean_segmentation_mask(
    mask_gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline.

    Steps:
        1. Threshold the soft RMBG mask.
        2. Apply morphological opening and closing.
        3. Extract the largest foreground contour.
        4. Validate its size.
        5. Keep only that contour.
        6. Fill internal holes.
        7. Smooth minor segmentation artefacts.
        8. Re-extract the final detailed contour.

    Returns:
        clean_mask:
            Final binary mask containing only the card.

        final_contour:
            Detailed card boundary used by Hough and geometry stages.
    """

    binary_mask = threshold_mask(
        mask_gray
    )

    morph_mask = apply_morphological_cleanup(
        binary_mask
    )

    contours = find_external_contours(
        morph_mask
    )

    largest_contour = select_largest_contour(
        contours
    )

    validate_largest_contour(
        largest_contour,
        morph_mask.shape,
    )

    component_mask = create_filled_contour_mask(
        morph_mask.shape,
        largest_contour,
    )

    filled_mask = fill_internal_holes(
        component_mask
    )

    clean_mask = smooth_component_mask(
        filled_mask
    )

    final_contour = reextract_largest_contour(
        clean_mask
    )

    validate_largest_contour(
        final_contour,
        clean_mask.shape,
    )

    return clean_mask, final_contour