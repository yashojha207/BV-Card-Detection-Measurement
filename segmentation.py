from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from config import MODEL_NAME


# ============================================================
# LOAD RMBG MODEL
# ============================================================

def load_segmenter() -> Any:
    """
    Loads BRIA RMBG-1.4.

    Automatically uses CUDA if available.
    """

    device = 0 if torch.cuda.is_available() else -1

    print("\nLoading RMBG-1.4...")
    print(f"Device: {'CUDA' if device == 0 else 'CPU'}")

    segmenter = pipeline(
        task="image-segmentation",
        model=MODEL_NAME,
        trust_remote_code=True,
        device=device,
    )

    print("Model loaded successfully.")

    return segmenter


# ============================================================
# EXTRACT MASK
# ============================================================

def extract_mask(
    prediction: Any,
) -> Image.Image:
    """
    Handles different return formats from transformers.
    """

    if isinstance(prediction, Image.Image):
        return prediction

    if isinstance(prediction, dict):

        if "mask" in prediction:
            return prediction["mask"]

    if isinstance(prediction, list):

        if len(prediction) == 0:
            raise RuntimeError("Empty prediction list.")

        first = prediction[0]

        if isinstance(first, Image.Image):
            return first

        if isinstance(first, dict):

            if "mask" in first:
                return first["mask"]

    raise RuntimeError(
        "Unknown segmentation output format."
    )


# ============================================================
# SEGMENT FOREGROUND
# ============================================================

def segment_foreground(
    segmenter,
    image_pillow: Image.Image,
) -> np.ndarray:
    """
    Runs RMBG.

    Returns a grayscale mask.
    """

    prediction = segmenter(
        image_pillow,
        return_mask=True,
    )

    mask = extract_mask(prediction)

    mask = mask.convert("L")

    mask = np.array(
        mask,
        dtype=np.uint8,
    )

    expected_height = image_pillow.height
    expected_width = image_pillow.width

    if mask.shape != (expected_height, expected_width):

        mask = cv2.resize(
            mask,
            (
                expected_width,
                expected_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    return mask


# ============================================================
# OPTIONAL DEBUG VISUALIZATION
# ============================================================

def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlays the mask on the original image.
    Useful for debugging.
    """

    overlay = image_rgb.copy()

    overlay[mask > 128] = (
        0,
        255,
        0,
    )

    result = cv2.addWeighted(
        image_rgb,
        1 - alpha,
        overlay,
        alpha,
        0,
    )

    return result


# ============================================================
# MASK STATISTICS
# ============================================================

def foreground_percentage(
    mask: np.ndarray,
) -> float:
    """
    Percentage of pixels classified
    as foreground.
    """

    foreground = np.count_nonzero(mask > 128)

    total = mask.shape[0] * mask.shape[1]

    return 100.0 * foreground / total


# ============================================================
# SANITY CHECK
# ============================================================

def validate_mask(
    mask: np.ndarray,
) -> None:
    """
    Basic sanity checks.
    """

    if mask.ndim != 2:
        raise RuntimeError(
            "Segmentation mask must be grayscale."
        )

    if mask.dtype != np.uint8:
        raise RuntimeError(
            "Segmentation mask must be uint8."
        )

    percent = foreground_percentage(mask)

    if percent < 0.5:
        print(
            "Warning: foreground area is very small."
        )

    if percent > 98:
        print(
            "Warning: segmentation covered almost the entire image."
        )