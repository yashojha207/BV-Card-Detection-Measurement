from __future__ import annotations

import cv2
import numpy as np

from config import (
    OUTPUT_CARD_WIDTH,
    OUTPUT_CARD_HEIGHT,
)


def destination_corners() -> np.ndarray:
    """
    Return destination coordinates for the rectified card.
    """

    width = OUTPUT_CARD_WIDTH
    height = OUTPUT_CARD_HEIGHT

    return np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )


def rectify_card(
    image_rgb: np.ndarray,
    ordered_corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a perspective transformation to obtain a front-facing
    credit-card image.

    Corner order must be:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    source = np.asarray(
        ordered_corners,
        dtype=np.float32,
    )

    if source.shape != (4, 2):
        raise ValueError(
            "ordered_corners must have shape (4, 2)."
        )

    destination = destination_corners()

    homography_matrix = cv2.getPerspectiveTransform(
        source,
        destination,
    )

    if not np.all(
        np.isfinite(homography_matrix)
    ):
        raise RuntimeError(
            "Homography calculation produced invalid values."
        )

    rectified = cv2.warpPerspective(
        image_rgb,
        homography_matrix,
        (
            OUTPUT_CARD_WIDTH,
            OUTPUT_CARD_HEIGHT,
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rectified, homography_matrix