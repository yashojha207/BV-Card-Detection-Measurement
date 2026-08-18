from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from config import (
    IMAGE_DIR,
    OUTPUT_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
)


# ============================================================
# DIRECTORY FUNCTIONS
# ============================================================

def create_required_directories() -> None:
    """
    Create project directories if they do not already exist.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def find_input_images() -> List[Path]:
    """
    Returns every supported image inside IMAGE_DIR.
    """

    if not IMAGE_DIR.exists():
        return []

    image_paths = []

    for file in IMAGE_DIR.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        image_paths.append(file)

    image_paths.sort(key=lambda p: p.name.lower())

    return image_paths


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(
    image_path: Path,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Returns

        Pillow RGB image

    and

        NumPy RGB image
    """

    pillow_image = Image.open(image_path).convert("RGB")

    image_rgb = np.array(
        pillow_image,
        dtype=np.uint8,
    )

    return pillow_image, image_rgb


# ============================================================
# SAVE IMAGES
# ============================================================

def save_grayscale_image(
    output_path: Path,
    image: np.ndarray,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        image,
    )


def save_rgb_image(
    output_path: Path,
    image_rgb: np.ndarray,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        str(output_path),
        image_bgr,
    )


# ============================================================
# SAVE TEXT
# ============================================================

def save_text_file(
    output_path: Path,
    text: str,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(text)


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def distance(
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """
    Euclidean distance.
    """

    return float(
        np.linalg.norm(
            p1 - p2
        )
    )


def midpoint(
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """
    Midpoint between two points.
    """

    return (p1 + p2) / 2.0


def normalize_vector(
    vector: np.ndarray,
) -> np.ndarray:
    """
    Returns a unit vector.
    """

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


# ============================================================
# CONTOUR HELPERS
# ============================================================

def contour_area(
    contour,
) -> float:

    return float(
        cv2.contourArea(contour)
    )


def contour_perimeter(
    contour,
) -> float:

    return float(
        cv2.arcLength(
            contour,
            True,
        )
    )


# ============================================================
# LINE HELPERS
# ============================================================

def line_length(
    line,
) -> float:
    """
    Computes the length of a line returned by HoughLinesP.

    Format:

        [[x1,y1,x2,y2]]
    """

    x1, y1, x2, y2 = line[0]

    return float(
        np.hypot(
            x2 - x1,
            y2 - y1,
        )
    )


def line_angle(
    line,
) -> float:
    """
    Returns line angle in degrees.

    Range:

        -180 to 180
    """

    x1, y1, x2, y2 = line[0]

    angle = np.degrees(
        np.arctan2(
            y2 - y1,
            x2 - x1,
        )
    )

    return float(angle)


# ============================================================
# DEBUG
# ============================================================

def print_header(
    text: str,
) -> None:

    print()

    print("=" * 70)

    print(text)

    print("=" * 70)


def print_subheader(
    text: str,
) -> None:

    print()

    print("-" * 40)

    print(text)

    print("-" * 40)