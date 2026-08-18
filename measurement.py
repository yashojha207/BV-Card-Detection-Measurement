from __future__ import annotations

from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import (
    OUTPUT_CARD_HEIGHT,
    OUTPUT_CARD_WIDTH,
)


# Standard ISO/IEC 7810 ID-1 card dimensions.
CARD_WIDTH_CM = 8.56
CARD_HEIGHT_CM = 5.398


def get_metric_scales() -> tuple[float, float]:
    """
    Return the horizontal and vertical pixel scales in the
    homography destination coordinate system.

    The homography maps the detected credit card to:
        OUTPUT_CARD_WIDTH × OUTPUT_CARD_HEIGHT pixels

    That region physically represents:
        8.56 × 5.398 cm
    """
    pixels_per_cm_x = (
        OUTPUT_CARD_WIDTH - 1
    ) / CARD_WIDTH_CM

    pixels_per_cm_y = (
        OUTPUT_CARD_HEIGHT - 1
    ) / CARD_HEIGHT_CM

    return (
        float(pixels_per_cm_x),
        float(pixels_per_cm_y),
    )


def select_points(
    image_rgb: np.ndarray,
    number_of_points: int,
    instruction: str,
) -> np.ndarray:
    """
    Display the original photograph and allow the user to click
    a fixed number of points.

    Press Backspace/Delete in the Matplotlib window to undo the
    most recent point.
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("The measurement image is empty.")

    figure, axis = plt.subplots()

    axis.imshow(image_rgb)
    axis.set_title(
        f"{instruction}\n"
        f"Click {number_of_points} point(s)."
    )
    axis.axis("off")

    points = plt.ginput(
        number_of_points,
        timeout=0,
        show_clicks=True,
    )

    plt.close(figure)

    if len(points) != number_of_points:
        raise RuntimeError(
            f"Expected {number_of_points} clicked points, "
            f"but received {len(points)}."
        )

    return np.asarray(
        points,
        dtype=np.float32,
    )


def transform_points(
    original_points: np.ndarray,
    homography_matrix: np.ndarray,
) -> np.ndarray:
    """
    Transform points from the original photograph into the
    credit-card-calibrated homography coordinate system.

    Points outside the card can still be transformed as long as
    they lie on the same physical plane as the credit card.
    """
    points = np.asarray(
        original_points,
        dtype=np.float32,
    )

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "original_points must have shape (N, 2)."
        )

    matrix = np.asarray(
        homography_matrix,
        dtype=np.float64,
    )

    if matrix.shape != (3, 3):
        raise ValueError(
            "homography_matrix must have shape (3, 3)."
        )

    transformed = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2),
        matrix,
    )

    return transformed.reshape(-1, 2)


def metric_distance(
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> float:
    """
    Calculate the real distance in centimetres between two points
    in the homography coordinate system.
    """
    pixels_per_cm_x, pixels_per_cm_y = (
        get_metric_scales()
    )

    delta_x_cm = (
        float(point_b[0] - point_a[0])
        / pixels_per_cm_x
    )

    delta_y_cm = (
        float(point_b[1] - point_a[1])
        / pixels_per_cm_y
    )

    return float(
        np.hypot(
            delta_x_cm,
            delta_y_cm,
        )
    )


def measure_line_segment(
    image_rgb: np.ndarray,
    homography_matrix: np.ndarray,
) -> dict[str, float]:
    """
    Measure a line segment from two manually selected endpoints.
    """
    original_points = select_points(
        image_rgb=image_rgb,
        number_of_points=2,
        instruction=(
            "Click the two endpoints of the line segment"
        ),
    )

    transformed_points = transform_points(
        original_points=original_points,
        homography_matrix=homography_matrix,
    )

    length_cm = metric_distance(
        transformed_points[0],
        transformed_points[1],
    )

    return {
        "length_cm": length_cm,
    }


def measure_rectangle(
    image_rgb: np.ndarray,
    homography_matrix: np.ndarray,
) -> dict[str, float]:
    """
    Measure a rectangular object.

    Click points in this order:
        top-left
        top-right
        bottom-right
        bottom-left
    """
    original_points = select_points(
        image_rgb=image_rgb,
        number_of_points=4,
        instruction=(
            "Click rectangle corners in order: "
            "top-left, top-right, bottom-right, bottom-left"
        ),
    )

    transformed = transform_points(
        original_points=original_points,
        homography_matrix=homography_matrix,
    )

    top_left = transformed[0]
    top_right = transformed[1]
    bottom_right = transformed[2]
    bottom_left = transformed[3]

    top_width_cm = metric_distance(
        top_left,
        top_right,
    )

    bottom_width_cm = metric_distance(
        bottom_left,
        bottom_right,
    )

    left_height_cm = metric_distance(
        top_left,
        bottom_left,
    )

    right_height_cm = metric_distance(
        top_right,
        bottom_right,
    )

    width_cm = (
        top_width_cm + bottom_width_cm
    ) / 2.0

    height_cm = (
        left_height_cm + right_height_cm
    ) / 2.0

    area_cm2 = width_cm * height_cm

    return {
        "width_cm": float(width_cm),
        "height_cm": float(height_cm),
        "area_cm2": float(area_cm2),
    }


def measure_circle(
    image_rgb: np.ndarray,
    homography_matrix: np.ndarray,
) -> dict[str, float]:
    """
    Measure a circular object by clicking two opposite points
    across its diameter.
    """
    original_points = select_points(
        image_rgb=image_rgb,
        number_of_points=2,
        instruction=(
            "Click two opposite points across the circle's diameter"
        ),
    )

    transformed = transform_points(
        original_points=original_points,
        homography_matrix=homography_matrix,
    )

    diameter_cm = metric_distance(
        transformed[0],
        transformed[1],
    )

    radius_cm = diameter_cm / 2.0
    circumference_cm = np.pi * diameter_cm
    area_cm2 = np.pi * radius_cm**2

    return {
        "diameter_cm": float(diameter_cm),
        "radius_cm": float(radius_cm),
        "circumference_cm": float(
            circumference_cm
        ),
        "area_cm2": float(area_cm2),
    }


def format_measurement_result(
    measurement_type: str,
    result: dict[str, float],
) -> str:
    """
    Convert the measurement result into readable text.
    """
    lines = [
        f"Measurement type: {measurement_type}",
        "",
    ]

    for name, value in result.items():
        readable_name = name.replace("_", " ")

        lines.append(
            f"{readable_name}: {value:.3f}"
        )

    return "\n".join(lines)


def run_measurement(
    image_rgb: np.ndarray,
    homography_matrix: np.ndarray,
) -> dict[str, Any] | None:
    """
    Show the measurement menu and run one manual measurement.

    The target object and reference card must be on the same plane.
    """
    print("\n" + "=" * 70)
    print("OBJECT MEASUREMENT")
    print("=" * 70)
    print("1 - Rectangular object")
    print("2 - Circular object")
    print("3 - Line segment")
    print("0 - Skip measurement")

    choice = input(
        "\nChoose measurement type: "
    ).strip()

    if choice == "0":
        print("Measurement skipped.")
        return None

    if choice == "1":
        measurement_type = "rectangle"

        result = measure_rectangle(
            image_rgb=image_rgb,
            homography_matrix=homography_matrix,
        )

    elif choice == "2":
        measurement_type = "circle"

        result = measure_circle(
            image_rgb=image_rgb,
            homography_matrix=homography_matrix,
        )

    elif choice == "3":
        measurement_type = "line_segment"

        result = measure_line_segment(
            image_rgb=image_rgb,
            homography_matrix=homography_matrix,
        )

    else:
        raise ValueError(
            "Invalid measurement choice. "
            "Enter 0, 1, 2, or 3."
        )

    result_text = format_measurement_result(
        measurement_type=measurement_type,
        result=result,
    )

    print("\n" + result_text)

    return {
        "type": measurement_type,
        "values": result,
        "text": result_text,
    }