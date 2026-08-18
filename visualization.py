from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import (
    RAW_LINE_THICKNESS,
    REFINED_LINE_THICKNESS,
    CORNER_RADIUS,
    CORNER_TEXT_OFFSET,
)

from geometry import FittedLine


def save_grayscale_image(
    output_path: Path,
    image: np.ndarray,
) -> None:
    """
    Save a grayscale image.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise RuntimeError(
            f"Could not save image: {output_path}"
        )


def save_rgb_image(
    output_path: Path,
    image_rgb: np.ndarray,
) -> None:
    """
    Save an RGB image through OpenCV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    success = cv2.imwrite(
        str(output_path),
        image_bgr,
    )

    if not success:
        raise RuntimeError(
            f"Could not save image: {output_path}"
        )


def save_text_file(
    output_path: Path,
    text: str,
) -> None:
    """
    Save plain text.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        text,
        encoding="utf-8",
    )


def create_contour_visualization(
    image_rgb: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:
    """
    Draw the largest segmented contour.
    """

    visualization = image_rgb.copy()

    cv2.drawContours(
        visualization,
        [contour],
        contourIdx=-1,
        color=(255, 0, 0),
        thickness=3,
    )

    return visualization


def draw_segment(
    image: np.ndarray,
    line: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """
    Draw one finite line segment.
    """

    x1, y1, x2, y2 = [
        int(round(value))
        for value in np.asarray(line).reshape(-1)
    ]

    cv2.line(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def create_hough_visualization(
    image_rgb: np.ndarray,
    raw_lines: list[np.ndarray],
    selected_lines: list[np.ndarray],
) -> np.ndarray:
    """
    Draw raw Hough lines and the four selected boundaries.
    """

    visualization = image_rgb.copy()

    for line in raw_lines:
        draw_segment(
            visualization,
            line,
            color=(255, 255, 0),
            thickness=RAW_LINE_THICKNESS,
        )

    selected_colours = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
    ]

    for index, line in enumerate(
        selected_lines
    ):
        draw_segment(
            visualization,
            line,
            color=selected_colours[
                index % len(selected_colours)
            ],
            thickness=3,
        )

    return visualization


def clipped_line_endpoints(
    line: FittedLine,
    image_shape: tuple[int, ...],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Generate long endpoints and clip them to the image boundary.
    """

    image_height, image_width = image_shape[:2]

    point = line.point
    direction = line.direction

    scale = float(
        max(
            image_width,
            image_height,
        )
        * 3
    )

    first = point - scale * direction
    second = point + scale * direction

    first_tuple = (
        int(round(first[0])),
        int(round(first[1])),
    )

    second_tuple = (
        int(round(second[0])),
        int(round(second[1])),
    )

    rectangle = (
        0,
        0,
        image_width,
        image_height,
    )

    success, clipped_first, clipped_second = (
        cv2.clipLine(
            rectangle,
            first_tuple,
            second_tuple,
        )
    )

    if not success:
        return first_tuple, second_tuple

    return clipped_first, clipped_second


def draw_infinite_line(
    image: np.ndarray,
    line: FittedLine,
    color: tuple[int, int, int],
) -> None:
    """
    Draw a fitted infinite line clipped to the image.
    """

    first, second = clipped_line_endpoints(
        line,
        image.shape,
    )

    cv2.line(
        image,
        first,
        second,
        color,
        REFINED_LINE_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def create_corner_visualization(
    image_rgb: np.ndarray,
    corners: np.ndarray,
    refined_lines: list[FittedLine],
) -> np.ndarray:
    """
    Draw fitted lines, geometric corners, labels, and the final
    quadrilateral.
    """

    visualization = image_rgb.copy()

    line_colours = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
    ]

    for index, line in enumerate(
        refined_lines
    ):
        draw_infinite_line(
            visualization,
            line,
            line_colours[
                index % len(line_colours)
            ],
        )

    corner_names = [
        "TL",
        "TR",
        "BR",
        "BL",
    ]

    corner_colours = [
        (255, 255, 255),
        (255, 255, 255),
        (255, 255, 255),
        (255, 255, 255),
    ]

    integer_corners = np.round(
        corners
    ).astype(np.int32)

    cv2.polylines(
        visualization,
        [
            integer_corners.reshape(
                -1,
                1,
                2,
            )
        ],
        isClosed=True,
        color=(255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    for index, corner in enumerate(
        integer_corners
    ):
        x, y = int(corner[0]), int(corner[1])

        cv2.circle(
            visualization,
            (x, y),
            CORNER_RADIUS,
            corner_colours[index],
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(
            visualization,
            (x, y),
            CORNER_RADIUS + 2,
            (0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            visualization,
            corner_names[index],
            (
                x + CORNER_TEXT_OFFSET,
                y - CORNER_TEXT_OFFSET,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    return visualization