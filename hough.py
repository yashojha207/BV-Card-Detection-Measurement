from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import (
    CANNY_LOW_THRESHOLD,
    CANNY_HIGH_THRESHOLD,
    HOUGH_RHO,
    HOUGH_THETA,
    HOUGH_THRESHOLD,
    HOUGH_MIN_LINE_LENGTH,
    HOUGH_MAX_LINE_GAP,
    MIN_LINE_LENGTH,
)


@dataclass
class HoughResult:
    """
    Stores all useful Hough-transform outputs.
    """

    edge_image: np.ndarray
    raw_lines: list[np.ndarray]
    selected_lines: list[np.ndarray]


def create_contour_edge_image(
    clean_mask: np.ndarray,
) -> np.ndarray:
    """
    Create a thin edge image from the cleaned foreground mask.
    """

    if clean_mask.ndim != 2:
        raise ValueError(
            "clean_mask must be a single-channel image."
        )

    edge_image = cv2.Canny(
        clean_mask,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD,
    )

    return edge_image


def run_probabilistic_hough(
    edge_image: np.ndarray,
) -> list[np.ndarray]:
    """
    Detect line segments using the probabilistic Hough transform.

    Each returned line has the format:

        [x1, y1, x2, y2]
    """

    detected = cv2.HoughLinesP(
        edge_image,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if detected is None:
        raise RuntimeError(
            "The Hough transform did not detect any lines."
        )

    lines: list[np.ndarray] = []

    for item in detected:
        line = np.asarray(
            item[0],
            dtype=np.float64,
        )

        x1, y1, x2, y2 = line

        length = float(
            np.hypot(
                x2 - x1,
                y2 - y1,
            )
        )

        if length >= MIN_LINE_LENGTH:
            lines.append(line)

    if len(lines) < 4:
        raise RuntimeError(
            "Fewer than four usable Hough segments were detected."
        )

    return lines


def canonical_line_angle(
    line: np.ndarray,
) -> float:
    """
    Return the undirected line angle in degrees.

    The result lies within:

        [0, 180)

    A line pointing in the opposite direction has the same angle.
    """

    x1, y1, x2, y2 = line

    angle = np.degrees(
        np.arctan2(
            y2 - y1,
            x2 - x1,
        )
    )

    angle = angle % 180.0

    return float(angle)


def angular_difference(
    angle_a: float,
    angle_b: float,
) -> float:
    """
    Smallest difference between two undirected line angles.
    """

    difference = abs(angle_a - angle_b)

    return min(
        difference,
        180.0 - difference,
    )


def line_length(
    line: np.ndarray,
) -> float:
    """
    Return the Euclidean length of a line segment.
    """

    x1, y1, x2, y2 = line

    return float(
        np.hypot(
            x2 - x1,
            y2 - y1,
        )
    )


def line_midpoint(
    line: np.ndarray,
) -> np.ndarray:
    """
    Return the midpoint of a segment.
    """

    x1, y1, x2, y2 = line

    return np.array(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        ],
        dtype=np.float64,
    )


def weighted_orientation_histogram(
    lines: list[np.ndarray],
    bin_count: int = 180,
) -> np.ndarray:
    """
    Build an orientation histogram weighted by line length.
    """

    histogram = np.zeros(
        bin_count,
        dtype=np.float64,
    )

    for line in lines:
        angle = canonical_line_angle(line)
        length = line_length(line)

        index = int(round(angle)) % bin_count

        histogram[index] += length

    # Circular smoothing.
    padded = np.concatenate(
        [
            histogram[-4:],
            histogram,
            histogram[:4],
        ]
    )

    kernel = np.ones(
        9,
        dtype=np.float64,
    ) / 9.0

    smoothed = np.convolve(
        padded,
        kernel,
        mode="same",
    )

    return smoothed[4:-4]


def find_two_dominant_orientations(
    lines: list[np.ndarray],
) -> tuple[float, float]:
    """
    Find two approximately perpendicular dominant directions.
    """

    histogram = weighted_orientation_histogram(
        lines
    )

    first_angle = float(
        np.argmax(histogram)
    )

    candidate_angles = []

    for angle in range(180):
        difference = angular_difference(
            float(angle),
            first_angle,
        )

        if 55.0 <= difference <= 125.0:
            candidate_angles.append(angle)

    if not candidate_angles:
        raise RuntimeError(
            "Could not find a second dominant line orientation."
        )

    second_angle = float(
        max(
            candidate_angles,
            key=lambda angle: histogram[angle],
        )
    )

    return first_angle, second_angle


def assign_lines_to_orientation_groups(
    lines: list[np.ndarray],
    angle_a: float,
    angle_b: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Assign each segment to whichever dominant direction is closer.
    """

    group_a: list[np.ndarray] = []
    group_b: list[np.ndarray] = []

    for line in lines:
        angle = canonical_line_angle(line)

        difference_a = angular_difference(
            angle,
            angle_a,
        )

        difference_b = angular_difference(
            angle,
            angle_b,
        )

        if difference_a <= difference_b:
            group_a.append(line)
        else:
            group_b.append(line)

    if len(group_a) < 2 or len(group_b) < 2:
        raise RuntimeError(
            "Hough lines could not be divided into two usable "
            "orientation groups."
        )

    return group_a, group_b


def contour_centroid(
    contour: np.ndarray,
) -> np.ndarray:
    """
    Compute the contour centroid.
    """

    moments = cv2.moments(contour)

    if abs(moments["m00"]) < 1e-9:
        points = contour.reshape(-1, 2).astype(
            np.float64
        )

        return np.mean(
            points,
            axis=0,
        )

    centre_x = moments["m10"] / moments["m00"]
    centre_y = moments["m01"] / moments["m00"]

    return np.array(
        [centre_x, centre_y],
        dtype=np.float64,
    )


def orientation_normal(
    angle_degrees: float,
) -> np.ndarray:
    """
    Return a unit normal vector for a line orientation.
    """

    radians = np.radians(angle_degrees)

    direction = np.array(
        [
            np.cos(radians),
            np.sin(radians),
        ],
        dtype=np.float64,
    )

    normal = np.array(
        [
            -direction[1],
            direction[0],
        ],
        dtype=np.float64,
    )

    norm = np.linalg.norm(normal)

    if norm < 1e-9:
        raise RuntimeError(
            "Could not construct line normal."
        )

    return normal / norm


def signed_segment_offset(
    line: np.ndarray,
    centroid: np.ndarray,
    normal: np.ndarray,
) -> float:
    """
    Signed midpoint displacement from the contour centroid.
    """

    midpoint = line_midpoint(line)

    return float(
        np.dot(
            midpoint - centroid,
            normal,
        )
    )


def choose_opposite_boundary_segments(
    lines: list[np.ndarray],
    dominant_angle: float,
    centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Choose one segment from each opposite side of the card.

    For one orientation family, the card should have two parallel
    boundaries located on opposite sides of the centroid.
    """

    normal = orientation_normal(
        dominant_angle
    )

    positive_candidates: list[
        tuple[float, float, np.ndarray]
    ] = []

    negative_candidates: list[
        tuple[float, float, np.ndarray]
    ] = []

    for line in lines:
        offset = signed_segment_offset(
            line,
            centroid,
            normal,
        )

        length = line_length(line)

        record = (
            abs(offset),
            length,
            line,
        )

        if offset >= 0:
            positive_candidates.append(record)
        else:
            negative_candidates.append(record)

    if not positive_candidates or not negative_candidates:
        # Fallback: select segments with the minimum and maximum
        # signed offsets.
        sorted_lines = sorted(
            lines,
            key=lambda line: signed_segment_offset(
                line,
                centroid,
                normal,
            ),
        )

        return (
            sorted_lines[0],
            sorted_lines[-1],
        )

    positive_line = max(
        positive_candidates,
        key=lambda item: (
            item[0],
            item[1],
        ),
    )[2]

    negative_line = max(
        negative_candidates,
        key=lambda item: (
            item[0],
            item[1],
        ),
    )[2]

    return negative_line, positive_line


def detect_card_edge_lines(
    clean_mask: np.ndarray,
    contour: np.ndarray,
) -> HoughResult:
    """
    Complete Hough-line stage.

    Steps:
        1. Create edge image.
        2. Detect probabilistic Hough segments.
        3. Estimate two dominant directions.
        4. Divide lines into two orientation families.
        5. Select two opposite boundaries from each family.

    Returns four selected lines, but their final top/right/bottom/left
    ordering is performed later in geometry.py.
    """

    edge_image = create_contour_edge_image(
        clean_mask
    )

    raw_lines = run_probabilistic_hough(
        edge_image
    )

    angle_a, angle_b = (
        find_two_dominant_orientations(
            raw_lines
        )
    )

    group_a, group_b = (
        assign_lines_to_orientation_groups(
            raw_lines,
            angle_a,
            angle_b,
        )
    )

    centroid = contour_centroid(
        contour
    )

    a_first, a_second = (
        choose_opposite_boundary_segments(
            group_a,
            angle_a,
            centroid,
        )
    )

    b_first, b_second = (
        choose_opposite_boundary_segments(
            group_b,
            angle_b,
            centroid,
        )
    )

    selected_lines = [
        np.asarray(a_first, dtype=np.float64),
        np.asarray(a_second, dtype=np.float64),
        np.asarray(b_first, dtype=np.float64),
        np.asarray(b_second, dtype=np.float64),
    ]

    return HoughResult(
        edge_image=edge_image,
        raw_lines=raw_lines,
        selected_lines=selected_lines,
    )