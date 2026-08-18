from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import FITLINE_DISTANCE_THRESHOLD


@dataclass
class FittedLine:
    """
    Infinite 2D line represented by:

        point + t * direction
    """

    point: np.ndarray
    direction: np.ndarray


@dataclass
class CornerResult:
    """
    Result returned to main.py.
    """

    ordered_corners: np.ndarray
    refined_lines: list[FittedLine]


def normalise_vector(
    vector: np.ndarray,
) -> np.ndarray:
    """
    Return a unit-length vector.
    """

    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        raise RuntimeError(
            "Cannot normalise a zero-length vector."
        )

    return vector / norm


def segment_to_fitted_line(
    segment: np.ndarray,
) -> FittedLine:
    """
    Convert a Hough segment into an infinite line.
    """

    x1, y1, x2, y2 = np.asarray(
        segment,
        dtype=np.float64,
    )

    point = np.array(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        ],
        dtype=np.float64,
    )

    direction = normalise_vector(
        np.array(
            [
                x2 - x1,
                y2 - y1,
            ],
            dtype=np.float64,
        )
    )

    return FittedLine(
        point=point,
        direction=direction,
    )


def point_to_line_distance(
    points: np.ndarray,
    line: FittedLine,
) -> np.ndarray:
    """
    Perpendicular distance from multiple points to an infinite line.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    direction = line.direction

    normal = np.array(
        [
            -direction[1],
            direction[0],
        ],
        dtype=np.float64,
    )

    offsets = points - line.point

    distances = np.abs(
        offsets @ normal
    )

    return distances


def collect_nearby_contour_points(
    contour: np.ndarray,
    initial_line: FittedLine,
    distance_threshold: float,
) -> np.ndarray:
    """
    Collect contour points lying near an initial Hough line.
    """

    contour_points = contour.reshape(
        -1,
        2,
    ).astype(np.float64)

    distances = point_to_line_distance(
        contour_points,
        initial_line,
    )

    selected_points = contour_points[
        distances <= distance_threshold
    ]

    return selected_points


def fit_line_to_points(
    points: np.ndarray,
) -> FittedLine:
    """
    Robustly fit an infinite line using OpenCV fitLine.
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    if len(points) < 2:
        raise RuntimeError(
            "At least two points are required to fit a line."
        )

    result = cv2.fitLine(
        points,
        cv2.DIST_HUBER,
        0,
        0.01,
        0.01,
    )

    vx, vy, x0, y0 = [
        float(value)
        for value in result.reshape(-1)
    ]

    direction = normalise_vector(
        np.array(
            [vx, vy],
            dtype=np.float64,
        )
    )

    point = np.array(
        [x0, y0],
        dtype=np.float64,
    )

    return FittedLine(
        point=point,
        direction=direction,
    )


def refine_selected_line(
    contour: np.ndarray,
    selected_segment: np.ndarray,
) -> FittedLine:
    """
    Refine one selected Hough segment using nearby contour points.
    """

    initial_line = segment_to_fitted_line(
        selected_segment
    )

    nearby_points = collect_nearby_contour_points(
        contour,
        initial_line,
        FITLINE_DISTANCE_THRESHOLD,
    )

    if len(nearby_points) < 8:
        # Relax the threshold when the first pass captures too few
        # contour pixels.
        nearby_points = collect_nearby_contour_points(
            contour,
            initial_line,
            FITLINE_DISTANCE_THRESHOLD * 2.0,
        )

    if len(nearby_points) < 2:
        return initial_line

    return fit_line_to_points(
        nearby_points
    )


def line_angle(
    line: FittedLine,
) -> float:
    """
    Return undirected line angle in degrees.
    """

    dx, dy = line.direction

    return float(
        np.degrees(
            np.arctan2(
                dy,
                dx,
            )
        )
        % 180.0
    )


def angular_difference(
    angle_a: float,
    angle_b: float,
) -> float:
    """
    Smallest difference between two undirected angles.
    """

    difference = abs(angle_a - angle_b)

    return min(
        difference,
        180.0 - difference,
    )


def divide_into_parallel_pairs(
    lines: list[FittedLine],
) -> tuple[
    list[FittedLine],
    list[FittedLine],
]:
    """
    Divide four lines into two approximately parallel pairs.

    Tests the three possible pairings and chooses the one with the
    lowest total within-pair angular difference.
    """

    if len(lines) != 4:
        raise ValueError(
            "Exactly four lines are required."
        )

    pairings = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]

    best_pairing = None
    best_score = float("inf")

    angles = [
        line_angle(line)
        for line in lines
    ]

    for first_pair, second_pair in pairings:
        score = (
            angular_difference(
                angles[first_pair[0]],
                angles[first_pair[1]],
            )
            +
            angular_difference(
                angles[second_pair[0]],
                angles[second_pair[1]],
            )
        )

        if score < best_score:
            best_score = score
            best_pairing = (
                first_pair,
                second_pair,
            )

    if best_pairing is None:
        raise RuntimeError(
            "Could not divide lines into parallel pairs."
        )

    first_indices, second_indices = best_pairing

    first_group = [
        lines[first_indices[0]],
        lines[first_indices[1]],
    ]

    second_group = [
        lines[second_indices[0]],
        lines[second_indices[1]],
    ]

    return first_group, second_group


def intersect_lines(
    line_a: FittedLine,
    line_b: FittedLine,
) -> np.ndarray:
    """
    Find the intersection of two infinite lines.
    """

    point_a = line_a.point
    point_b = line_b.point

    direction_a = line_a.direction
    direction_b = line_b.direction

    matrix = np.column_stack(
        (
            direction_a,
            -direction_b,
        )
    )

    determinant = np.linalg.det(matrix)

    if abs(determinant) < 1e-8:
        raise RuntimeError(
            "Two fitted lines are nearly parallel and cannot "
            "produce a stable intersection."
        )

    parameters = np.linalg.solve(
        matrix,
        point_b - point_a,
    )

    intersection = (
        point_a
        + parameters[0] * direction_a
    )

    return intersection.astype(
        np.float64
    )


def generate_four_intersections(
    first_pair: list[FittedLine],
    second_pair: list[FittedLine],
) -> np.ndarray:
    """
    Intersect every line in one orientation family with every line
    in the other family.
    """

    corners = []

    for first_line in first_pair:
        for second_line in second_pair:
            corner = intersect_lines(
                first_line,
                second_line,
            )

            corners.append(corner)

    return np.asarray(
        corners,
        dtype=np.float64,
    )


def order_corners(
    corners: np.ndarray,
) -> np.ndarray:
    """
    Order corners as:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    corners = np.asarray(
        corners,
        dtype=np.float64,
    )

    if corners.shape != (4, 2):
        raise ValueError(
            "Corners must have shape (4, 2)."
        )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float64,
    )

    coordinate_sums = (
        corners[:, 0]
        + corners[:, 1]
    )

    coordinate_differences = (
        corners[:, 0]
        - corners[:, 1]
    )

    ordered[0] = corners[
        np.argmin(coordinate_sums)
    ]

    ordered[2] = corners[
        np.argmax(coordinate_sums)
    ]

    ordered[1] = corners[
        np.argmax(coordinate_differences)
    ]

    ordered[3] = corners[
        np.argmin(coordinate_differences)
    ]

    return ordered


def validate_corners(
    corners: np.ndarray,
    image_shape: tuple[int, ...],
) -> None:
    """
    Perform basic geometric sanity checks.
    """

    if not np.all(
        np.isfinite(corners)
    ):
        raise RuntimeError(
            "Corner coordinates contain invalid numeric values."
        )

    polygon = corners.astype(
        np.float32
    )

    area = abs(
        cv2.contourArea(
            polygon.reshape(-1, 1, 2)
        )
    )

    image_height, image_width = image_shape[:2]

    image_area = float(
        image_height * image_width
    )

    if area < image_area * 0.005:
        raise RuntimeError(
            "Detected quadrilateral is implausibly small."
        )

    margin_x = image_width * 0.5
    margin_y = image_height * 0.5

    for x, y in corners:
        if not (
            -margin_x <= x <= image_width + margin_x
            and
            -margin_y <= y <= image_height + margin_y
        ):
            raise RuntimeError(
                "A detected geometric corner lies implausibly far "
                "outside the image."
            )


def compute_card_corners(
    contour: np.ndarray,
    selected_lines: list[np.ndarray],
    image_shape: tuple[int, ...],
) -> CornerResult:
    """
    Refine the four selected Hough lines and intersect them to obtain
    the card's geometric corners.
    """

    if len(selected_lines) != 4:
        raise RuntimeError(
            "Exactly four selected Hough lines are required."
        )

    refined_lines = [
        refine_selected_line(
            contour,
            segment,
        )
        for segment in selected_lines
    ]

    first_pair, second_pair = (
        divide_into_parallel_pairs(
            refined_lines
        )
    )

    corners = generate_four_intersections(
        first_pair,
        second_pair,
    )

    ordered_corners = order_corners(
        corners
    )

    validate_corners(
        ordered_corners,
        image_shape,
    )

    return CornerResult(
        ordered_corners=ordered_corners,
        refined_lines=refined_lines,
    )