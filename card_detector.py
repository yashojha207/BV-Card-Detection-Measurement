from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


@dataclass
class CardCandidate:
    box: np.ndarray
    confidence: float
    source: str


class CardDetector:
    """
    Generates possible card detections.

    IMPORTANT:
    This class does NOT blindly decide which detection is the card.
    Candidate validation happens later using RMBG segmentation.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
    ) -> None:

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        self.model = YOLO(
            str(self.model_path)
        )

        self.conf = conf

    # ========================================================
    # SINGLE INFERENCE
    # ========================================================

    def _get_predictions(
        self,
        image_rgb: np.ndarray,
        confidence: float,
        image_size: int,
    ) -> list[tuple[np.ndarray, float]]:

        result = self.model.predict(
            source=image_rgb,
            conf=confidence,
            imgsz=image_size,
            max_det=20,
            verbose=False,
        )[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            return []

        boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        scores = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        predictions = []

        image_height, image_width = (
            image_rgb.shape[:2]
        )

        for box, score in zip(
            boxes,
            scores,
        ):
            box = box.astype(
                np.float64
            )

            x1, y1, x2, y2 = box

            width = x2 - x1
            height = y2 - y1

            if (
                width <= 0
                or height <= 0
            ):
                continue

            area_ratio = (
                width * height
            ) / (
                image_width
                * image_height
            )

            # Only reject absurdly tiny noise.
            if area_ratio < 0.0003:
                continue

            predictions.append(
                (
                    box,
                    float(score),
                )
            )

        return predictions

    # ========================================================
    # FULL IMAGE CANDIDATES
    # ========================================================

    def get_full_image_candidates(
        self,
        image_rgb: np.ndarray,
    ) -> list[CardCandidate]:

        attempts = [
            (0.25, 640),
            (0.20, 960),
            (0.12, 1280),
        ]

        candidates = []

        for confidence, image_size in attempts:

            print(
                f"Trying full image: "
                f"imgsz={image_size}, "
                f"conf={confidence:.2f}"
            )

            predictions = self._get_predictions(
                image_rgb=image_rgb,
                confidence=confidence,
                image_size=image_size,
            )

            for box, score in predictions:
                candidates.append(
                    CardCandidate(
                        box=box,
                        confidence=score,
                        source=(
                            f"full_{image_size}"
                        ),
                    )
                )

        return candidates

    # ========================================================
    # TILE GENERATION
    # ========================================================

    def generate_tiles(
        self,
        image_rgb: np.ndarray,
        tile_fraction: float = 0.55,
        overlap: float = 0.30,
    ):

        image_height, image_width = (
            image_rgb.shape[:2]
        )

        tile_width = max(
            1,
            int(
                image_width
                * tile_fraction
            ),
        )

        tile_height = max(
            1,
            int(
                image_height
                * tile_fraction
            ),
        )

        step_x = max(
            1,
            int(
                tile_width
                * (1.0 - overlap)
            ),
        )

        step_y = max(
            1,
            int(
                tile_height
                * (1.0 - overlap)
            ),
        )

        max_x = max(
            0,
            image_width - tile_width,
        )

        max_y = max(
            0,
            image_height - tile_height,
        )

        x_positions = list(
            range(
                0,
                max_x + 1,
                step_x,
            )
        )

        y_positions = list(
            range(
                0,
                max_y + 1,
                step_y,
            )
        )

        if (
            not x_positions
            or x_positions[-1] != max_x
        ):
            x_positions.append(
                max_x
            )

        if (
            not y_positions
            or y_positions[-1] != max_y
        ):
            y_positions.append(
                max_y
            )

        tile_number = 0

        for y1 in y_positions:
            for x1 in x_positions:

                tile_number += 1

                x2 = min(
                    image_width,
                    x1 + tile_width,
                )

                y2 = min(
                    image_height,
                    y1 + tile_height,
                )

                tile = image_rgb[
                    y1:y2,
                    x1:x2,
                ].copy()

                if tile.size == 0:
                    continue

                yield (
                    tile_number,
                    tile,
                    x1,
                    y1,
                )

    # ========================================================
    # TILED CANDIDATES
    # ========================================================

    def get_tiled_candidates(
        self,
        image_rgb: np.ndarray,
    ) -> list[CardCandidate]:

        print(
            "Searching overlapping tiles..."
        )

        candidates = []

        for (
            tile_number,
            tile,
            offset_x,
            offset_y,
        ) in self.generate_tiles(
            image_rgb
        ):

            predictions = self._get_predictions(
                image_rgb=tile,
                confidence=0.08,
                image_size=960,
            )

            for tile_box, score in predictions:

                x1, y1, x2, y2 = (
                    tile_box
                )

                full_box = np.array(
                    [
                        x1 + offset_x,
                        y1 + offset_y,
                        x2 + offset_x,
                        y2 + offset_y,
                    ],
                    dtype=np.float64,
                )

                candidates.append(
                    CardCandidate(
                        box=full_box,
                        confidence=score,
                        source=(
                            f"tile_{tile_number}"
                        ),
                    )
                )

        return candidates

    # ========================================================
    # DUPLICATE REMOVAL
    # ========================================================

    @staticmethod
    def _iou(
        box_a: np.ndarray,
        box_b: np.ndarray,
    ) -> float:

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)

        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(
            0.0,
            ix2 - ix1,
        )

        ih = max(
            0.0,
            iy2 - iy1,
        )

        intersection = (
            iw * ih
        )

        area_a = (
            (ax2 - ax1)
            * (ay2 - ay1)
        )

        area_b = (
            (bx2 - bx1)
            * (by2 - by1)
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return float(
            intersection / union
        )

    def remove_duplicates(
        self,
        candidates: list[CardCandidate],
        iou_threshold: float = 0.60,
    ) -> list[CardCandidate]:

        candidates = sorted(
            candidates,
            key=lambda c: c.confidence,
            reverse=True,
        )

        kept = []

        for candidate in candidates:

            duplicate = False

            for existing in kept:

                if (
                    self._iou(
                        candidate.box,
                        existing.box,
                    )
                    >= iou_threshold
                ):
                    duplicate = True
                    break

            if not duplicate:
                kept.append(
                    candidate
                )

        return kept

    # ========================================================
    # PUBLIC CANDIDATE FUNCTION
    # ========================================================

    def get_candidates(
        self,
        image_rgb: np.ndarray,
    ) -> list[CardCandidate]:

        if (
            image_rgb is None
            or image_rgb.size == 0
        ):
            raise ValueError(
                "Input image is empty."
            )

        print(
            "Generating possible card detections..."
        )

        candidates = (
            self.get_full_image_candidates(
                image_rgb
            )
        )

        candidates.extend(
            self.get_tiled_candidates(
                image_rgb
            )
        )

        candidates = (
            self.remove_duplicates(
                candidates
            )
        )

        print(
            f"YOLO produced "
            f"{len(candidates)} unique "
            f"candidate(s)."
        )

        return candidates


# ============================================================
# CARD CROPPING
# ============================================================

def crop_card(
    image: np.ndarray,
    box: np.ndarray,
    padding: float = 0.10,
) -> tuple[
    np.ndarray,
    tuple[int, int],
]:

    x1, y1, x2, y2 = [
        float(value)
        for value in np.asarray(
            box
        ).reshape(-1)
    ]

    if (
        x2 <= x1
        or y2 <= y1
    ):
        raise RuntimeError(
            f"Invalid card box: "
            f"({x1}, {y1}, {x2}, {y2})"
        )

    width = (
        x2 - x1
    )

    height = (
        y2 - y1
    )

    pad_x = (
        width * padding
    )

    pad_y = (
        height * padding
    )

    image_height, image_width = (
        image.shape[:2]
    )

    crop_x1 = max(
        0,
        int(
            round(
                x1 - pad_x
            )
        ),
    )

    crop_y1 = max(
        0,
        int(
            round(
                y1 - pad_y
            )
        ),
    )

    crop_x2 = min(
        image_width,
        int(
            round(
                x2 + pad_x
            )
        ),
    )

    crop_y2 = min(
        image_height,
        int(
            round(
                y2 + pad_y
            )
        ),
    )

    crop = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ].copy()

    if crop.size == 0:
        raise RuntimeError(
            "Card crop is empty."
        )

    return (
        crop,
        (
            crop_x1,
            crop_y1,
        ),
    )