from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from segmentation import segment_foreground


CARD_WIDTH_CM = 8.56
CARD_HEIGHT_CM = 5.398


@dataclass
class TargetDetection:
    box_xyxy: np.ndarray
    confidence: float
    label: str


class GenericObjectMeasurer:
    """
    Generic open-vocabulary target detection and measurement.

    Pipeline:
        Grounding DINO
        → target crop
        → RMBG segmentation
        → target contour
        → card-based metric homography
        → rectangle / circle / line measurement
    """

    def __init__(
        self,
        confidence_threshold: float = 0.15,
    ) -> None:

        print("\nLoading Grounding DINO...")

        self.model_id = (
            "IDEA-Research/grounding-dino-tiny"
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Grounding DINO device: {self.device}"
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                self.model_id
            )
        )

        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(
                self.model_id
            )
            .to(self.device)
        )

        self.model.eval()

        self.confidence_threshold = (
            confidence_threshold
        )

        print(
            "Grounding DINO loaded successfully."
        )

    # ========================================================
    # TARGET DETECTION
    # ========================================================

    def detect_target(
        self,
        image_rgb: np.ndarray,
        prompt: str,
    ) -> TargetDetection:

        if (
            image_rgb is None
            or image_rgb.size == 0
        ):
            raise ValueError(
                "Target detection image is empty."
            )

        prompt = prompt.strip().lower()

        if not prompt:
            raise ValueError(
                "Target prompt cannot be empty."
            )

        aliases = {
            "paper": [
                "paper",
                "sheet of paper",
                "document",
                "white sheet of paper",
            ],
            "sheet of paper": [
                "sheet of paper",
                "paper",
                "document",
                "white sheet of paper",
            ],
            "coin": [
                "coin",
                "round coin",
            ],
            "shoe": [
                "shoe",
                "sneaker",
                "footwear",
            ],
            "book": [
                "book",
                "notebook",
            ],
            "phone": [
                "phone",
                "mobile phone",
                "smartphone",
            ],
            "credit card": [
            "credit card",
            "bank card",
            "payment card",
            "plastic card",
            "ID card",
        ],
        }

        labels = aliases.get(
            prompt,
            [prompt],
        )

        print(
            "Grounding DINO prompts:"
        )

        for label in labels:
            print(
                f"  - {label}"
            )

        image_pillow = Image.fromarray(
            image_rgb
        )

        text_labels = [
            labels
        ]

        inputs = self.processor(
            images=image_pillow,
            text=text_labels,
            return_tensors="pt",
        )

        inputs = {
            key: (
                value.to(self.device)
                if hasattr(value, "to")
                else value
            )
            for key, value in inputs.items()
        }

        print(
            "Running Grounding DINO detection..."
        )

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        results = (
            self.processor
            .post_process_grounded_object_detection(
                outputs,
                threshold=self.confidence_threshold,
                text_threshold=0.15,
                target_sizes=[
                    (
                        image_pillow.height,
                        image_pillow.width,
                    )
                ],
            )
        )

        if not results:
            raise RuntimeError(
                f"No Grounding DINO result "
                f"for '{prompt}'."
            )

        result = results[0]

        boxes = result.get(
            "boxes"
        )

        scores = result.get(
            "scores"
        )

        detected_labels = result.get(
            "text_labels",
            result.get(
                "labels",
                [],
            ),
        )

        if (
            boxes is None
            or scores is None
            or len(boxes) == 0
        ):
            raise RuntimeError(
                f"Could not detect target object: "
                f"{prompt}"
            )

        scores_numpy = (
            scores
            .detach()
            .cpu()
            .numpy()
        )

        best_index = int(
            np.argmax(
                scores_numpy
            )
        )

        best_score = float(
            scores_numpy[
                best_index
            ]
        )

        best_box = (
            boxes[
                best_index
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        if (
            len(detected_labels)
            > best_index
        ):
            best_label = str(
                detected_labels[
                    best_index
                ]
            )
        else:
            best_label = prompt

        x1, y1, x2, y2 = (
            best_box
        )

        box_width = (
            x2 - x1
        )

        box_height = (
            y2 - y1
        )

        image_height, image_width = (
            image_rgb.shape[:2]
        )

        area_ratio = (
            box_width
            * box_height
        ) / (
            image_width
            * image_height
        )

        if area_ratio < 0.001:
            raise RuntimeError(
                "Detected target bounding box "
                "is implausibly small."
            )

        print()
        print(
            f"Target detected as: "
            f"{best_label}"
        )

        print(
            f"Confidence: "
            f"{best_score:.3f}"
        )

        print(
            f"Box: "
            f"{best_box}"
        )

        return TargetDetection(
            box_xyxy=best_box,
            confidence=best_score,
            label=best_label,
        )

    # ========================================================
    # TARGET CROP
    # ========================================================

    def crop_target(
        self,
        image_rgb: np.ndarray,
        box_xyxy: np.ndarray,
        padding: float = 0.05,
    ) -> tuple[
        np.ndarray,
        tuple[int, int],
    ]:

        x1, y1, x2, y2 = [
            float(value)
            for value in np.asarray(
                box_xyxy
            ).reshape(-1)
        ]

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
            image_rgb.shape[:2]
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

        crop = image_rgb[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ].copy()

        if crop.size == 0:
            raise RuntimeError(
                "Target crop is empty."
            )

        return (
            crop,
            (
                crop_x1,
                crop_y1,
            ),
        )

    # ========================================================
    # TARGET SEGMENTATION
    # ========================================================

    def segment_target(
        self,
        crop_rgb: np.ndarray,
        segmenter,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        crop_pillow = (
            Image.fromarray(
                crop_rgb
            )
        )

        raw_mask = (
            segment_foreground(
                segmenter=segmenter,
                image_pillow=crop_pillow,
            )
        )

        _, binary = (
            cv2.threshold(
                raw_mask,
                128,
                255,
                cv2.THRESH_BINARY,
            )
        )

        kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (5, 5),
            )
        )

        binary = (
            cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                kernel,
                iterations=1,
            )
        )

        binary = (
            cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=2,
            )
        )

        contours, _ = (
            cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
        )

        if not contours:
            raise RuntimeError(
                "RMBG could not isolate "
                "the target object."
            )

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        contour_area = float(
            cv2.contourArea(
                contour
            )
        )

        if contour_area <= 0:
            raise RuntimeError(
                "Target contour has zero area."
            )

        clean_mask = (
            np.zeros_like(
                binary
            )
        )

        cv2.drawContours(
            clean_mask,
            [contour],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
        )

        return (
            clean_mask,
            contour,
        )

    # ========================================================
    # CONTOUR COORDINATES
    # ========================================================

    def contour_to_original_points(
        self,
        contour: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> np.ndarray:

        points = (
            contour
            .reshape(
                -1,
                2,
            )
            .astype(
                np.float32
            )
        )

        points[
            :,
            0,
        ] += offset_x

        points[
            :,
            1,
        ] += offset_y

        return points

    # ========================================================
    # METRIC HOMOGRAPHY
    # ========================================================

    def create_metric_homography(
        self,
        card_corners: np.ndarray,
    ) -> np.ndarray:
        """
        Map original image coordinates directly into centimetres.

        Card corners must be ordered:
            TL
            TR
            BR
            BL
        """

        source = np.asarray(
            card_corners,
            dtype=np.float32,
        )

        if source.shape != (
            4,
            2,
        ):
            raise ValueError(
                "card_corners must "
                "have shape (4, 2)."
            )

        destination_cm = (
            np.array(
                [
                    [0.0, 0.0],
                    [
                        CARD_WIDTH_CM,
                        0.0,
                    ],
                    [
                        CARD_WIDTH_CM,
                        CARD_HEIGHT_CM,
                    ],
                    [
                        0.0,
                        CARD_HEIGHT_CM,
                    ],
                ],
                dtype=np.float32,
            )
        )

        metric_homography = (
            cv2.getPerspectiveTransform(
                source,
                destination_cm,
            )
        )

        if not np.all(
            np.isfinite(
                metric_homography
            )
        ):
            raise RuntimeError(
                "Metric homography "
                "contains invalid values."
            )

        return metric_homography

    def transform_points_to_cm(
        self,
        points: np.ndarray,
        metric_homography: np.ndarray,
    ) -> np.ndarray:

        points = np.asarray(
            points,
            dtype=np.float32,
        )

        transformed = (
            cv2.perspectiveTransform(
                points.reshape(
                    -1,
                    1,
                    2,
                ),
                metric_homography,
            )
        )

        points_cm = (
            transformed.reshape(
                -1,
                2,
            )
        )

        if not np.all(
            np.isfinite(
                points_cm
            )
        ):
            raise RuntimeError(
                "Metric transformation "
                "produced invalid coordinates."
            )

        return points_cm

    # ========================================================
    # RECTANGLE MEASUREMENT
    # ========================================================

    def measure_rectangle(
        self,
        points_cm: np.ndarray,
    ) -> dict[str, float]:

        rectangle = (
            cv2.minAreaRect(
                points_cm.astype(
                    np.float32
                )
            )
        )

        width_cm, height_cm = (
            rectangle[1]
        )

        long_side = max(
            width_cm,
            height_cm,
        )

        short_side = min(
            width_cm,
            height_cm,
        )

        return {
            "length_cm": float(
                long_side
            ),
            "width_cm": float(
                short_side
            ),
            "area_cm2": float(
                long_side
                * short_side
            ),
        }

    # ========================================================
    # CIRCLE MEASUREMENT
    # ========================================================

    def measure_circle(
        self,
        points_cm: np.ndarray,
    ) -> dict[str, float]:

        _, radius = (
            cv2.minEnclosingCircle(
                points_cm.astype(
                    np.float32
                )
            )
        )

        diameter = (
            2.0 * radius
        )

        return {
            "diameter_cm": float(
                diameter
            ),
            "radius_cm": float(
                radius
            ),
            "circumference_cm": float(
                np.pi
                * diameter
            ),
            "area_cm2": float(
                np.pi
                * radius**2
            ),
        }

    # ========================================================
    # LINE / ELONGATED OBJECT MEASUREMENT
    # ========================================================

    def measure_line(
        self,
        points_cm: np.ndarray,
    ) -> dict[str, float]:

        points = np.asarray(
            points_cm,
            dtype=np.float64,
        )

        centre = np.mean(
            points,
            axis=0,
        )

        centred = (
            points - centre
        )

        covariance = (
            np.cov(
                centred,
                rowvar=False,
            )
        )

        eigenvalues, eigenvectors = (
            np.linalg.eigh(
                covariance
            )
        )

        major_axis = (
            eigenvectors[
                :,
                np.argmax(
                    eigenvalues
                ),
            ]
        )

        projections = (
            centred
            @ major_axis
        )

        length_cm = float(
            projections.max()
            - projections.min()
        )

        return {
            "length_cm": length_cm,
        }

    # ========================================================
    # COMPLETE MEASUREMENT PIPELINE
    # ========================================================

    def measure(
        self,
        image_rgb: np.ndarray,
        card_corners: np.ndarray,
        prompt: str,
        measurement_type: str,
        segmenter,
    ) -> dict:

        # ----------------------------------------------------
        # 1. DETECT TARGET
        # ----------------------------------------------------

        detection = (
            self.detect_target(
                image_rgb=image_rgb,
                prompt=prompt,
            )
        )

        # ----------------------------------------------------
        # 2. CROP TARGET
        # ----------------------------------------------------

        (
            target_crop,
            (
                offset_x,
                offset_y,
            ),
        ) = self.crop_target(
            image_rgb=image_rgb,
            box_xyxy=(
                detection.box_xyxy
            ),
            padding=0.05,
        )

        # ----------------------------------------------------
        # 3. SEGMENT TARGET
        # ----------------------------------------------------

        (
            target_mask,
            target_contour,
        ) = self.segment_target(
            crop_rgb=target_crop,
            segmenter=segmenter,
        )

        # ----------------------------------------------------
        # 4. CROP COORDS → ORIGINAL COORDS
        # ----------------------------------------------------

        original_points = (
            self.contour_to_original_points(
                contour=target_contour,
                offset_x=offset_x,
                offset_y=offset_y,
            )
        )

        # ----------------------------------------------------
        # 5. BUILD DIRECT CM HOMOGRAPHY
        # ----------------------------------------------------

        metric_homography = (
            self.create_metric_homography(
                card_corners=card_corners,
            )
        )

        # ----------------------------------------------------
        # 6. TARGET POINTS → CENTIMETRES
        # ----------------------------------------------------

        points_cm = (
            self.transform_points_to_cm(
                points=original_points,
                metric_homography=(
                    metric_homography
                ),
            )
        )

        # ----------------------------------------------------
        # 7. MEASURE
        # ----------------------------------------------------

        measurement_type = (
            measurement_type
            .strip()
            .lower()
        )

        if (
            measurement_type
            == "rectangle"
        ):

            values = (
                self.measure_rectangle(
                    points_cm
                )
            )

        elif (
            measurement_type
            == "circle"
        ):

            values = (
                self.measure_circle(
                    points_cm
                )
            )

        elif (
            measurement_type
            == "line"
        ):

            values = (
                self.measure_line(
                    points_cm
                )
            )

        else:
            raise ValueError(
                "measurement_type must be "
                "'rectangle', 'circle', "
                "or 'line'."
            )

        return {
            "prompt": prompt,
            "detected_label": (
                detection.label
            ),
            "measurement_type": (
                measurement_type
            ),
            "confidence": (
                detection.confidence
            ),
            "box_xyxy": (
                detection.box_xyxy
            ),
            "target_crop": (
                target_crop
            ),
            "target_mask": (
                target_mask
            ),
            "points_cm": (
                points_cm
            ),
            "values": values,
        }


def format_result(
    result: dict,
) -> str:

    lines = [
        (
            f"Target prompt: "
            f"{result['prompt']}"
        ),
        (
            f"Detected as: "
            f"{result['detected_label']}"
        ),
        (
            f"Measurement type: "
            f"{result['measurement_type']}"
        ),
        (
            f"Detection confidence: "
            f"{result['confidence']:.3f}"
        ),
        "",
    ]

    for (
        name,
        value,
    ) in result[
        "values"
    ].items():

        readable_name = (
            name.replace(
                "_",
                " "
            )
        )

        lines.append(
            f"{readable_name}: "
            f"{value:.3f}"
        )

    return "\n".join(
        lines
    )