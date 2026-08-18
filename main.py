from __future__ import annotations
from object_measurement import (
    GenericObjectMeasurer,
    format_result,
)

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from card_detector import CardDetector, crop_card
from config import (
    CROP_PADDING,
    IMAGE_DIR,
    OUTPUT_DIR,
    YOLO_CONFIDENCE,
    YOLO_MODEL,
)
from geometry import compute_card_corners
from homography import rectify_card
from hough import detect_card_edge_lines
from measurement import run_measurement
from preprocessing import clean_segmentation_mask
from segmentation import load_segmenter, segment_foreground
from utils import (
    create_required_directories,
    find_input_images,
    load_image,
)
from visualization import (
    create_contour_visualization,
    create_corner_visualization,
    create_hough_visualization,
    save_grayscale_image,
    save_rgb_image,
    save_text_file,
)


def map_corners_to_original(
    crop_corners: np.ndarray,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    """
    Convert card-corner coordinates from the cropped image coordinate
    system back to the original full-image coordinate system.
    """
    original_corners = np.asarray(
        crop_corners,
        dtype=np.float64,
    ).copy()

    if original_corners.shape != (4, 2):
        raise ValueError(
            "crop_corners must have shape (4, 2)."
        )

    original_corners[:, 0] += offset_x
    original_corners[:, 1] += offset_y

    return original_corners


def create_detection_visualization(
    image_rgb: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """
    Draw the YOLO bounding box on the original image.
    """
    visualization = image_rgb.copy()

    x1, y1, x2, y2 = [
        int(round(value))
        for value in np.asarray(box).reshape(-1)
    ]

    cv2.rectangle(
        visualization,
        (x1, y1),
        (x2, y2),
        color=(0, 255, 0),
        thickness=4,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        visualization,
        "card",
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3,
        lineType=cv2.LINE_AA,
    )

    return visualization


def create_original_corner_visualization(
    image_rgb: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """
    Draw the geometric card corners on the original full image.
    """
    visualization = image_rgb.copy()

    integer_corners = np.round(
        corners
    ).astype(np.int32)

    cv2.polylines(
        visualization,
        [integer_corners.reshape(-1, 1, 2)],
        isClosed=True,
        color=(255, 255, 255),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    corner_names = [
        "TL",
        "TR",
        "BR",
        "BL",
    ]

    for name, corner in zip(
        corner_names,
        integer_corners,
    ):
        x = int(corner[0])
        y = int(corner[1])

        cv2.circle(
            visualization,
            (x, y),
            radius=8,
            color=(255, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(
            visualization,
            (x, y),
            radius=10,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            visualization,
            name,
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    return visualization

def upscale_card_crop(
    crop_rgb: np.ndarray,
    minimum_long_side: int = 800,
) -> tuple[np.ndarray, float]:
    """
    Upscale small card crops before segmentation and geometry.

    Returns:
        resized_crop
        scale_factor

    scale_factor = 1.0 means no resizing was required.
    """
    height, width = crop_rgb.shape[:2]

    long_side = max(width, height)

    if long_side >= minimum_long_side:
        return crop_rgb, 1.0

    scale_factor = (
        minimum_long_side / long_side
    )

    new_width = int(
        round(width * scale_factor)
    )

    new_height = int(
        round(height * scale_factor)
    )

    resized = cv2.resize(
        crop_rgb,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return resized, scale_factor

def find_valid_card_candidate(
    image_rgb: np.ndarray,
    detector: CardDetector,
    segmenter,
):
    """
    Generate several YOLO candidates and validate each one
    using the RMBG foreground shape.
    """

    candidates = detector.get_candidates(
        image_rgb
    )

    if not candidates:
        raise RuntimeError(
            "YOLO produced no possible card candidates."
        )

    print(
        "Validating candidate shapes..."
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"\nCandidate {index}: "
            f"confidence="
            f"{candidate.confidence:.3f}, "
            f"source={candidate.source}"
        )

        try:
            crop_rgb, offset = crop_card(
                image=image_rgb,
                box=candidate.box,
                padding=CROP_PADDING,
            )

            # Upscale only for validation if candidate is small.
            height, width = crop_rgb.shape[:2]

            long_side = max(
                width,
                height,
            )

            scale = 1.0

            if long_side < 500:

                scale = (
                    500.0
                    / long_side
                )

                crop_rgb = cv2.resize(
                    crop_rgb,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            crop_pillow = (
                Image.fromarray(
                    crop_rgb
                )
            )

            raw_mask = segment_foreground(
                segmenter=segmenter,
                image_pillow=crop_pillow,
            )

            clean_mask, contour = (
                clean_segmentation_mask(
                    raw_mask
                )
            )

            contour_area = float(
                cv2.contourArea(
                    contour
                )
            )

            rect = cv2.minAreaRect(
                contour
            )

            rect_width, rect_height = (
                rect[1]
            )

            if (
                rect_width <= 0
                or rect_height <= 0
            ):
                print(
                    "Rejected: invalid rectangle."
                )
                continue

            rect_area = (
                rect_width
                * rect_height
            )

            rectangularity = (
                contour_area
                / rect_area
            )

            aspect_ratio = (
                max(
                    rect_width,
                    rect_height,
                )
                / min(
                    rect_width,
                    rect_height,
                )
            )

            print(
                f"  rectangularity="
                f"{rectangularity:.3f}"
            )

            print(
                f"  segmented aspect="
                f"{aspect_ratio:.3f}"
            )

            # Loose enough for perspective distortion,
            # strict enough to reject pen-like objects.
            if rectangularity < 0.70:
                print(
                    "Rejected: not rectangular enough."
                )
                continue

            if not (
                1.15
                <= aspect_ratio
                <= 2.60
            ):
                print(
                    "Rejected: shape too elongated "
                    "to be a card."
                )
                continue

            print(
                "*** CARD CANDIDATE ACCEPTED ***"
            )

            return (
                candidate.box,
                candidate.confidence,
            )

        except Exception as exc:

            print(
                f"Rejected because validation "
                f"failed: {exc}"
            )

            continue

    raise RuntimeError(
        "YOLO produced candidates, "
        "but none passed card validation."
    )

def process_single_image(
    image_path: Path,
    segmenter,
    detector: CardDetector,
    object_measurer: GenericObjectMeasurer,
    enable_measurement: bool,
) -> None:
    """
    Process one image through the complete pipeline.

    Pipeline:
        1. Load the full image.
        2. Detect the card using YOLO.
        3. Crop the card with padding.
        4. Segment the card crop using RMBG-1.4.
        5. Clean the segmentation mask.
        6. Detect card edges using the Hough transform.
        7. Extend and refine the four card-edge lines.
        8. Intersect the lines to obtain geometric corners.
        9. Map crop-relative corners to the original image.
        10. Compute homography.
        11. Optionally measure a target object.
        12. Save all outputs.
    """
    print("\n" + "=" * 70)
    print(f"Processing: {image_path.name}")
    print("=" * 70)

    output_folder = OUTPUT_DIR / image_path.stem
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # STEP 1: LOAD FULL IMAGE
    # --------------------------------------------------------

    _, image_rgb = load_image(
        image_path
    )

    full_height, full_width = image_rgb.shape[:2]

    print(
        f"Original image size: "
        f"{full_width} × {full_height}"
    )

    # --------------------------------------------------------
    # STEP 2: YOLO CARD DETECTION
    # --------------------------------------------------------

    print("Detecting card with YOLO...")

    try:
        detection_box, detection_confidence = (
        find_valid_card_candidate(
            image_rgb=image_rgb,
            detector=detector,
            segmenter=segmenter,
        )
    )

        print(
            f"\nValidated card confidence: "
            f"{detection_confidence:.3f}"
        )

    except RuntimeError:
        print(
            "YOLO could not detect the card."
        )
        print(
            "Trying Grounding DINO fallback..."
        )

        card_detection = object_measurer.detect_target(
            image_rgb=image_rgb,
            prompt="credit card",
        )

        detection_box = (
            card_detection.box_xyxy
        )

        print(
            f"Grounding DINO card confidence: "
            f"{card_detection.confidence:.3f}"
        )

    detection_visualization = (
        create_detection_visualization(
            image_rgb=image_rgb,
            box=detection_box,
        )
    )

    save_rgb_image(
        output_folder / "00_yolo_detection.png",
        detection_visualization,
    )

    # --------------------------------------------------------
    # STEP 3: PADDED CARD CROP
    # --------------------------------------------------------

    cropped_rgb, (offset_x, offset_y) = crop_card(
        image=image_rgb,
        box=detection_box,
        padding=CROP_PADDING,
    )

    cropped_rgb, crop_scale = upscale_card_crop(
        cropped_rgb
    )

    print(
        f"Card processing scale: "
        f"{crop_scale:.3f}x"
    )

    print(
        f"Card processing size: "
        f"{cropped_rgb.shape[1]} × "
        f"{cropped_rgb.shape[0]}"
    )

    if cropped_rgb is None or cropped_rgb.size == 0:
        raise RuntimeError(
            "YOLO produced an empty card crop."
        )

    crop_height, crop_width = cropped_rgb.shape[:2]

    print(
        f"Crop size: {crop_width} × {crop_height}"
    )
    print(
        f"Crop offset: ({offset_x}, {offset_y})"
    )

    save_rgb_image(
        output_folder / "01_card_crop.png",
        cropped_rgb,
    )

    cropped_pillow = Image.fromarray(
        cropped_rgb
    )

    # --------------------------------------------------------
    # STEP 4: FOREGROUND SEGMENTATION
    # --------------------------------------------------------

    print(
        "Running foreground segmentation "
        "on the cropped image..."
    )

    raw_mask = segment_foreground(
        segmenter=segmenter,
        image_pillow=cropped_pillow,
    )

    save_grayscale_image(
        output_folder / "02_raw_mask.png",
        raw_mask,
    )

    # --------------------------------------------------------
    # STEP 5: MASK CLEANING
    # --------------------------------------------------------

    print("Cleaning segmentation mask...")

    clean_mask, largest_contour = (
        clean_segmentation_mask(
            raw_mask
        )
    )

    save_grayscale_image(
        output_folder / "03_clean_mask.png",
        clean_mask,
    )

    contour_visualization = (
        create_contour_visualization(
            image_rgb=cropped_rgb,
            contour=largest_contour,
        )
    )

    save_rgb_image(
        output_folder / "04_largest_contour.png",
        contour_visualization,
    )

    # --------------------------------------------------------
    # STEP 6: HOUGH / FALLBACK CORNER DETECTION
    # --------------------------------------------------------

    print(
        "Detecting card edges with "
        "the Hough Transform..."
    )

    hough_success = False

    try:
        hough_result = detect_card_edge_lines(
            clean_mask=clean_mask,
            contour=largest_contour,
        )

        edge_image = hough_result.edge_image
        raw_hough_lines = hough_result.raw_lines
        selected_lines = hough_result.selected_lines

        corner_result = compute_card_corners(
            contour=largest_contour,
            selected_lines=selected_lines,
            image_shape=cropped_rgb.shape,
        )

        crop_corners = corner_result.ordered_corners
        refined_lines = corner_result.refined_lines

        hough_success = True

    except RuntimeError as exc:
        print(
            f"Hough corner detection failed: {exc}"
        )

        print(
            "Falling back to minAreaRect..."
        )

        rect = cv2.minAreaRect(
            largest_contour
        )

        box = cv2.boxPoints(
            rect
        ).astype(np.float32)

        # Order corners TL, TR, BR, BL
        sums = box.sum(axis=1)
        diffs = np.diff(
            box,
            axis=1,
        ).reshape(-1)

        crop_corners = np.zeros(
            (4, 2),
            dtype=np.float32,
        )

        crop_corners[0] = box[
            np.argmin(sums)
        ]  # TL

        crop_corners[2] = box[
            np.argmax(sums)
        ]  # BR

        crop_corners[1] = box[
            np.argmin(diffs)
        ]  # TR

        crop_corners[3] = box[
            np.argmax(diffs)
        ]  # BL

        refined_lines = []

    # --------------------------------------------------------
    # SAVE HOUGH OUTPUTS ONLY IF HOUGH WORKED
    # --------------------------------------------------------

    if hough_success:
        save_grayscale_image(
            output_folder
            / "05_edge_image.png",
            edge_image,
        )

        hough_visualization = (
            create_hough_visualization(
                image_rgb=cropped_rgb,
                raw_lines=raw_hough_lines,
                selected_lines=selected_lines,
            )
        )

        save_rgb_image(
            output_folder
            / "06_hough_lines.png",
            hough_visualization,
        )

    # --------------------------------------------------------
    # SAVE CORNER VISUALIZATION
    # --------------------------------------------------------

    corner_visualization = (
        create_corner_visualization(
            image_rgb=cropped_rgb,
            corners=crop_corners,
            refined_lines=refined_lines,
        )
    )

    save_rgb_image(
        output_folder
        / "07_crop_corners.png",
        corner_visualization,
    )

    # --------------------------------------------------------
    # STEP 7: CORNERS IN CROP COORDINATES
    # --------------------------------------------------------

    print(
        "Computing geometric card corners..."
    )

    #corner_result = compute_card_corners(
    #    contour=largest_contour,
    #    selected_lines=selected_lines,
    #    image_shape=cropped_rgb.shape,
    #)

    #crop_corners = corner_result.ordered_corners
    #refined_lines = corner_result.refined_lines

    crop_corners_original_scale = (
        crop_corners / crop_scale
    )

    crop_corner_visualization = (
        create_corner_visualization(
            image_rgb=cropped_rgb,
            corners=crop_corners,
            refined_lines=refined_lines,
        )
    )

    save_rgb_image(
        output_folder / "07_crop_corners.png",
        crop_corner_visualization,
    )

    # --------------------------------------------------------
    # STEP 8: MAP CORNERS TO ORIGINAL IMAGE
    # --------------------------------------------------------

    original_corners = map_corners_to_original(
        crop_corners=crop_corners_original_scale,
        offset_x=offset_x,
        offset_y=offset_y,
    )

    original_corner_visualization = (
        create_original_corner_visualization(
            image_rgb=image_rgb,
            corners=original_corners,
        )
    )

    save_rgb_image(
        output_folder / "08_original_image_corners.png",
        original_corner_visualization,
    )

    # --------------------------------------------------------
    # STEP 9: HOMOGRAPHY
    # --------------------------------------------------------

    print("Applying homography...")

    rectified_card, homography_matrix = rectify_card(
        image_rgb=image_rgb,
        ordered_corners=original_corners,
    )

    save_rgb_image(
        output_folder / "09_rectified_card.png",
        rectified_card,
    )

    # --------------------------------------------------------
    # STEP 10: OPTIONAL OBJECT MEASUREMENT
    # --------------------------------------------------------

    measurement_result = None

    if enable_measurement:

        print("\n" + "=" * 70)
        print("AUTOMATIC OBJECT MEASUREMENT")
        print("=" * 70)

        target_prompt = input(
            "What object should I detect? "
        ).strip()

        print("\nMeasurement type:")
        print("1 - Rectangle")
        print("2 - Circle")
        print("3 - Line")

        choice = input(
            "Choose 1, 2, or 3: "
        ).strip()

        type_map = {
            "1": "rectangle",
            "2": "circle",
            "3": "line",
        }

        if choice not in type_map:
            raise ValueError(
                "Measurement type must be 1, 2, or 3."
            )

        measurement_type = type_map[choice]

        print(
            f"\nDetecting '{target_prompt}' "
            f"automatically..."
        )

        result = object_measurer.measure(
            image_rgb=image_rgb,
            card_corners=original_corners,
            prompt=target_prompt,
            measurement_type=measurement_type,
            segmenter=segmenter,
        )

        result_text = format_result(result)

        save_rgb_image(
            output_folder / "10_target_crop.png",
            result["target_crop"],
        )

        save_grayscale_image(
            output_folder / "11_target_mask.png",
            result["target_mask"],
        )

        result_text = format_result(
            result
        )

        print("\n" + result_text)

        save_text_file(
            output_folder / "12_measurement.txt",
            result_text,
        )

        print("\n" + result_text)

        save_text_file(
            output_folder / "10_measurement.txt",
            result_text,
        )

    # --------------------------------------------------------
    # STEP 11: SAVE GEOMETRY INFORMATION
    # --------------------------------------------------------

    crop_top_left = crop_corners[0]
    crop_top_right = crop_corners[1]
    crop_bottom_right = crop_corners[2]
    crop_bottom_left = crop_corners[3]

    top_left = original_corners[0]
    top_right = original_corners[1]
    bottom_right = original_corners[2]
    bottom_left = original_corners[3]

    x1, y1, x2, y2 = [
        float(value)
        for value in np.asarray(
            detection_box
        ).reshape(-1)
    ]

    coordinate_text = (
        f"Image: {image_path.name}\n\n"

        f"Original image size:\n"
        f"{full_width} x {full_height}\n\n"

        f"YOLO detection box:\n"
        f"x1 = {x1:.3f}\n"
        f"y1 = {y1:.3f}\n"
        f"x2 = {x2:.3f}\n"
        f"y2 = {y2:.3f}\n\n"

        f"Padded crop:\n"
        f"Offset x = {offset_x}\n"
        f"Offset y = {offset_y}\n"
        f"Crop width = {crop_width}\n"
        f"Crop height = {crop_height}\n\n"

        f"Corners relative to crop:\n"
        f"Top-left:     "
        f"({crop_top_left[0]:.3f}, "
        f"{crop_top_left[1]:.3f})\n"
        f"Top-right:    "
        f"({crop_top_right[0]:.3f}, "
        f"{crop_top_right[1]:.3f})\n"
        f"Bottom-right: "
        f"({crop_bottom_right[0]:.3f}, "
        f"{crop_bottom_right[1]:.3f})\n"
        f"Bottom-left:  "
        f"({crop_bottom_left[0]:.3f}, "
        f"{crop_bottom_left[1]:.3f})\n\n"

        f"Corners relative to original image:\n"
        f"Top-left:     "
        f"({top_left[0]:.3f}, "
        f"{top_left[1]:.3f})\n"
        f"Top-right:    "
        f"({top_right[0]:.3f}, "
        f"{top_right[1]:.3f})\n"
        f"Bottom-right: "
        f"({bottom_right[0]:.3f}, "
        f"{bottom_right[1]:.3f})\n"
        f"Bottom-left:  "
        f"({bottom_left[0]:.3f}, "
        f"{bottom_left[1]:.3f})\n\n"

        f"Homography matrix:\n"
        f"{homography_matrix}\n"
    )

    save_text_file(
        output_folder / "11_geometry.txt",
        coordinate_text,
    )

    print(
        "Processing completed successfully."
    )
    print(
        f"Outputs saved to: {output_folder}"
    )


def ask_measurement_mode(
    image_count: int,
) -> bool:
    """
    Decide whether interactive target measurement should run.

    Interactive measurement is intended for one image at a time.
    """
    if image_count > 1:
        print(
            "\nInteractive measurement is disabled because "
            f"{image_count} images were found."
        )
        print(
            "Leave one image in the images folder to use "
            "rectangle, circle, or line measurement."
        )
        return False

    answer = input(
        "\nRun interactive object measurement? "
        "(y/n): "
    ).strip().lower()

    return answer in {
        "y",
        "yes",
    }


def process_all_images() -> None:
    """
    Find and process every supported image inside IMAGE_DIR.
    """
    create_required_directories()

    image_paths = find_input_images()

    if not image_paths:
        print(
            "\nNo input images were found."
        )
        print(
            f"Place images inside:\n{IMAGE_DIR}"
        )
        return

    print(
        f"\nFound {len(image_paths)} "
        f"input image(s)."
    )

    #enable_measurement = ask_measurement_mode(
    #    len(image_paths)
    #)

    print("\nLoading YOLO card detector...")

    detector = CardDetector(
        YOLO_MODEL,
        YOLO_CONFIDENCE,
    )
    object_measurer = GenericObjectMeasurer()

    print(
        "YOLO detector loaded successfully."
    )

    segmenter = load_segmenter()

    successful_images = 0
    failed_images = 0

    for image_path in image_paths:
        try:
            print(
                f"\nAbout to process: {image_path.name}"
            )

            answer = input(
                "Measure an object in this image? (y/n): "
            ).strip().lower()

            enable_measurement = answer in {
                "y",
                "yes",
            }

            process_single_image(
                image_path=image_path,
                segmenter=segmenter,
                detector=detector,
                object_measurer=object_measurer,
                enable_measurement=enable_measurement,
            )

            successful_images += 1

        except KeyboardInterrupt:
            print(
                "\nProcessing interrupted by the user."
            )
            raise

        except Exception as exc:
            failed_images += 1

            print("\n" + "!" * 70)
            print(
                f"Failed to process: {image_path.name}"
            )
            print(
                f"Reason: {exc}"
            )
            print("!" * 70)

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(
        f"Successful: {successful_images}"
    )
    print(
        f"Failed:     {failed_images}"
    )
    print(
        f"Total:      {len(image_paths)}"
    )
    print("=" * 70)


def main() -> int:
    """
    Main program entry point.
    """
    print("=" * 70)
    print(
        "CREDIT CARD GEOMETRIC "
        "CORNER DETECTION"
    )
    print(
        "YOLO + RMBG-1.4 + HOUGH TRANSFORM "
        "+ LINE INTERSECTIONS + HOMOGRAPHY"
    )
    print("=" * 70)

    try:
        process_all_images()
        return 0

    except KeyboardInterrupt:
        print("\nProgram stopped.")
        return 130

    except Exception as exc:
        print(
            "\nA fatal error occurred."
        )
        print(
            f"Reason: {exc}"
        )

        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())