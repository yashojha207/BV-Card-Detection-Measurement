from pathlib import Path

# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "images"

OUTPUT_DIR = BASE_DIR / "outputs"

# ============================================================
# IMAGE TYPES
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# ============================================================
# SEGMENTATION
# ============================================================

MODEL_NAME = "briaai/RMBG-1.4"

MASK_THRESHOLD = 128

# ============================================================
# MORPHOLOGY
# ============================================================

MORPH_KERNEL_SIZE = 3

MORPH_OPEN_ITERATIONS = 1

MORPH_CLOSE_ITERATIONS = 2

# ============================================================
# FOREGROUND VALIDATION
# ============================================================

MIN_FOREGROUND_AREA_RATIO = 0.01

MAX_FOREGROUND_AREA_RATIO = 0.98

# ============================================================
# CANNY EDGE DETECTOR
# ============================================================

CANNY_LOW_THRESHOLD = 50

CANNY_HIGH_THRESHOLD = 150

# ============================================================
# HOUGH TRANSFORM
# ============================================================

HOUGH_RHO = 1

HOUGH_THETA = 1 * 3.1415926535 / 180

HOUGH_THRESHOLD = 50

HOUGH_MIN_LINE_LENGTH = 60

HOUGH_MAX_LINE_GAP = 15

# ============================================================
# LINE CLUSTERING
# ============================================================

ANGLE_CLUSTER_THRESHOLD = 10

DISTANCE_CLUSTER_THRESHOLD = 25

MIN_LINE_LENGTH = 40

# ============================================================
# LINE REFINEMENT
# ============================================================

FITLINE_DISTANCE_THRESHOLD = 8

# ============================================================
# DRAWING
# ============================================================

RAW_LINE_THICKNESS = 1

REFINED_LINE_THICKNESS = 3

CORNER_RADIUS = 8

CORNER_TEXT_OFFSET = 12

# ============================================================
# HOMOGRAPHY
# ============================================================

OUTPUT_CARD_WIDTH = 856

OUTPUT_CARD_HEIGHT = 540

# ============================================================
# DEBUG FLAGS
# ============================================================

SAVE_INTERMEDIATE_RESULTS = True

VERBOSE = True

# ============================================================
# YOLO DETECTOR
# ============================================================

YOLO_MODEL = BASE_DIR / "models" / "card_detector.pt"

YOLO_CONFIDENCE = 0.25

CROP_PADDING = 0.10