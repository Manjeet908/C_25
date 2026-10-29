import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "bottle"
TRAIN_GOOD_DIR = DATASET_DIR / "train" / "good"
TEST_DIR = DATASET_DIR / "test"
GROUND_TRUTH_DIR = DATASET_DIR / "ground_truth"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "bottle_classical_model.npz"

# Processing Dimensions
RAW_IMAGE_SIZE = (900, 900)
GRID_SIZE = 300  # (300, 300) working resolution for classical feature extraction
GRID_CENTER = (150, 150)
BOTTLE_RADIUS_RATIO = 136  # Bottle radius in grid coordinates

# Detection Thresholds (3-Tier Classification)
# Score < SUSPICIOUS_THRESHOLD          -> GOOD (PASS)
# SUSPICIOUS_THRESHOLD <= Score < DEFECT_THRESHOLD -> CLOSE CALL / REVIEW NEEDED
# Score >= DEFECT_THRESHOLD             -> DEFECTIVE (FAIL)
DEFAULT_SUSPICIOUS_THRESHOLD = 16.0
DEFAULT_DEFECT_THRESHOLD = 18.5

MIN_DEFECT_AREA = 300      # Minimum pixel area in 900x900 coordinates to count as defect blob
GAUSSIAN_BLUR_KERNEL = 11  # Heatmap smoothing kernel
