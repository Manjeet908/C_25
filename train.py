import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MODEL_PATH, TRAIN_GOOD_DIR
from src.detector import ClassicalAnomalyDetector


def main():
    print("=" * 60)
    print("MVTec Bottle Anomaly Detection: Training (Classical IPCV)")
    print("=" * 60)

    if not TRAIN_GOOD_DIR.exists():
        print(f"Error: Training directory not found at {TRAIN_GOOD_DIR}")
        sys.exit(1)

    train_paths = sorted(list(TRAIN_GOOD_DIR.glob("*.png")))
    print(f"Discovered {len(train_paths)} normal training images in {TRAIN_GOOD_DIR}")

    t0 = time.time()
    detector = ClassicalAnomalyDetector()
    detector.fit(train_paths, save_path=MODEL_PATH)
    elapsed = time.time() - t0

    print("=" * 60)
    print(f"Training completed successfully in {elapsed:.2f} seconds!")
    print(f"Model saved to: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

