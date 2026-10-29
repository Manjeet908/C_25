import argparse
import sys
from pathlib import Path
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DEFAULT_DEFECT_THRESHOLD,
    DEFAULT_SUSPICIOUS_THRESHOLD,
    MODEL_PATH,
)
from src.detector import ClassicalAnomalyDetector
from src.visualizer import (
    create_defect_overlay,
    create_heatmap_image,
    create_tri_view_mosaic,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MVTec Bottle Anomaly Detection (IPCV Classical Pipeline)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the bottle image to inspect (.png, .jpg, .jpeg)",
    )
    parser.add_argument(
        "--defect-threshold",
        type=float,
        default=DEFAULT_DEFECT_THRESHOLD,
        help=f"Defect threshold (default: {DEFAULT_DEFECT_THRESHOLD})",
    )
    parser.add_argument(
        "--suspicious-threshold",
        type=float,
        default=DEFAULT_SUSPICIOUS_THRESHOLD,
        help=f"Suspicious / close-call threshold (default: {DEFAULT_SUSPICIOUS_THRESHOLD})",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Directory to save the visualization images (default: results)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}. Please run train.py first.")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Failed to read image from {image_path}")
        sys.exit(1)

    detector = ClassicalAnomalyDetector(
        defect_threshold=args.defect_threshold,
        suspicious_threshold=args.suspicious_threshold,
    )
    detector.load(MODEL_PATH)

    print(f"\nAnalyzing: {image_path.name}")
    result = detector.predict(
        image,
        defect_threshold=args.defect_threshold,
        suspicious_threshold=args.suspicious_threshold,
    )

    print("-" * 55)
    print(f"Status:             {result['status']}")
    print(f"Message:            {result['message']}")
    print(f"Confidence:         {result['confidence']:.1f}%")
    print(f"Anomaly Score:      {result['anomaly_score']:.2f}")
    print(f"Close-Call Boundary:{result['suspicious_threshold']:.2f}")
    print(f"Defect Boundary:    {result['defect_threshold']:.2f}")
    print(f"Regions Highlighted:{len(result['bounding_boxes'])}")

    if result["bounding_boxes"]:
        print("\nDetected Regions:")
        for i, box in enumerate(result["bounding_boxes"]):
            print(
                f"  [{i+1}] (x={box['x']}, y={box['y']}, w={box['w']}, h={box['h']}) - Area: {box['area']} px"
            )

    # Save output visualizations
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    heatmap = create_heatmap_image(image, result["anomaly_map"])
    overlay = create_defect_overlay(
        image,
        result["defect_mask"],
        result["bounding_boxes"],
        result["status"],
        result["confidence"],
    )
    tri_view = create_tri_view_mosaic(image, heatmap, overlay)

    stem = image_path.stem
    cv2.imwrite(str(save_dir / f"{stem}_heatmap.png"), heatmap)
    cv2.imwrite(str(save_dir / f"{stem}_overlay.png"), overlay)
    cv2.imwrite(str(save_dir / f"{stem}_tri_view.png"), tri_view)

    print("-" * 55)
    print(f"Saved visualizations to: {save_dir.resolve()}")
    print(f"  Heatmap:  {stem}_heatmap.png")
    print(f"  Overlay:  {stem}_overlay.png")
    print(f"  Tri-View: {stem}_tri_view.png\n")


if __name__ == "__main__":
    main()
