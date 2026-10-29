import sys
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DEFAULT_DEFECT_THRESHOLD,
    DEFAULT_SUSPICIOUS_THRESHOLD,
    MODEL_PATH,
    TEST_DIR,
)
from src.detector import ClassicalAnomalyDetector


def evaluate(
    defect_threshold: float = DEFAULT_DEFECT_THRESHOLD,
    suspicious_threshold: float = DEFAULT_SUSPICIOUS_THRESHOLD,
):
    print("=" * 75)
    print("MVTec Bottle Anomaly Detection: 3-Tier Evaluation Benchmark")
    print("  Tier 1: Good (< suspicious_threshold)")
    print("  Tier 2: Close Call / Suspicious (suspicious_threshold <= score < defect_threshold)")
    print("  Tier 3: Defective (>= defect_threshold)")
    print("=" * 75)

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}. Run train.py first.")
        sys.exit(1)

    detector = ClassicalAnomalyDetector(
        defect_threshold=defect_threshold,
        suspicious_threshold=suspicious_threshold,
    )
    detector.load(MODEL_PATH)

    categories = ["good", "broken_large", "broken_small", "contamination"]
    all_scores = []
    all_labels = []
    category_results = {}

    for cat in categories:
        cat_dir = TEST_DIR / cat
        if not cat_dir.exists():
            continue

        images = sorted(list(cat_dir.glob("*.png")))
        is_defect_cat = cat != "good"
        true_label = 1 if is_defect_cat else 0

        counts = {"GOOD": 0, "SUSPICIOUS": 0, "DEFECTIVE": 0}
        scores = []

        for img_path in images:
            img = cv2.imread(str(img_path))
            result = detector.predict(
                img,
                defect_threshold=defect_threshold,
                suspicious_threshold=suspicious_threshold,
            )
            score = result["anomaly_score"]
            status = result["status"]

            scores.append(score)
            counts[status] += 1
            all_scores.append(score)
            all_labels.append(true_label)

        category_results[cat] = {
            "total": len(images),
            "counts": counts,
            "min_score": min(scores) if scores else 0.0,
            "median_score": float(np.median(scores)) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
        }

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    auroc = roc_auc_score(y_true, y_scores)

    print("\n--- 3-Tier Classification Breakdown ---")
    print(
        f"{'Category':<16} | {'Total':<6} | {'Good (Pass)':<12} | {'Close Call (Review)':<20} | {'Defective (Fail)':<16}"
    )
    print("-" * 80)
    for cat, res in category_results.items():
        c = res["counts"]
        print(
            f"{cat:<16} | {res['total']:<6} | {c['GOOD']:<12} | {c['SUSPICIOUS']:<20} | {c['DEFECTIVE']:<16}"
        )

    # Verification of user requirement:
    good_defective_count = category_results["good"]["counts"]["DEFECTIVE"]
    print("\n" + "=" * 75)
    print("--- User Quality Requirement Verification ---")
    print(
        f"Good items incorrectly marked DEFECTIVE: {good_defective_count} / {category_results['good']['total']} "
        f"({(good_defective_count / category_results['good']['total']) * 100:.1f}%)"
    )
    if good_defective_count == 0:
        print(">>> SUCCESS: 0 good items marked defective! All borderline cases routed to Review.")
    else:
        print(">>> WARNING: Some good items were marked defective. Adjust defect_threshold.")

    print(f"Continuous ROC-AUC Score: {auroc * 100:.2f}%")
    print("=" * 75)

    return auroc


if __name__ == "__main__":
    d_thresh = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DEFECT_THRESHOLD
    s_thresh = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SUSPICIOUS_THRESHOLD
    evaluate(defect_threshold=d_thresh, suspicious_threshold=s_thresh)
