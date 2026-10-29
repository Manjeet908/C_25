import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.alignment import align_bottle, restore_heatmap_to_original
from src.config import (
    BOTTLE_RADIUS_RATIO,
    DEFAULT_DEFECT_THRESHOLD,
    DEFAULT_SUSPICIOUS_THRESHOLD,
    GAUSSIAN_BLUR_KERNEL,
    GRID_CENTER,
    GRID_SIZE,
    MIN_DEFECT_AREA,
    MODEL_PATH,
)
from src.feature_extractor import ClassicalFeatureExtractor


class ClassicalAnomalyDetector:
    """
    Classical IPCV Anomaly Detector for MVTec Bottle dataset.
    Supports a 3-tier classification:
      - GOOD: Clean normal bottle (Pass)
      - SUSPICIOUS / CLOSE CALL: Near the decision boundary (Review required, not marked defective)
      - DEFECTIVE: Clear anomaly detected (Fail)
    """

    def __init__(
        self,
        defect_threshold: float = DEFAULT_DEFECT_THRESHOLD,
        suspicious_threshold: float = DEFAULT_SUSPICIOUS_THRESHOLD,
    ):
        self.defect_threshold = defect_threshold
        self.suspicious_threshold = suspicious_threshold
        self.extractor = ClassicalFeatureExtractor()
        self.mean_feat: Optional[np.ndarray] = None
        self.std_feat: Optional[np.ndarray] = None
        self.weights = np.array(
            [1.0, 1.2, 1.2, 1.5, 1.0, 1.0, 1.0, 1.0, 1.2], dtype=np.float32
        )
        self.train_score_stats: Dict[str, float] = {}

        # Circular bottle mask in grid coordinates
        Y, X = np.ogrid[:GRID_SIZE, :GRID_SIZE]
        self.bottle_mask = (X - GRID_CENTER[0]) ** 2 + (Y - GRID_CENTER[1]) ** 2 < (
            BOTTLE_RADIUS_RATIO
        ) ** 2

    def fit(self, train_paths: List[Path], save_path: Optional[Path] = MODEL_PATH):
        """Fits the classical statistical model on normal training bottles."""
        print(f"Fitting Classical Anomaly Detector on {len(train_paths)} normal images...")
        features_list = []
        for p in train_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            al, _ = align_bottle(img)
            f = self.extractor.extract(al)
            features_list.append(f)

        if not features_list:
            raise ValueError("No valid training images found!")

        feat_stack = np.stack(features_list, axis=0)  # (N, H, W, D)
        self.mean_feat = np.mean(feat_stack, axis=0).astype(np.float32)
        # Regularized std to avoid division by near-zero variance
        self.std_feat = (np.std(feat_stack, axis=0) + 0.015).astype(np.float32)

        # Baseline evaluation on normal training images
        normal_scores = []
        for f in features_list:
            score, _ = self._compute_score_and_grid_map(f)
            normal_scores.append(score)

        self.train_score_stats = {
            "min": float(np.min(normal_scores)),
            "mean": float(np.mean(normal_scores)),
            "median": float(np.median(normal_scores)),
            "max": float(np.max(normal_scores)),
            "std": float(np.std(normal_scores)),
        }
        print(
            f"Fit complete! Training normal scores: "
            f"mean={self.train_score_stats['mean']:.2f}, "
            f"max={self.train_score_stats['max']:.2f}"
        )

        if save_path:
            self.save(save_path)

    def save(self, path: Path):
        """Saves model parameters to .npz file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            mean_feat=self.mean_feat,
            std_feat=self.std_feat,
            weights=self.weights,
            defect_threshold=np.array([self.defect_threshold]),
            suspicious_threshold=np.array([self.suspicious_threshold]),
            stats_mean=np.array([self.train_score_stats.get("mean", 12.0)]),
            stats_max=np.array([self.train_score_stats.get("max", 18.0)]),
        )
        print(f"Saved model to {path}")

    def load(self, path: Path = MODEL_PATH):
        """Loads model parameters from .npz file."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found at {path}. Run train.py first.")
        data = np.load(str(path))
        self.mean_feat = data["mean_feat"]
        self.std_feat = data["std_feat"]
        self.weights = data["weights"]
        if "defect_threshold" in data:
            self.defect_threshold = float(data["defect_threshold"][0])
            self.suspicious_threshold = float(data["suspicious_threshold"][0])
        elif "threshold" in data:
            self.defect_threshold = DEFAULT_DEFECT_THRESHOLD
            self.suspicious_threshold = DEFAULT_SUSPICIOUS_THRESHOLD

        self.train_score_stats = {
            "mean": float(data["stats_mean"][0]),
            "max": float(data["stats_max"][0]),
        }

    def _compute_score_and_grid_map(
        self, features: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Computes standardized multi-feature distance map and image-level anomaly score."""
        z_dist = np.abs(features - self.mean_feat) / self.std_feat
        dist_map = np.sum(z_dist * self.weights, axis=-1)
        dist_map[~self.bottle_mask] = 0.0

        smooth_map = cv2.GaussianBlur(
            dist_map, (GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_KERNEL), 0
        )
        smooth_map[~self.bottle_mask] = 0.0

        score = float(np.percentile(smooth_map[self.bottle_mask], 99.8))
        return score, smooth_map

    def predict(
        self,
        image: np.ndarray,
        defect_threshold: Optional[float] = None,
        suspicious_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Inference on an input bottle image with 3-tier classification:
        - GOOD (Pass)
        - SUSPICIOUS (Close call - review needed, not marked defective)
        - DEFECTIVE (Confirmed anomaly)
        """
        if self.mean_feat is None:
            raise RuntimeError("Model is not fitted or loaded yet.")

        t_defect = (
            defect_threshold if defect_threshold is not None else self.defect_threshold
        )
        t_suspicious = (
            suspicious_threshold
            if suspicious_threshold is not None
            else self.suspicious_threshold
        )

        # 1. Align bottle
        aligned, transform_info = align_bottle(image)

        # 2. Extract features
        features = self.extractor.extract(aligned)

        # 3. Compute anomaly score & grid map
        score, grid_map = self._compute_score_and_grid_map(features)

        # 4. Restore anomaly heatmap to original image resolution
        orig_anomaly_map = restore_heatmap_to_original(grid_map, transform_info)

        # 5. 3-Tier Classification
        if score >= t_defect:
            status = "DEFECTIVE"
            is_anomaly = True
            is_close_call = False
            message = "Defect confirmed: Anomaly score exceeds defect threshold."
            surplus = (score - t_defect) / max(1.0, t_defect * 0.8)
            confidence = float(np.clip(70.0 + surplus * 29.9, 70.0, 99.9))
        elif score >= t_suspicious:
            status = "SUSPICIOUS"
            is_anomaly = False  # DO NOT mark as defective, as user explicitly requested!
            is_close_call = True
            message = "Close call: Borderline variation detected. Flagged for review so no good item is falsely rejected."
            # Confidence represents degree of suspicion
            band_pct = (score - t_suspicious) / max(0.1, t_defect - t_suspicious)
            confidence = float(np.clip(50.0 + band_pct * 25.0, 50.0, 75.0))
        else:
            status = "GOOD"
            is_anomaly = False
            is_close_call = False
            message = "Pass: Bottle structure and appearance are within normal limits."
            safety = (t_suspicious - score) / max(1.0, t_suspicious)
            confidence = float(np.clip(50.0 + safety * 49.9, 50.0, 99.9))

        # 6. Defect Mask & Contours/Bounding Boxes
        # For DEFECTIVE or SUSPICIOUS items, localize the area of concern
        bounding_boxes = []
        defect_mask = np.zeros(transform_info["orig_shape"], dtype=np.uint8)

        if is_anomaly or is_close_call:
            # Pixel threshold scaled to the relevant boundary
            pixel_thresh = (t_defect if is_anomaly else t_suspicious) * 0.75
            binary_grid = (grid_map > pixel_thresh).astype(np.uint8) * 255

            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_opened = cv2.morphologyEx(binary_grid, cv2.MORPH_OPEN, k_open)

            defect_mask = restore_heatmap_to_original(
                binary_opened.astype(np.float32), transform_info
            )
            defect_mask = (defect_mask > 127).astype(np.uint8) * 255

            cnts, _ = cv2.findContours(
                defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in cnts:
                area = cv2.contourArea(c)
                if area >= MIN_DEFECT_AREA:
                    x, y, w, h = cv2.boundingRect(c)
                    bounding_boxes.append(
                        {
                            "x": int(x),
                            "y": int(y),
                            "w": int(w),
                            "h": int(h),
                            "area": int(area),
                        }
                    )

            if not bounding_boxes and cnts:
                c = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                bounding_boxes.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "area": int(cv2.contourArea(c)),
                    }
                )

        return {
            "is_anomaly": is_anomaly,
            "is_close_call": is_close_call,
            "status": status,
            "message": message,
            "anomaly_score": round(score, 2),
            "defect_threshold": round(t_defect, 2),
            "suspicious_threshold": round(t_suspicious, 2),
            "confidence": round(confidence, 1),
            "anomaly_map": orig_anomaly_map,
            "defect_mask": defect_mask,
            "bounding_boxes": bounding_boxes,
        }
