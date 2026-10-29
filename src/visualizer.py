from typing import Dict, List, Optional
import cv2
import numpy as np


def create_heatmap_image(
    original_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    max_val: float = 40.0,
    blend_alpha: float = 0.6,
) -> np.ndarray:
    """
    Renders the continuous anomaly map as a thermal heatmap blended over the original image.
    """
    clipped = np.clip(anomaly_map / max(1.0, max_val), 0.0, 1.0)
    uint8_map = (clipped * 255).astype(np.uint8)

    colored_heatmap = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)

    blended = cv2.addWeighted(
        original_bgr, 1.0 - blend_alpha, colored_heatmap, blend_alpha, 0
    )
    return blended


def create_defect_overlay(
    original_bgr: np.ndarray,
    defect_mask: np.ndarray,
    bounding_boxes: List[Dict],
    status: str,
    confidence: float,
) -> np.ndarray:
    """
    Overlays defect masks and draws bounding boxes according to classification:
    - DEFECTIVE: Red highlight and red bounding boxes
    - SUSPICIOUS: Amber / Orange highlight and amber bounding boxes (Close call)
    - GOOD: Green pass indicator
    """
    overlay = original_bgr.copy()

    if status == "DEFECTIVE":
        # Red highlight for confirmed defects
        mask_color = (0, 0, 230)
        box_color = (0, 0, 255)
        text_color = (255, 255, 255)
        banner_bg = (0, 0, 220)
        banner_text = f"FAIL: DEFECTIVE ({confidence:.1f}%)"

        mask_3ch = defect_mask[:, :, None] > 0
        tint_layer = np.full_like(original_bgr, mask_color, dtype=np.uint8)
        blended = cv2.addWeighted(original_bgr, 0.55, tint_layer, 0.45, 0)
        overlay = np.where(mask_3ch, blended, overlay)

        for i, box in enumerate(bounding_boxes):
            bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), box_color, 3)

            label = f"Defect #{i+1} ({box['area']} px)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                overlay,
                (bx, max(0, by - th - 8)),
                (bx + tw + 6, max(th + 4, by)),
                box_color,
                cv2.FILLED,
            )
            cv2.putText(
                overlay,
                label,
                (bx + 3, max(th, by - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
                cv2.LINE_AA,
            )

        cv2.rectangle(overlay, (20, 20), (390, 70), banner_bg, cv2.FILLED)
        cv2.putText(
            overlay,
            banner_text,
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    elif status == "SUSPICIOUS":
        # Amber / Orange highlight for close calls
        mask_color = (0, 140, 255)  # Orange BGR
        box_color = (0, 165, 255)   # Amber
        banner_bg = (0, 140, 255)
        banner_text = f"REVIEW: CLOSE CALL ({confidence:.1f}%)"

        mask_3ch = defect_mask[:, :, None] > 0
        tint_layer = np.full_like(original_bgr, mask_color, dtype=np.uint8)
        blended = cv2.addWeighted(original_bgr, 0.65, tint_layer, 0.35, 0)
        overlay = np.where(mask_3ch, blended, overlay)

        for i, box in enumerate(bounding_boxes):
            bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), box_color, 2)

            label = f"Suspect #{i+1} ({box['area']} px)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                overlay,
                (bx, max(0, by - th - 8)),
                (bx + tw + 6, max(th + 4, by)),
                box_color,
                cv2.FILLED,
            )
            cv2.putText(
                overlay,
                label,
                (bx + 3, max(th, by - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.rectangle(overlay, (20, 20), (450, 70), banner_bg, cv2.FILLED)
        cv2.putText(
            overlay,
            banner_text,
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    else:
        # Green banner for normal bottle
        banner_text = f"PASS: GOOD ({confidence:.1f}%)"
        cv2.rectangle(overlay, (20, 20), (350, 70), (0, 180, 0), cv2.FILLED)
        cv2.putText(
            overlay,
            banner_text,
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def create_tri_view_mosaic(
    original_bgr: np.ndarray, heatmap_bgr: np.ndarray, overlay_bgr: np.ndarray
) -> np.ndarray:
    """Concatenates the three views into a single horizontal composite."""
    return np.hstack([original_bgr, heatmap_bgr, overlay_bgr])
