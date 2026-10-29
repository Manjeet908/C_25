import cv2
import numpy as np
from src.config import GRID_SIZE, GRID_CENTER


def detect_bottle_circle(image: np.ndarray):
    """
    Detects the outer boundary of the bottle using thresholding and contour analysis.
    Returns (cx, cy), radius
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Bottle background is white/light (> 225)
    _, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        return (float(cx), float(cy)), float(r)
    
    # Fallback to image center
    h, w = gray.shape[:2]
    return (w / 2.0, h / 2.0), min(h, w) * 0.45


def align_bottle(image: np.ndarray, target_center=GRID_CENTER, out_size=GRID_SIZE):
    """
    Downsamples the input image, locates the bottle, and centers it
    in standard coordinates.
    Returns:
        aligned_img: np.ndarray of shape (out_size, out_size, 3)
        transform_info: dict with scale, translation (dx, dy), and original dimensions
    """
    orig_h, orig_w = image.shape[:2]
    
    # Resize to working grid
    resized = cv2.resize(image, (out_size, out_size), interpolation=cv2.INTER_AREA)
    
    # Locate circle in resized coordinates
    (cx, cy), r = detect_bottle_circle(resized)
    
    # Translation to align to target center
    dx = target_center[0] - cx
    dy = target_center[1] - cy
    
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(
        resized,
        M,
        (out_size, out_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    transform_info = {
        "orig_shape": (orig_h, orig_w),
        "out_size": out_size,
        "dx": dx,
        "dy": dy,
        "circle_center": (cx, cy),
        "radius": r,
    }
    
    return aligned, transform_info


def restore_heatmap_to_original(heatmap_grid: np.ndarray, transform_info: dict) -> np.ndarray:
    """
    Warps the anomaly heatmap from grid coordinates back into the original image space.
    """
    orig_h, orig_w = transform_info["orig_shape"]
    out_size = transform_info["out_size"]
    dx = transform_info["dx"]
    dy = transform_info["dy"]
    
    # Inverse translation in grid space
    M_inv = np.float32([[1, 0, -dx], [0, 1, -dy]])
    untranslated_grid = cv2.warpAffine(
        heatmap_grid,
        M_inv,
        (out_size, out_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0
    )
    
    # Resize to original image dimensions
    heatmap_orig = cv2.resize(untranslated_grid, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return heatmap_orig

