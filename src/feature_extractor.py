import cv2
import numpy as np


class ClassicalFeatureExtractor:
    """
    Extracts multi-channel classical computer vision features from an aligned image:
    - LAB color space (L, a, b)
    - HSV color space (S, V)
    - Spatial gradients (Sobel magnitude)
    - Second derivative / edge sharpness (Laplacian)
    - Local texture variance (patch standard deviation)
    - Morphological Top-Hat / Black-Hat (particle & speck detection)
    """

    def __init__(self):
        self.feature_names = [
            "lab_l", "lab_a", "lab_b",
            "hsv_s", "hsv_v",
            "sobel_mag", "laplacian",
            "local_std",
            "tophat"
        ]

    def extract(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts multi-feature tensor from aligned BGR image.
        Returns: np.ndarray of shape (H, W, num_features) with dtype float32.
        """
        # Convert color spaces
        lab = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 1. LAB channels (normalized to 0..1)
        l_chan = lab[:, :, 0] / 255.0
        a_chan = lab[:, :, 1] / 255.0
        b_chan = lab[:, :, 2] / 255.0

        # 2. HSV Saturation & Value
        s_chan = hsv[:, :, 1] / 255.0
        v_chan = hsv[:, :, 2] / 255.0

        # 3. Spatial Gradients (Sobel)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(gx**2 + gy**2) / 255.0

        # 4. Laplacian (high-frequency fracture / crack response)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)) / 255.0

        # 5. Local Texture Contrast (local standard deviation)
        mean_local = cv2.blur(gray, (7, 7))
        mean_sq_local = cv2.blur(gray**2, (7, 7))
        local_std = np.sqrt(np.maximum(0.0, mean_sq_local - mean_local**2)) / 255.0

        # 6. Morphological Top-Hat / Black-Hat (specks and stains)
        gray_u8 = gray.astype(np.uint8)
        k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        w_top = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, k_morph)
        b_top = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, k_morph)
        tophat = np.maximum(w_top, b_top).astype(np.float32) / 255.0

        features = np.stack(
            [l_chan, a_chan, b_chan, s_chan, v_chan, sobel_mag, lap, local_std, tophat],
            axis=-1
        )
        return features.astype(np.float32)

