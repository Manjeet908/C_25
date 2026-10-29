import os
import sys
from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DEFAULT_DEFECT_THRESHOLD,
    DEFAULT_SUSPICIOUS_THRESHOLD,
    MODEL_PATH,
    TEST_DIR,
)
from src.detector import ClassicalAnomalyDetector
from src.visualizer import create_defect_overlay, create_heatmap_image


# Page configuration
st.set_page_config(
    page_title="MVTec Bottle Anomaly Detection",
    page_icon="🍾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pass-banner {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        color: #065F46;
        padding: 16px 20px;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .warning-banner {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        color: #92400E;
        padding: 16px 20px;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .fail-banner {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
        padding: 16px 20px;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .banner-subtext {
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector():
    """Loads and caches the classical anomaly detector model."""
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at {MODEL_PATH}. Please run `python train.py` first."
        )
        st.stop()
    detector = ClassicalAnomalyDetector(
        defect_threshold=DEFAULT_DEFECT_THRESHOLD,
        suspicious_threshold=DEFAULT_SUSPICIOUS_THRESHOLD,
    )
    detector.load(MODEL_PATH)
    return detector


detector = load_detector()

# ----------------- SIDEBAR -----------------
st.sidebar.title("🍾 Configuration")
st.sidebar.markdown(
    """
    **Project**: IPCV Anomaly Detection  
    **Method**: Classical Computer Vision (Zero Deep Learning)  
    **Dataset**: MVTec AD Bottle
    """
)
st.sidebar.markdown("---")

st.sidebar.subheader("Decision Boundaries")
thresh_range = st.sidebar.slider(
    "Good | Close-Call Review Zone | Defective",
    min_value=10.0,
    max_value=30.0,
    value=(DEFAULT_SUSPICIOUS_THRESHOLD, DEFAULT_DEFECT_THRESHOLD),
    step=0.5,
    help="Scores below lower boundary = Good. Scores between boundaries = Close Call (Review). Scores above upper boundary = Defective.",
)
suspicious_thresh, defect_thresh = thresh_range

st.sidebar.markdown(
    f"""
    - 🟢 **Good (Pass)**: Score < `{suspicious_thresh:.1f}`
    - 🟡 **Close Call (Review)**: `{suspicious_thresh:.1f}` ≤ Score < `{defect_thresh:.1f}`
    - 🔴 **Defective (Fail)**: Score ≥ `{defect_thresh:.1f}`
    """
)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Test Samples")
sample_mode = st.sidebar.selectbox(
    "Choose Test Source",
    [
        "Upload Custom Image",
        "Sample: Good Bottle",
        "Sample: Broken Large",
        "Sample: Broken Small",
        "Sample: Contamination",
    ],
)

sample_image_path = None
if sample_mode != "Upload Custom Image":
    cat_map = {
        "Sample: Good Bottle": "good",
        "Sample: Broken Large": "broken_large",
        "Sample: Broken Small": "broken_small",
        "Sample: Contamination": "contamination",
    }
    cat = cat_map[sample_mode]
    cat_dir = TEST_DIR / cat
    if cat_dir.exists():
        files = sorted(list(cat_dir.glob("*.png")))
        file_names = [f.name for f in files]
        selected_file = st.sidebar.selectbox("Select Sample Image", file_names)
        sample_image_path = cat_dir / selected_file

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Zero False Defectives**: Close-call samples are routed for secondary review rather than being falsely rejected as defective."
)

# ----------------- MAIN CONTENT -----------------
st.markdown('<div class="main-header">MVTec Bottle Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Industrial Quality Inspection using Classical Image Processing & Computer Vision (IPCV)</div>',
    unsafe_allow_html=True,
)

# Image acquisition
uploaded_file = None
image_bgr = None
display_name = ""

if sample_mode == "Upload Custom Image":
    uploaded_file = st.file_uploader(
        "Upload a bottle inspection image (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        display_name = uploaded_file.name
else:
    if sample_image_path and sample_image_path.exists():
        image_bgr = cv2.imread(str(sample_image_path))
        display_name = f"{sample_mode} ({sample_image_path.name})"

# Processing & Display
if image_bgr is not None:
    with st.spinner("Analyzing bottle with classical computer vision pipeline..."):
        # Run prediction
        result = detector.predict(
            image_bgr,
            defect_threshold=defect_thresh,
            suspicious_threshold=suspicious_thresh,
        )

        # Generate Visualizations (Tri-View)
        heatmap_img = create_heatmap_image(
            image_bgr, result["anomaly_map"], max_val=45.0
        )
        overlay_img = create_defect_overlay(
            image_bgr,
            result["defect_mask"],
            result["bounding_boxes"],
            result["status"],
            result["confidence"],
        )

        orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        heat_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        over_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    # 1. Primary Classification Banner (3 Tiers)
    if result["status"] == "DEFECTIVE":
        st.markdown(
            f"""
            <div class="fail-banner">
                🚨 DEFECT DETECTED &nbsp;|&nbsp; Status: DEFECTIVE BOTTLE &nbsp;|&nbsp; Defect Confidence: {result['confidence']:.1f}%
                <div class="banner-subtext">{result['message']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif result["status"] == "SUSPICIOUS":
        st.markdown(
            f"""
            <div class="warning-banner">
                ⚠️ CLOSE CALL / SUSPICIOUS &nbsp;|&nbsp; Status: MANUAL REVIEW REQUIRED &nbsp;|&nbsp; Score: {result['anomaly_score']:.1f}
                <div class="banner-subtext"><b>Note:</b> {result['message']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="pass-banner">
                ✅ PASS &nbsp;|&nbsp; Status: NORMAL / GOOD BOTTLE &nbsp;|&nbsp; Normal Confidence: {result['confidence']:.1f}%
                <div class="banner-subtext">{result['message']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Key Metrics Row
    m1, m2, m3, m4 = st.columns(4)

    status_color = (
        "#EF4444"
        if result["status"] == "DEFECTIVE"
        else ("#F59E0B" if result["status"] == "SUSPICIOUS" else "#10B981")
    )

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {status_color};">
                    {result['status']}
                </div>
                <div class="metric-label">Classification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{result['confidence']:.1f}%</div>
                <div class="metric-label">Confidence Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{result['anomaly_score']:.1f}</div>
                <div class="metric-label">Anomaly Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(result['bounding_boxes'])}</div>
                <div class="metric-label">Regions Highlighted</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🔍 Tri-View Visual Inspection")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            orig_rgb,
            caption=f"1. Original Image: {display_name}",
            use_container_width=True,
        )

    with col2:
        st.image(
            heat_rgb,
            caption="2. Anomaly Thermal Heatmap (Jet Colormap)",
            use_container_width=True,
        )

    with col3:
        st.image(
            over_rgb,
            caption=f"3. {'Defect' if result['status'] == 'DEFECTIVE' else ('Suspect' if result['status'] == 'SUSPICIOUS' else 'Normal')} Overlay & Bounding Boxes",
            use_container_width=True,
        )

    # 3. Defect Details Expander
    with st.expander(
        "📊 Detailed Inspection Report & Region Coordinates",
        expanded=(result["status"] != "GOOD"),
    ):
        st.write(f"**Image Dimensions**: {image_bgr.shape[1]} × {image_bgr.shape[0]} px")
        st.write(f"**Calculated Anomaly Score**: `{result['anomaly_score']}`")
        st.write(f"**Close-Call Lower Boundary**: `{suspicious_thresh:.1f}`")
        st.write(f"**Defect Upper Boundary**: `{defect_thresh:.1f}`")

        if result["bounding_boxes"]:
            st.markdown("#### Highlighted Regions:")
            box_data = []
            prefix = "Defect" if result["status"] == "DEFECTIVE" else "Suspect"
            for i, b in enumerate(result["bounding_boxes"]):
                box_data.append(
                    {
                        "Region": f"{prefix} #{i+1}",
                        "X (px)": b["x"],
                        "Y (px)": b["y"],
                        "Width (px)": b["w"],
                        "Height (px)": b["h"],
                        "Area (pixels)": b["area"],
                    }
                )
            st.table(box_data)
        else:
            st.info("No localized regions exceeded the minimum defect size threshold.")
else:
    st.info("👆 Please upload an image or select a sample bottle from the sidebar to inspect.")
