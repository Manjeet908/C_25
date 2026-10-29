# MVTec Bottle Anomaly Detection (IPCV)

A high-performance Image Processing and Computer Vision (IPCV) anomaly detection system built specifically for industrial bottle inspection on the **MVTec AD Bottle dataset**.

Built with **pure classical computer vision** using multi-channel statistical feature modeling, spatial alignment, and morphological defect localization.

---

## 🌟 Key Features

1. **3-Tier Quality Classification**:
   - 🟢 **GOOD (Pass)**: Structural and visual features are strictly within normal limits.
   - 🟡 **CLOSE CALL / SUSPICIOUS (Manual Review)**: Borderline variation near the decision boundary. Flagged for review rather than marked defective **so that no good item is falsely rejected**.
   - 🔴 **DEFECTIVE (Fail)**: Confirmed anomaly exceeding the defect threshold.
2. **Defect Localization & Bounding Boxes**: Automatically isolates and outlines defect or suspicious regions with precise coordinates and bounding boxes.
3. **Calibrated Confidence Scoring**: Generates an anomaly score and a calibrated confidence percentage (0% to 100%).
4. **Tri-View Visual Inspection**:
   - **Original Image**: The raw bottle inspection image.
   - **Thermal Anomaly Heatmap**: Continuous Jet colormap showing spatial anomaly intensity.
   - **Defect / Suspicion Overlay**: Highlighted mask (Red for Defect, Amber for Close Call) with bounding boxes.
5. **Interactive Streamlit Web Application**: Complete with dual-boundary slider (Good | Close Call | Defective) and one-click test sample selection.

---

## 📊 Benchmark Performance (MVTec Bottle Test Set)

Evaluated on all 83 images of the MVTec Bottle test benchmark with 3-tier classification:

| Category | Total | Good (Pass) | Close Call (Review) | Defective (Fail) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal / Good** | 20 | **15** | **5** | **0 (0.0% False Defectives)** |
| **Broken Large** | 20 | 0 | 0 | **20 (100.0%)** |
| **Broken Small** | 22 | 0 | 1 | **21 (95.5%)** |
| **Contamination** | 21 | 2 | 0 | **19 (90.5%)** |

- **Continuous ROC-AUC**: **98.69%**
- **False Rejection of Good Bottles**: **0 / 20 (0.0%)**

---

## 🏗️ IPCV Pipeline Architecture

```
Input Image (900x900)
       │
       ▼
1. Outer Boundary & Circle Centroid Detection (cv2.minEnclosingCircle & Moments)
       │
       ▼
2. Spatial Alignment & Canonical Centering (Translation M to (150, 150))
       │
       ▼
3. Multi-Feature Descriptors:
   ├── Perceptual Color: LAB (L, a, b) & HSV (S, V)
   ├── First-Order Gradients: Sobel Magnitude
   ├── Second-Order Sharpness: Laplacian
   ├── Local Texture Contrast: Patch Standard Deviation (7x7)
   └── Particle & Spot Response: Morphological Top-Hat / Black-Hat
       │
       ▼
4. Standardized Statistical Residual (Mahalanobis Distance vs Normal Training Model)
       │
       ▼
5. Spatial Regularization & Gaussian Smoothing Filter
       │
       ▼
6. 3-Tier Classification Decision:
   ├── Score < Suspicious Thresh (16.0)  ──►  🟢 Status: PASS (GOOD BOTTLE)
   ├── 16.0 <= Score < Defect Thresh (18.5) ──►  🟡 Status: CLOSE CALL / SUSPICIOUS (MANUAL REVIEW)
   └── Score >= Defect Thresh (18.5)     ──►  🔴 Status: FAIL (DEFECTIVE BOTTLE)
                                                   │
                                                   ▼
       7. Morphological Blob Filtering & Connected Component Contours
                                                   │
                                                   ▼
       8. Tri-View Output Generation (Original + Heatmap + Bounding Box Overlay)
```

---

## 📁 Project Directory Structure

```
C_25/
├── dataset/bottle/                 # MVTec Bottle Dataset
│   ├── train/good/                 # 209 normal training images
│   ├── test/                       # 83 test images (good, broken_large, broken_small, contamination)
│   └── ground_truth/               # Ground truth pixel masks for defective samples
├── models/
│   └── bottle_classical_model.npz  # Pre-computed normal statistical model weights
├── src/
│   ├── __init__.py
│   ├── config.py                   # Paths, dimensions, thresholds, and hyper-parameters
│   ├── alignment.py                # Bottle boundary detection and coordinate projection
│   ├── feature_extractor.py        # Classical multi-feature extractor (LAB, HSV, Sobel, Laplacian, etc.)
│   ├── detector.py                 # ClassicalAnomalyDetector class (fit, predict, save, load)
│   └── visualizer.py               # Heatmap generation and defect overlay rendering
├── app.py                          # Streamlit Interactive Web Application
├── detect.py                       # Standalone CLI inference script
├── train.py                        # Model training script
├── evaluate.py                     # 3-tier benchmark evaluation script
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

---

## 🚀 Setup & Execution

### 1. Environment Setup

#### Step 1: Navigate to the project directory
```powershell
cd C_25
```

#### Step 2: Create a virtual environment
```powershell
# Using standard Python
python -m venv .venv

# Or using Python launcher (e.g., Python 3.11)
py -3.11 -m venv .venv
```

#### Step 3: Activate the virtual environment
- **PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Note: If execution of scripts is disabled, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

- **Command Prompt (CMD):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

#### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
Run the interactive application:
```powershell
.\.venv\Scripts\streamlit run app.py
```
Open your browser at `http://localhost:8501`.
- Select from the pre-loaded benchmark samples directly on the main page.
- Or upload any custom bottle image via drag-and-drop.
- Adjust the decision boundary range slider in the sidebar to tune sensitivity.

### 3. Run Benchmark Evaluation
Evaluate performance across all 83 test benchmark images:
```powershell
.\.venv\Scripts\python evaluate.py
```

### 4. Single-Image CLI Detection
Run inference directly from the command line:
```powershell
# Test on a normal bottle (Pass):
.\.venv\Scripts\python detect.py --image dataset/bottle/test/good/001.png

# Test on a borderline bottle (Close Call):
.\.venv\Scripts\python detect.py --image dataset/bottle/test/good/002.png

# Test on a broken bottle (Defective):
.\.venv\Scripts\python detect.py --image dataset/bottle/test/broken_small/000.png
```
Visualizations (`*_heatmap.png`, `*_overlay.png`, and `*_tri_view.png`) will be automatically saved in the `results/` folder.
