# Weapon Hazard Detection Dashboard (YOLOv8 + Streamlit)

This project implements a **weapon hazard detection system** using a trained **YOLOv8 object detection model** and an interactive **Streamlit dashboard**.

The system is designed for **demonstration, evaluation, and inference**, allowing users to:
- Evaluate a trained model on a **test dataset**
- Upload their own images for **weapon detection**
- View **visual explanations** (confusion matrix, curves)
- Analyse **KPIs** across all uploaded images

Large artifacts (datasets, training outputs, virtual environments) are intentionally excluded from the repository to keep it portable and compliant with GitHub size limits.

---

## 1. Features

### 1.1 Evaluation (Test Dataset)
- Test-only evaluation (no training or re-validation)
- Metrics:
  - mAP@50
  - mAP@50–95
  - Precision
  - Recall
- Visuals:
  - Confusion Matrix
  - Precision–Recall Curve
  - F1 Curve
  - Precision Curve
  - Recall Curve
- Per-class performance table

### 1.2 Inference (User Images)
- Upload one or multiple images
- Detected weapons highlighted using **red circular overlays**
- Class name + confidence displayed
- Supports multiple detections per image

### 1.3 KPI Dashboard
- Aggregated statistics across all uploaded images:
  - Total images
  - Images with detections
  - Total detections
  - Detection rate
  - Average confidence
- Per-class detection counts and average confidence
- Raw detection table (image, class, confidence, bounding box)

---

## 2. Project Structure (Simplified)


weapon-hazard-detection-yolov8/
├── weapon_detection/
│ ├── app.py # Streamlit dashboard
│ ├── weapon_training/
│ │ └── data.yaml # Dataset configuration (relative paths)
│ └── ...
├── requirements.txt # Python dependencies
├── .gitignore # Excludes venv, datasets, runs, large files
└── README.md


---

## 3. System Requirements

- **Python 3.9 – 3.11** (recommended)
- **Git**
- Internet connection (for installing dependencies)

---

## 4. Environment Setup

### 4.1 Create a Virtual Environment

#### Windows (PowerShell)

From the repository root:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

You should see (.venv) at the start of your terminal prompt.

macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
5. Install Dependencies

With the virtual environment activated:

pip install -r requirements.txt

Verify Streamlit installation:

python -m streamlit --version
6. Model Weights

Due to GitHub file size limits, trained model weights are not guaranteed to be stored in the repository.

The dashboard supports two loading methods:

Option A (Recommended): Upload Weights via UI

Start the dashboard

In the sidebar, upload best.pt

Option B: Local Weights

Place the weights locally, for example:

weapon_detection/weights/best.pt

The dashboard will detect available .pt files automatically if configured.

7. Running the Application

Always use the module form to avoid PATH issues:

python -m streamlit run weapon_detection/app.py

The dashboard will open in your browser at:

http://localhost:8501
8. Dataset Configuration (YOLO Format)

The system uses a standard YOLO dataset layout.

Example Directory Structure
dataset_splitted/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
Example data.yaml
train: dataset_splitted/images/train
val: dataset_splitted/images/val
test: dataset_splitted/images/test
nc: 2
names:
  0: knife
  1: gun

Note: Even when evaluating test-only, some Ultralytics versions require the val directories to exist (they may be empty).

YOLO automatically resolves labels using the rule:

images/.../file.jpg  →  labels/.../file.txt
9. Using the Dashboard
9.1 Test Evaluation

Open “1) Test Evaluation”

Upload a ZIP containing:

data.yaml

images/test/

labels/test/

Run evaluation

Review metrics and plots

9.2 Inference

Open “2) Inference”

Upload images

View detections with red circular highlights

9.3 KPI

Open “3) KPI”

Review aggregated metrics across all uploaded images

Inspect per-class statistics and raw detection tables

10. Git & Repository Policy

The following are intentionally excluded from version control:

.venv/

runs/

Datasets

Large ZIP files

Torch / OpenCV binaries

This ensures:

GitHub compatibility

Fast cloning

Reproducible environments via requirements.txt

11. Troubleshooting
Streamlit command not found

Use:

python -m streamlit run weapon_detection/app.py
Model loads but no detections

Check confidence threshold

Verify correct weights were loaded

Ensure class names match training

Dataset not found during evaluation

Ensure relative paths in data.yaml

Ensure images/val exists (even if empty)

12. Author

Zhyldyz Davydova
Bachelor of Computer Science & Artificial Intelligence
Murdoch University Dubai
