Weapon Hazard Detection Dashboard (YOLOv8 + Streamlit)
This project implements a weapon hazard detection system using a trained YOLOv8 object detection model and an interactive Streamlit dashboard.

The system is designed for demonstration, evaluation, and inference, allowing users to:

Evaluate a trained model on a test dataset

Upload their own images for weapon detection

View visual explanations (confusion matrix, curves)

Analyze KPIs across all uploaded images

Large artifacts (datasets, training outputs, virtual environments) are intentionally excluded from the repository to keep it portable and compliant with GitHub size limits.

1. Features
1.1 Evaluation (Test Dataset)
Test-only evaluation: No training or re-validation required.

Metrics: mAP@50, mAP@50–95, Precision, and Recall.

Visuals: * Confusion Matrix

Precision–Recall Curve

F1 Curve

Precision/Recall Curves

Per-class performance tables.

1.2 Inference (User Images)
Upload one or multiple images simultaneously.

Detected weapons highlighted using red circular overlays.

Class name and confidence score displayed for every detection.

1.3 KPI Dashboard
Aggregated statistics: Total images, images with detections, total detection count, and detection rate.

Average confidence per class and overall.

Raw detection table: Exportable data including image name, class, confidence, and bounding box coordinates.

2. Project Structure (Simplified)
Plaintext
weapon-hazard-detection-yolov8/
├── weapon_detection/
│   ├── app.py                       # Streamlit dashboard
│   ├── weapon_training/
│   │   └── data.yaml                # Dataset configuration
│   └── ...
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Excludes venv, datasets, runs
└── README.md
3. System Requirements
Python: 3.9 – 3.11 (recommended)

Git

Internet Connection: Required for initial dependency installation

4. Environment Setup
4.1 Create a Virtual Environment
Windows (PowerShell):

PowerShell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
macOS / Linux:

Bash
python3 -m venv .venv
source .venv/bin/activate
4.2 Install Dependencies
With the virtual environment activated:

Bash
pip install -r requirements.txt
Verify the installation:

Bash
python -m streamlit --version
5. Model Weights
Due to GitHub file size limits, trained model weights (.pt files) are not guaranteed to be in the repository. The dashboard supports two methods:

Option A (Recommended): Start the dashboard and use the sidebar to upload your best.pt file directly.

Option B (Local): Place your weights in weapon_detection/weights/best.pt. The dashboard detects .pt files in that directory automatically.

6. Running the Application
Always use the module form to run the application to avoid PATH issues:

Bash
python -m streamlit run weapon_detection/app.py
The dashboard will open in your default browser at: http://localhost:8501

7. Dataset Configuration (YOLO Format)
The system uses a standard YOLO dataset layout.

Example Directory Structure:

Plaintext
dataset_splitted/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
Example data.yaml:

YAML
train: dataset_splitted/images/train
val: dataset_splitted/images/val
test: dataset_splitted/images/test
nc: 2
names:
  0: knife
  1: gun
Note: Even when evaluating test-only, some Ultralytics versions require the val directories to exist (even if they are empty).

8. Using the Dashboard
Test Evaluation: Open the “1) Test Evaluation” tab. Upload a ZIP containing your data.yaml and the test/ folders. Run evaluation to review metrics and plots.

Inference: Open the “2) Inference” tab. Upload raw images to see real-time detections with red circular highlights.

KPI: Open the “3) KPI” tab to see the breakdown of all detections processed during the current session.

9. Troubleshooting
Streamlit command not found: Ensure you use python -m streamlit instead of just the streamlit command.

No Detections: Check your confidence threshold in the sidebar and verify that your class names in data.yaml match the model training.

Dataset Errors: Ensure paths in data.yaml are relative to the root or are absolute paths. Ensure images/val exists even if empty.

10. Author
Zhyldyz Davydova
