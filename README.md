# weapon-hazard-detection-yolov8
YOLOv8 object detection model for identifying hazardous weapons (guns &amp; knives) in images. Trained on a custom labeled dataset with CSPDarkNet backbone. Includes training, evaluation, and demo inference scripts.
# 🔫🔪 Weapon Hazard Detection — YOLOv8

Detects hazardous objects (**guns** and **knives**) in images using YOLOv8.

## Classes
| ID | Class |
|----|-------|
| 0  | Gun   |
| 1  | Knife |

## Run in Google Colab (recommended)
Open `notebook/hazard_detection_yolov8.ipynb` directly in Colab — no setup needed.

## Run Locally
```bash
git clone https://github.com/yourusername/weapon-hazard-detection-yolov8.git
cd weapon-hazard-detection-yolov8
pip install -r requirements.txt

python scripts/train.py
python scripts/evaluate.py
python scripts/demo.py path/to/image.jpg
```

## Dataset
Place your dataset in the `dataset/` folder following the structure above,
or update the path in `data.yaml`.

## Results
| Split      | mAP@50 | Precision | Recall | F1 |
|------------|--------|-----------|--------|----|
| Validation | -      | -         | -      | -  |
| Test       | -      | -         | -      | -  |

*(Fill in after training)*
