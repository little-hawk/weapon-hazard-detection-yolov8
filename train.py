# train.py (only the key part to change)
from ultralytics import YOLO
from common import (check_gpu, YAML, MODEL, DEVICE, PROJECT, RUN_NAME, STABLE_BEST, find_latest_best)
import shutil

# override training params for CPU
EPOCHS = 40
IMGSZ  = 640
BATCH  = 24

def main():
    check_gpu()
    print(f"\n🚀 Training YOLOv8n | {EPOCHS} epochs | imgsz={IMGSZ} | batch={BATCH}\n")

    model = YOLO(MODEL)
    model.train(
        data=YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=RUN_NAME,
        optimizer="AdamW",
        lr0=0.001,
        patience=5,
        workers=2,      # better on Windows
        cache=True,     # OK with 24GB RAM
        save=True,
        plots=True,
        verbose=True,
    )

    best = find_latest_best()
    shutil.copy2(best, STABLE_BEST)
    print(f"\n✅ Training complete.")
    print(f"✅ Stable checkpoint saved to: {STABLE_BEST}")

if __name__ == "__main__":
    main()