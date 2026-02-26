# demo.py
import argparse, sys
from pathlib import Path
from ultralytics import YOLO
from common import IMGSZ, PROJECT, NAMES, get_model_ckpt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", "-i", required=True)
    args = ap.parse_args()

    img = Path(args.image)
    if not img.exists():
        print(f" Image not found: {img}")
        sys.exit(1)

    ckpt = get_model_ckpt(prefer_stable=True)
    print(f"Using checkpoint: {ckpt}")
    model = YOLO(str(ckpt))

    results = model.predict(
        source=str(img),
        imgsz=IMGSZ,
        conf=0.25,
        save=True,
        project=PROJECT,
        name="demo",
        verbose=False,
    )

    r0 = results[0]
    print(f"\nDetections: {len(r0.boxes)}")
    for box in r0.boxes:
        cls_id = int(box.cls.item())
        conf   = float(box.conf.item())
        name   = NAMES[cls_id] if cls_id < len(NAMES) else f"class_{cls_id}"
        print(f" - {name}: {conf:.2%}")

if __name__ == "__main__":
    main()