# evaluate.py
from ultralytics import YOLO
from common import YAML, IMGSZ, NAMES, PROJECT, get_model_ckpt

def main():
    ckpt = get_model_ckpt(prefer_stable=True)
    print(f" Using checkpoint: {ckpt}")
    model = YOLO(str(ckpt))

    for split in ["val", "test"]:
        r = model.val(
            data=YAML,
            split=split,
            imgsz=IMGSZ,
            conf=0.25,
            iou=0.50,
            plots=True,
            verbose=True,
            project=PROJECT,
            name=f"eval_{split}",
        )

        p  = r.box.mp
        re = r.box.mr
        f1 = 2 * p * re / (p + re + 1e-9)

        print(f"\n{'='*50}")
        print(f"  {split.upper()} RESULTS")
        print(f"{'='*50}")
        print(f"  mAP@50     : {r.box.map50:.4f}")
        print(f"  mAP@50-95  : {r.box.map:.4f}")
        print(f"  Precision  : {p:.4f}")
        print(f"  Recall     : {re:.4f}")
        print(f"  F1-Score   : {f1:.4f}")
        print(f"\n  Per-class mAP@50:")
        for name, ap in zip(NAMES, r.box.ap50):
            print(f"    {name:6}: {ap:.4f}")

if __name__ == "__main__":
    main()