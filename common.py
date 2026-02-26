# common.py
import sys
from pathlib import Path
import torch

YAML   = "data.yaml"
IMGSZ  = 640
BATCH  = 12
EPOCHS = 40
MODEL  = "yolov8n.pt"
NAMES  = ["knife", "gun"]
DEVICE = 0 if torch.cuda.is_available() else "cpu"

RUNS_DIR   = Path("runs/detect")
PROJECT    = str(RUNS_DIR)
RUN_NAME   = "hazard_fast"

STABLE_DIR = Path("models")
STABLE_DIR.mkdir(parents=True, exist_ok=True)
STABLE_BEST = STABLE_DIR / "best.pt"   # stable path

def check_gpu():
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU found — running on CPU (will be slow)")

def find_latest_best() -> Path:
    # Most recent best.pt inside runs/detect/**
    candidates = sorted(RUNS_DIR.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print(" No trained model found under runs/. Train first.")
        sys.exit(1)
    return candidates[-1]

def get_model_ckpt(prefer_stable=True) -> Path:
    if prefer_stable and STABLE_BEST.exists():
        return STABLE_BEST
    return find_latest_best()