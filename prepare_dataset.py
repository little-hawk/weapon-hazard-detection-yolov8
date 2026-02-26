from pathlib import Path
import random, shutil, yaml, os, stat

# ── CONFIG ────────────────────────────────────────────────────────
SRC = Path("dataset")               # READ-ONLY source
OUT = Path("dataset_splitted")      # NEW dataset with fresh splits

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Class mapping (IMPORTANT)
# 0 = knife
# 1 = gun
CLASS_NAMES = {0: "knife", 1: "gun"}
# ──────────────────────────────────────────────────────────────────

def _on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def collect_pairs(root: Path):
    """
    Collect image/label pairs from:
      dataset/images/train + dataset/labels/train
      dataset/images/val   + dataset/labels/val
    """
    pairs = {}

    for split in ["train", "val"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue

            lbl = lbl_dir / img.with_suffix(".txt").name
            if lbl.exists():
                pairs[img.name] = (img, lbl)

    if not pairs:
        raise RuntimeError("❌ No image/label pairs found in dataset.")

    return list(pairs.values())

def copy_pairs(pairs, split):
    img_out = OUT / "images" / split
    lbl_out = OUT / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lbl, lbl_out / lbl.name)

def count_instances(pairs):
    """
    Count class instances (bounding boxes) in a list of (img, lbl) pairs.
    """
    counts = {0: 0, 1: 0}  # 0=knife, 1=gun

    for _, lbl in pairs:
        for line in lbl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls = int(line.split()[0])
            if cls in counts:
                counts[cls] += 1

    return counts

def main():
    random.seed(SEED)

    all_pairs = collect_pairs(SRC)
    total = len(all_pairs)

    print(f"📦 Total labeled images found: {total}")

    random.shuffle(all_pairs)

    n_train = int(TRAIN_RATIO * total)
    n_val   = int(VAL_RATIO * total)

    train_pairs = all_pairs[:n_train]
    val_pairs   = all_pairs[n_train:n_train + n_val]
    test_pairs  = all_pairs[n_train + n_val:]

    # wipe output safely (Windows)
    if OUT.exists():
        shutil.rmtree(OUT, onerror=_on_rm_error)

    print("\n📂 Creating dataset_splitted/")
    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    copy_pairs(test_pairs, "test")

    print("\nFinal split counts (images):")
    print(f"  Train: {len(train_pairs)}")
    print(f"  Val  : {len(val_pairs)}")
    print(f"  Test : {len(test_pairs)}")

    train_cnt = count_instances(train_pairs)
    val_cnt   = count_instances(val_pairs)
    test_cnt  = count_instances(test_pairs)

    print("\nClass instance counts (bounding boxes):")
    print("  TRAIN:")
    print(f"    Knife (0): {train_cnt[0]}")
    print(f"    Gun   (1): {train_cnt[1]}")

    print("  VAL:")
    print(f"    Knife (0): {val_cnt[0]}")
    print(f"    Gun   (1): {val_cnt[1]}")

    print("  TEST:")
    print(f"    Knife (0): {test_cnt[0]}")
    print(f"    Gun   (1): {test_cnt[1]}")

    # write data.yaml
    cfg = {
        "path": str(OUT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": CLASS_NAMES
    }

    with open("data.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print("\n✅ data.yaml written")
    print(cfg)

if __name__ == "__main__":
    main()