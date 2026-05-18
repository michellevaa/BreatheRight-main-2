"""
Inhaler YOLOv8-Pose Training Script
=====================================
This script will:
  1. Read all labelled images from images/ and images/labels/
  2. Auto-split into train (80%) and val (20%)
  3. Generate dataset.yaml automatically
  4. Train YOLOv8n-pose on your RTX 5060 Ti (CUDA)

Folder structure expected:
  Inhaler Detection/
    ├── images/
    │     ├── img001.jpg
    │     └── labels/
    │           └── img001.txt
    └── train_model.py  ← this script
"""

import os
import shutil
import random
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR    = Path(__file__).parent
IMAGE_DIR     = SCRIPT_DIR / "images"
LABEL_DIR     = IMAGE_DIR  / "labels"
DATASET_DIR   = SCRIPT_DIR / "dataset"        # train/val split copied here
YAML_PATH     = SCRIPT_DIR / "dataset.yaml"

VAL_SPLIT     = 0.20    # 20% validation, 80% train
RANDOM_SEED   = 42

# Training hyperparameters
EPOCHS        = 150     # increased: more time to learn L-shape vs I-shape distinction
IMG_SIZE      = 640
BATCH_SIZE    = 16      # RTX 5060 Ti 16GB can handle 16-32 easily
WORKERS       = 0

IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — CHECK GPU
# ══════════════════════════════════════════════════════════════════════════════

def check_gpu():
    print("── GPU Check ──────────────────────────────────────")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  CUDA available : {gpu_name}")
        print(f"  VRAM           : {vram:.1f} GB")
        device = "0"
    else:
        print("  CUDA not available — training on CPU (very slow!)")
        print("  Fix: install the CUDA version of PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        device = "cpu"
    print()
    return device


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — COLLECT & VALIDATE LABELLED IMAGES
# ══════════════════════════════════════════════════════════════════════════════

def collect_labelled_images():
    print("── Collecting labelled images ─────────────────────")

    all_images = sorted([
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ])

    paired        = []
    missing_label = []

    for img_path in all_images:
        label_path = LABEL_DIR / (img_path.stem + ".txt")
        if label_path.exists() and label_path.stat().st_size > 0:
            paired.append((img_path, label_path))
        else:
            missing_label.append(img_path.name)

    print(f"  Total images found   : {len(all_images)}")
    print(f"  Labelled pairs found : {len(paired)}")
    if missing_label:
        print(f"  No label (skipped)   : {len(missing_label)}")

    if len(paired) < 10:
        print("\n  [ERROR] Not enough labelled images to train.")
        print("  Label at least 50-80 images using inhaler_labeller.py first.")
        raise SystemExit(1)

    print()
    return paired


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — TRAIN / VAL SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split_dataset(paired):
    print("── Splitting dataset ──────────────────────────────")

    # Clean previous split so stale images don't bleed in
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    random.seed(RANDOM_SEED)
    shuffled = paired[:]
    random.shuffle(shuffled)

    val_count   = max(1, int(len(shuffled) * VAL_SPLIT))
    val_pairs   = shuffled[:val_count]
    train_pairs = shuffled[val_count:]

    print(f"  Train : {len(train_pairs)} images")
    print(f"  Val   : {len(val_pairs)} images")
    print()

    for split in ["train", "val"]:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    def copy_pairs(pairs, split):
        for img_path, label_path in pairs:
            shutil.copy2(img_path,   DATASET_DIR / split / "images" / img_path.name)
            shutil.copy2(label_path, DATASET_DIR / split / "labels" / label_path.name)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs,   "val")

    return len(train_pairs), len(val_pairs)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — GENERATE dataset.yaml
# ══════════════════════════════════════════════════════════════════════════════

def generate_yaml():
    print("── Generating dataset.yaml ────────────────────────")

    config = {
        "path"      : str(DATASET_DIR.resolve()),
        "train"     : "train/images",
        "val"       : "val/images",
        "nc"        : 1,
        "names"     : ["inhaler"],
        "kpt_shape" : [3, 3],   # 3 keypoints, each with (x, y, visibility)
        "flip_idx"  : [0, 1, 2] # L-shaped — no left/right symmetry, no flip swap
    }

    with open(YAML_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  Saved to : {YAML_PATH}")
    print(f"  Contents :")
    with open(YAML_PATH) as f:
        for line in f:
            print(f"    {line}", end="")
    print("\n")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(device):
    print("── Training ───────────────────────────────────────")
    print(f"  Model   : yolov8n-pose.pt")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  ImgSize : {IMG_SIZE}")
    print(f"  Batch   : {BATCH_SIZE}")
    print(f"  Device  : {'GPU (CUDA)' if device != 'cpu' else 'CPU'}")
    print()

    model = YOLO("yolov8n-pose.pt")

    model.train(
        data        = str(YAML_PATH),
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = BATCH_SIZE,
        device      = device,
        workers     = WORKERS,
        project     = str(SCRIPT_DIR / "runs"),
        name        = "inhaler_pose",
        # Keypoint loss weights
        kobj        = 4.0,      # increased: forces model to rely more on keypoint geometry (L-shape)
        pose        = 12.0,     # upweight keypoint regression
        # Augmentation tuned for inhaler
        fliplr      = 0.0,      # NO horizontal flip — breaks L-shape orientation
        flipud      = 0.0,      # NO vertical flip
        degrees     = 30.0,     # rotate up to 30°
        scale       = 0.9,      # increased: generates far (zoomed out) and close (zoomed in) variations
        hsv_h       = 0.05,    # increased: teaches model colour is not a key feature
        hsv_s       = 0.9,    # increased: handles new inhaler colour variation
        hsv_v       = 0.5,    # increased: handles different lighting conditions
        mosaic      = 1.0,      # increased: more multi-object context
        multi_scale = 0.5,      # randomly resizes images during training — helps far/close detection
        plots       = True,
        verbose     = True,
    )

    weights_path = SCRIPT_DIR / "runs" / "inhaler_pose" / "weights" / "best.pt"
    print(f"\n── Training complete ──────────────────────────────")
    print(f"  Best weights : {weights_path}")
    print(f"\n  Next steps:")
    print(f"  1. Copy best.pt next to inhaler_auto_labeller.py")
    print(f"  2. Set MODE = 'auto' in inhaler_auto_labeller.py and run it")
    print(f"  3. Set MODE = 'verify' to check the auto-labels")
    print(f"  4. Once all labels are verified, retrain with all data (increase EPOCHS to 100-150)")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 52)
    print("  Inhaler YOLOv8-Pose Training")
    print("=" * 52 + "\n")

    device         = check_gpu()
    paired         = collect_labelled_images()
    n_train, n_val = split_dataset(paired)
    generate_yaml()
    train(device)