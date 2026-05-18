"""
Background Image Helper
========================
Point this script at a folder of background-only images (no inhaler).
It creates an empty .txt label file for each image, which tells YOLO
"there is nothing to detect here" — this is how you add negative samples.

Usage:
  1. Take 50-80 photos of your environment WITHOUT any inhaler in frame
  2. Put them in a folder called  background_images/  next to this script
  3. Run this script
  4. The images + empty labels get copied into images/ and images/labels/
     so they're included in the next training run automatically
"""

import shutil
from pathlib import Path

SCRIPT_DIR       = Path(__file__).parent
BACKGROUND_DIR   = SCRIPT_DIR / "background_images"   # put your bg photos here
IMAGE_DIR        = SCRIPT_DIR / "images"
LABEL_DIR        = IMAGE_DIR  / "labels"
IMG_EXTS         = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not BACKGROUND_DIR.exists():
        print(f"[ERROR] Folder not found: {BACKGROUND_DIR}")
        print(f"  Create a folder called 'background_images' next to this script")
        print(f"  and put your background-only photos in it.")
        return

    bg_images = sorted([
        p for p in BACKGROUND_DIR.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ])

    if not bg_images:
        print(f"No images found in {BACKGROUND_DIR}")
        return

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    copied  = 0
    skipped = 0

    print(f"Found {len(bg_images)} background images.")
    print(f"Copying to {IMAGE_DIR} with empty labels...\n")

    for img_path in bg_images:
        # Prefix with bg_ to avoid name collisions
        dest_name  = "bg_" + img_path.name
        dest_img   = IMAGE_DIR / dest_name
        dest_label = LABEL_DIR / (dest_img.stem + ".txt")

        if dest_img.exists():
            skipped += 1
            continue

        # Copy image
        shutil.copy2(img_path, dest_img)

        # Write empty label file — empty = no detections = negative sample
        dest_label.write_text("")

        print(f"  + {dest_name}")
        copied += 1

    print(f"\nDone.")
    print(f"  Copied  : {copied} background images")
    print(f"  Skipped : {skipped} (already existed)")
    print(f"\nNow retrain using train_model.py — the background images")
    print(f"will be included automatically in the next split.")


if __name__ == "__main__":
    main()
