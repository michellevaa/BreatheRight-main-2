"""
Inhaler Auto-Labeller + Verification Tool
==========================================
Step 1 (auto-label): Run your weak YOLOv8-pose model on all unlabelled images
                     and save predictions as YOLO label files.
Step 2 (verify):     Open each auto-labelled image so you can accept, correct,
                     or reject the predicted keypoints before finalising.

Workflow:
  1. Run this script with MODE = "auto"   → generates label files for all unlabelled images
  2. Run this script with MODE = "verify" → lets you check / fix each auto-label one by one

Folder structure expected:
  images/
    ├── img001.jpg
    ├── img002.jpg
    └── ...
  labels/                  ← manually labelled images already here
    ├── img001.txt
    └── ...

Controls (verify mode):
  A / Enter   accept prediction as-is, move to next
  E           edit — clear predicted points and place manually (same as labeller)
  Left click  place keypoint (in edit mode)
  Right click undo last keypoint (in edit mode)
  R           reset to original prediction
  D           delete label and skip (mark as bad image)
  Q           quit and save progress
"""

import cv2
import os
import sys
import json
import glob
import shutil
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — edit these before running
# ══════════════════════════════════════════════════════════════════════════════

MODE           = "verify"          # "auto" first, then "verify"
MODEL_PATH     = "best.pt"       # path to your trained YOLOv8-pose weights
IMAGE_FOLDER   = "images"
LABEL_FOLDER   = "labels"        # existing + new labels go here
AUTO_PROGRESS  = "auto_progress.json"   # tracks auto-labelled files
VERIFY_PROGRESS= "verify_progress.json" # tracks verified files

CONF_THRESHOLD = 0.25            # min detection confidence to accept a prediction
# If model confidence < CONF_THRESHOLD the image is flagged for manual labelling

KEYPOINT_NAMES  = ["Mouthpiece tip", "L-corner / bend", "Button / actuator"]
KEYPOINT_COLORS = [
    (0,   255, 100),
    (0,   180, 255),
    (255,  80,  80),
]
POINT_RADIUS   = 8
WINDOW_NAME    = "Auto-Label Verifier"


# ══════════════════════════════════════════════════════════════════════════════
#  YOLO LABEL I/O
# ══════════════════════════════════════════════════════════════════════════════

def compute_bbox(kps, margin=0.05):
    xs = [p[0] for p in kps]
    ys = [p[1] for p in kps]
    xmin = max(0.0, min(xs) - margin)
    ymin = max(0.0, min(ys) - margin)
    xmax = min(1.0, max(xs) + margin)
    ymax = min(1.0, max(ys) + margin)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    return cx, cy, xmax - xmin, ymax - ymin


def save_label(label_path, kps):
    cx, cy, bw, bh = compute_bbox(kps)
    kp_str = "  ".join(f"{x:.6f} {y:.6f} 2" for x, y in kps)
    with open(label_path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}  {kp_str}\n")


def load_label(label_path, img_w, img_h):
    """Read YOLO keypoints back into list of (nx, ny) normalised coords."""
    kps = []
    with open(label_path) as f:
        parts = f.read().split()
    # format: class cx cy bw bh  kp1x kp1y v1  kp2x kp2y v2  kp3x kp3y v3
    if len(parts) >= 14:
        for i in range(3):
            base = 5 + i * 3
            kps.append((float(parts[base]), float(parts[base + 1])))
    return kps


# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESS TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def load_progress(path):
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_progress(path, done_set):
    with open(path, "w") as f:
        json.dump(list(done_set), f)


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def redraw(base_img, kps, mode_label="", confidence=None, edit_mode=False):
    canvas = base_img.copy()
    h, w   = canvas.shape[:2]

    placed = []
    for i, (nx, ny) in enumerate(kps):
        px_x = int(nx * w)
        px_y = int(ny * h)
        placed.append((px_x, px_y))
        col = KEYPOINT_COLORS[i]
        cv2.circle(canvas, (px_x, px_y), POINT_RADIUS + 3, (0,0,0), 2)
        cv2.circle(canvas, (px_x, px_y), POINT_RADIUS, col, -1)
        label = f"KP{i+1}: {KEYPOINT_NAMES[i]}"
        cv2.putText(canvas, label, (px_x + 14, px_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (px_x + 14, px_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

    if len(placed) >= 2:
        for i in range(len(placed) - 1):
            cv2.line(canvas, placed[i], placed[i+1], (200,200,200), 1, cv2.LINE_AA)

    # Instruction panel
    if edit_mode:
        next_idx = len(kps)
        if next_idx < 3:
            inst = [
                (f"EDIT MODE — place KP{next_idx+1}: {KEYPOINT_NAMES[next_idx]}",
                 KEYPOINT_COLORS[next_idx]),
                ("LEFT CLICK place  |  RIGHT CLICK undo", (180,180,180)),
                ("R  reset to prediction", (180,180,180)),
            ]
        else:
            inst = [
                ("All 3 placed — press A/Enter to accept", (80,220,80)),
                ("R  reset to prediction", (180,180,180)),
            ]
    else:
        conf_str = f"  Confidence: {confidence:.2f}" if confidence else ""
        inst = [
            (f"PREDICTED{conf_str}", (200,200,80)),
            ("A / Enter  accept", (180,180,180)),
            ("E          edit manually", (180,180,180)),
            ("D          delete / bad image", (180,180,180)),
            ("Q          quit", (180,180,180)),
        ]

    overlay = canvas.copy()
    panel_h = 22 * len(inst) + 16
    cv2.rectangle(overlay, (0,0), (400, panel_h), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    y = 20
    for text, col in inst:
        cv2.putText(canvas, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
        y += 22

    # mode badge top-right
    badge_col = (60, 180, 60) if not edit_mode else (60, 120, 230)
    badge_txt = "PREVIEW" if not edit_mode else "EDIT"
    cv2.rectangle(canvas, (w-110, 0), (w, 28), badge_col, -1)
    cv2.putText(canvas, badge_txt, (w-100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-LABEL MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_auto_label(image_dir, label_dir, auto_progress_path, model_path):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"\nLoading model: {model_path}")
    model = YOLO(str(model_path))

    all_images  = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}
    ])
    done_set    = load_progress(auto_progress_path)
    manual_done = set(
        p.stem for p in label_dir.iterdir() if p.suffix == ".txt"
    ) if label_dir.exists() else set()

    # Only process images that have no label yet (manual or auto)
    to_process = [
        p for p in all_images
        if p.name not in done_set and p.stem not in manual_done
    ]

    print(f"  Total images       : {len(all_images)}")
    print(f"  Already labelled   : {len(manual_done)}")
    print(f"  Already auto-done  : {len(done_set)}")
    print(f"  To auto-label now  : {len(to_process)}\n")

    label_dir.mkdir(parents=True, exist_ok=True)
    flagged = []   # low-confidence — needs manual attention

    for i, img_path in enumerate(to_process):
        results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        result  = results[0]

        label_path = label_dir / (img_path.stem + ".txt")

        if (result.keypoints is None
                or len(result.keypoints.xy) == 0
                or result.boxes is None
                or len(result.boxes) == 0):
            flagged.append(img_path.name)
            print(f"  [{i+1:>4}/{len(to_process)}] NO DETECTION  — {img_path.name}")
            done_set.add(img_path.name)
            continue

        # Take highest-confidence detection
        best_idx  = int(result.boxes.conf.argmax())
        conf      = float(result.boxes.conf[best_idx])
        kps_px    = result.keypoints.xy[best_idx].cpu().numpy()  # shape [3, 2]

        img       = cv2.imread(str(img_path))
        ih, iw    = img.shape[:2]

        kps_norm  = [(float(kps_px[k][0]) / iw, float(kps_px[k][1]) / ih)
                     for k in range(3)]

        save_label(str(label_path), kps_norm)
        done_set.add(img_path.name)

        flag = "  [LOW CONF]" if conf < 0.5 else ""
        print(f"  [{i+1:>4}/{len(to_process)}] conf={conf:.2f}{flag}  {img_path.name}")

        if conf < 0.5:
            flagged.append(img_path.name)

        # Save progress every 20 images
        if i % 20 == 0:
            save_progress(str(auto_progress_path), done_set)

    save_progress(str(auto_progress_path), done_set)

    print(f"\nAuto-labelling complete.")
    print(f"  Labels written : {len(to_process) - len(flagged)}")
    print(f"  Flagged (low confidence / no detection) : {len(flagged)}")
    if flagged:
        flag_path = image_dir.parent / "flagged_for_review.txt"
        with open(flag_path, "w") as f:
            f.write("\n".join(flagged))
        print(f"  Flagged list saved to: {flag_path}")
    print(f"\nNow set MODE = 'verify' and run again to review predictions.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  VERIFY MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_verify(image_dir, label_dir, verify_progress_path):
    all_images = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}
    ])

    # Only verify images that have an auto-generated label
    to_verify = [
        p for p in all_images
        if (label_dir / (p.stem + ".txt")).exists()
    ]

    done_set = load_progress(verify_progress_path)
    remaining = [p for p in to_verify if p.name not in done_set]

    print(f"\nVerify mode")
    print(f"  Images with labels : {len(to_verify)}")
    print(f"  Already verified   : {len(done_set)}")
    print(f"  Remaining          : {len(remaining)}\n")

    if not remaining:
        print("All labels verified!")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 750)

    # Mutable state shared with mouse callback
    state = {
        "kps":       [],
        "pred_kps":  [],
        "edit_mode": False,
        "img":       None,
        "img_w":     1,
        "img_h":     1,
        "confidence": None,
    }

    def mouse_cb(event, x, y, flags, param):
        if not state["edit_mode"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN and len(state["kps"]) < 3:
            state["kps"].append((x / state["img_w"], y / state["img_h"]))
            cv2.imshow(WINDOW_NAME,
                       redraw(state["img"], state["kps"],
                              edit_mode=True))
        elif event == cv2.EVENT_RBUTTONDOWN and state["kps"]:
            state["kps"].pop()
            cv2.imshow(WINDOW_NAME,
                       redraw(state["img"], state["kps"],
                              edit_mode=True))

    cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

    idx = 0
    while idx < len(remaining):
        img_path   = remaining[idx]
        label_path = label_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            done_set.add(img_path.name)
            idx += 1
            continue

        ih, iw = img.shape[:2]

        # Load predicted keypoints
        pred_kps = load_label(str(label_path), iw, ih)
        if not pred_kps:
            done_set.add(img_path.name)
            idx += 1
            continue

        state.update({
            "kps":       list(pred_kps),
            "pred_kps":  list(pred_kps),
            "edit_mode": False,
            "img":       img,
            "img_w":     iw,
            "img_h":     ih,
            "confidence": None,
        })

        progress_pct = int((len(done_set)) / len(to_verify) * 100)
        cv2.setWindowTitle(WINDOW_NAME,
            f"Verifier  |  {img_path.name}  "
            f"[{len(done_set)+1}/{len(to_verify)}  {progress_pct}%]")

        cv2.imshow(WINDOW_NAME,
                   redraw(img, state["kps"], edit_mode=False,
                          confidence=state["confidence"]))

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("a"), 13):          # Accept
                if len(state["kps"]) == 3:
                    save_label(str(label_path), state["kps"])
                    done_set.add(img_path.name)
                    save_progress(str(verify_progress_path), done_set)
                    print(f"  [ACCEPTED] {img_path.name}")
                    idx += 1
                    break
                else:
                    # Still in edit mode and not all points placed
                    warn = redraw(img, state["kps"], edit_mode=True)
                    cv2.putText(warn,
                        f"Place all 3 keypoints first! ({len(state['kps'])}/3)",
                        (10, ih - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.70, (0,50,255), 2, cv2.LINE_AA)
                    cv2.imshow(WINDOW_NAME, warn)

            elif key == ord("e"):              # Edit mode toggle
                state["edit_mode"] = True
                state["kps"]       = []
                cv2.imshow(WINDOW_NAME,
                           redraw(img, state["kps"], edit_mode=True))

            elif key == ord("r"):              # Reset to prediction
                state["kps"]       = list(state["pred_kps"])
                state["edit_mode"] = False
                cv2.imshow(WINDOW_NAME,
                           redraw(img, state["kps"], edit_mode=False,
                                  confidence=state["confidence"]))

            elif key == ord("d"):              # Delete / bad image
                if label_path.exists():
                    label_path.unlink()
                done_set.add(img_path.name)
                save_progress(str(verify_progress_path), done_set)
                print(f"  [DELETED]  {img_path.name}")
                idx += 1
                break

            elif key == ord("q"):              # Quit
                save_progress(str(verify_progress_path), done_set)
                print(f"\nProgress saved. Verified: {len(done_set)}/{len(to_verify)}")
                cv2.destroyAllWindows()
                sys.exit(0)

            else:
                cv2.imshow(WINDOW_NAME,
                           redraw(img, state["kps"],
                                  edit_mode=state["edit_mode"],
                                  confidence=state["confidence"]))

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                save_progress(str(verify_progress_path), done_set)
                cv2.destroyAllWindows()
                sys.exit(0)

    save_progress(str(verify_progress_path), done_set)
    print(f"\nAll labels verified! Total: {len(to_verify)}")
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    script_dir        = Path(__file__).parent
    image_dir         = script_dir / IMAGE_FOLDER
    label_dir         = image_dir  / LABEL_FOLDER
    auto_prog_path    = script_dir / AUTO_PROGRESS
    verify_prog_path  = script_dir / VERIFY_PROGRESS
    model_path        = script_dir / MODEL_PATH   # always relative to this script

    if not image_dir.exists():
        print(f"[ERROR] Image folder not found: {image_dir}")
        sys.exit(1)

    if not model_path.exists() and MODE == "auto":
        print(f"[ERROR] Model weights not found: {model_path}")
        print(f"  Make sure best.pt is in the same folder as this script:")
        print(f"  {script_dir}")
        sys.exit(1)

    if MODE == "auto":
        run_auto_label(image_dir, label_dir, auto_prog_path, model_path)
    elif MODE == "verify":
        run_verify(image_dir, label_dir, verify_prog_path)
    else:
        print(f"[ERROR] Unknown MODE '{MODE}'. Set MODE to 'auto' or 'verify'.")
        sys.exit(1)


if __name__ == "__main__":
    main()