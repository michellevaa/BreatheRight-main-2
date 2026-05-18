"""
Inhaler Keypoint Labelling Tool v2
====================================
Click 3 keypoints on each image in order:
  1. Mouthpiece tip
  2. L-corner / bend
  3. Button / actuator

Controls:
  Left click        – place next keypoint (visible, v=2)
  O + Left click    – place occluded keypoint (hidden by mouth/hand, v=1)
  Right click       – undo last keypoint
  S / Enter         – save label & next image
  N                 – negative sample (no inhaler here) — saves empty label instantly
  R                 – redo current image (clear all points)
  D                 – skip image (no label saved, won't appear again)
  Q                 – quit and save progress

Visibility values:
  v=2  fully visible keypoint  (normal click)
  v=1  occluded keypoint       (O + click — e.g. mouthpiece hidden by mouth)
  v=0  missing / not in frame  (not used here)

Output format (YOLO pose):
  Positive: 0 cx cy bw bh  kp1x kp1y v1  kp2x kp2y v2  kp3x kp3y v3
  Negative: empty file  (tells YOLO "nothing to detect here")
"""

import cv2
import sys
import json
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
IMAGE_FOLDER  = "images"
LABEL_FOLDER  = "labels"
PROGRESS_FILE = "labelling_progress.json"

KEYPOINT_NAMES  = ["Mouthpiece tip", "L-corner / bend", "Button / actuator"]
KEYPOINT_COLORS = [
    (0,   255, 100),   # KP1 – green
    (0,   180, 255),   # KP2 – cyan
    (255,  80,  80),   # KP3 – red
]
OCCLUDED_RING_COLOR = (80, 80, 220)   # blue ring around occluded points
POINT_RADIUS        = 8
LINE_COLOR          = (200, 200, 200)
IMG_EXTS            = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW_NAME         = "Inhaler Labeller v2"

# ── Global state ───────────────────────────────────────────────────────────────
# Each keypoint stored as (nx, ny, visibility)  where visibility = 1 or 2
keypoints    = []
img_display  = None
img_h = img_w = 0
o_held       = False   # True while O key is held down


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_progress(path):
    if path.exists():
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_progress(path, done_set):
    with open(path, "w") as f:
        json.dump(list(done_set), f)


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


def save_positive_label(label_path, kps):
    """Save a positive label with keypoints. kps = list of (nx, ny, vis)."""
    cx, cy, bw, bh = compute_bbox([(x, y) for x, y, v in kps])
    kp_str = "  ".join(f"{x:.6f} {y:.6f} {v}" for x, y, v in kps)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}  {kp_str}\n")


def save_negative_label(label_path):
    """Save an empty label file — negative sample, nothing to detect."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("")


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def redraw(base_img, kps, o_mode=False):
    canvas = base_img.copy()
    h, w   = canvas.shape[:2]

    # ── Draw placed keypoints ─────────────────────────────────────────────
    placed = []
    for i, (nx, ny, vis) in enumerate(kps):
        px_x = int(nx * w)
        px_y = int(ny * h)
        placed.append((px_x, px_y))

        col       = KEYPOINT_COLORS[i]
        occluded  = (vis == 1)

        # Outer ring — blue if occluded, black if visible
        ring_col = OCCLUDED_RING_COLOR if occluded else (0, 0, 0)
        cv2.circle(canvas, (px_x, px_y), POINT_RADIUS + 3, ring_col, 2)

        # Fill — semi-transparent look for occluded
        fill_col = tuple(int(c * 0.5) for c in col) if occluded else col
        cv2.circle(canvas, (px_x, px_y), POINT_RADIUS, fill_col, -1)

        # Dashed outline for occluded
        if occluded:
            cv2.circle(canvas, (px_x, px_y), POINT_RADIUS + 5,
                       OCCLUDED_RING_COLOR, 1, cv2.LINE_AA)

        # Label
        suffix = " [OCC]" if occluded else ""
        label  = f"KP{i+1}: {KEYPOINT_NAMES[i]}{suffix}"
        cv2.putText(canvas, label, (px_x + 14, px_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (px_x + 14, px_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

    # Connecting lines
    if len(placed) >= 2:
        for i in range(len(placed) - 1):
            cv2.line(canvas, placed[i], placed[i+1], LINE_COLOR, 1, cv2.LINE_AA)

    # ── Instruction panel ─────────────────────────────────────────────────
    panel_lines = []

    if len(kps) < 3:
        next_idx = len(kps)
        next_col = KEYPOINT_COLORS[next_idx]
        if o_mode:
            panel_lines.append(("O HELD — click to place OCCLUDED:", OCCLUDED_RING_COLOR))
        else:
            panel_lines.append(("LEFT CLICK to place:", (200,200,200)))
        panel_lines.append((f"  KP{next_idx+1}: {KEYPOINT_NAMES[next_idx]}", next_col))
    else:
        panel_lines.append(("All 3 keypoints placed!", (80,220,80)))

    panel_lines += [
        ("", (130,130,130)),
        ("O + click    occluded keypoint",  OCCLUDED_RING_COLOR),
        ("RIGHT CLICK  undo last point",    (150,150,150)),
        ("N            negative sample",    (200,120, 40)),
        ("R            redo (clear all)",   (150,150,150)),
        ("S / ENTER    save & next",        (150,150,150)),
        ("D            skip image",         (150,150,150)),
        ("Q            quit",               (150,150,150)),
    ]

    overlay = canvas.copy()
    panel_h = 22 * len(panel_lines) + 16
    cv2.rectangle(overlay, (0,0), (360, panel_h), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    y = 20
    for text, col in panel_lines:
        if text:
            cv2.putText(canvas, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
        y += 22

    # O-mode badge
    if o_mode:
        cv2.rectangle(canvas, (w-160, 0), (w, 28), OCCLUDED_RING_COLOR, -1)
        cv2.putText(canvas, "OCCLUDED MODE", (w-152, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)

    return canvas


def flash_message(base_img, text, color=(0,60,230)):
    """Show a temporary warning on screen."""
    h = base_img.shape[0]
    warn = base_img.copy()
    cv2.putText(warn, text, (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, warn)


# ══════════════════════════════════════════════════════════════════════════════
#  MOUSE CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def mouse_callback(event, x, y, flags, param):
    global keypoints, img_display, img_h, img_w, o_held

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(keypoints) < 3:
            nx  = x / img_w
            ny  = y / img_h
            vis = 1 if o_held else 2
            keypoints.append((nx, ny, vis))
            cv2.imshow(WINDOW_NAME, redraw(img_display, keypoints, o_held))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if keypoints:
            keypoints.pop()
            cv2.imshow(WINDOW_NAME, redraw(img_display, keypoints, o_held))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global keypoints, img_display, img_h, img_w, o_held

    script_dir    = Path(__file__).parent
    image_dir     = script_dir / IMAGE_FOLDER
    label_dir     = image_dir  / LABEL_FOLDER
    progress_path = script_dir / PROGRESS_FILE

    if not image_dir.exists():
        print(f"[ERROR] Image folder not found: {image_dir}")
        sys.exit(1)

    label_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ])

    if not all_images:
        print(f"[ERROR] No images found in {image_dir}")
        sys.exit(1)

    done_set  = load_progress(progress_path)
    remaining = [p for p in all_images if p.name not in done_set]
    total     = len(all_images)
    labelled  = total - len(remaining)

    print(f"Inhaler Keypoint Labeller v2")
    print(f"  Total images  : {total}")
    print(f"  Already done  : {labelled}")
    print(f"  Remaining     : {len(remaining)}")
    print(f"  Labels dir    : {label_dir}")
    print(f"\n  O + click = occluded keypoint")
    print(f"  N         = negative sample (no inhaler)\n")

    if not remaining:
        print("All images labelled!")
        sys.exit(0)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 750)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    idx = 0
    while idx < len(remaining):
        img_path   = remaining[idx]
        label_path = label_dir / (img_path.stem + ".txt")

        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"  [SKIP] Could not read: {img_path.name}")
            done_set.add(img_path.name)
            idx += 1
            continue

        img_h, img_w = raw.shape[:2]
        img_display  = raw.copy()
        keypoints    = []
        o_held       = False

        progress_pct = int((labelled + idx) / total * 100)
        cv2.setWindowTitle(WINDOW_NAME,
            f"Inhaler Labeller v2  |  {img_path.name}  "
            f"[{labelled + idx + 1}/{total}  {progress_pct}%]")

        cv2.imshow(WINDOW_NAME, redraw(img_display, keypoints, o_held))

        while True:
            key = cv2.waitKey(20) & 0xFF

            # Track O key held state
            if key == ord("o"):
                o_held = not o_held          # toggle occluded mode on keypress
                cv2.imshow(WINDOW_NAME, redraw(img_display, keypoints, o_held))
                continue

            if key in (ord("s"), 13):        # S / Enter — save positive label
                if len(keypoints) == 3:
                    save_positive_label(label_path, keypoints)
                    done_set.add(img_path.name)
                    save_progress(progress_path, done_set)
                    occ_count = sum(1 for _, _, v in keypoints if v == 1)
                    tag = f"  ({occ_count} occluded)" if occ_count else ""
                    print(f"  [SAVED]    {img_path.name}{tag}")
                    idx += 1
                    break
                else:
                    flash_message(img_display,
                        f"Need all 3 keypoints! ({len(keypoints)}/3 placed)")

            elif key == ord("n"):            # N — negative sample
                save_negative_label(label_path)
                done_set.add(img_path.name)
                save_progress(progress_path, done_set)
                print(f"  [NEGATIVE] {img_path.name}")
                idx += 1
                break

            elif key == ord("r"):            # R — redo
                keypoints = []
                o_held    = False
                cv2.imshow(WINDOW_NAME, redraw(img_display, keypoints, o_held))

            elif key == ord("d"):            # D — skip
                done_set.add(img_path.name)
                save_progress(progress_path, done_set)
                print(f"  [SKIPPED]  {img_path.name}")
                idx += 1
                break

            elif key == ord("q"):            # Q — quit
                save_progress(progress_path, done_set)
                print(f"\nProgress saved. Done: {labelled + idx}/{total}")
                cv2.destroyAllWindows()
                sys.exit(0)

            else:
                cv2.imshow(WINDOW_NAME, redraw(img_display, keypoints, o_held))

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                save_progress(progress_path, done_set)
                cv2.destroyAllWindows()
                sys.exit(0)

    save_progress(progress_path, done_set)
    print(f"\nAll images labelled! Total: {total}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
