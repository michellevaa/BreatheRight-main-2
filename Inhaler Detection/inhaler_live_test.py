"""
Inhaler Live Camera Test
=========================
Loads your trained best.pt and runs it on your webcam in real time.
Draws the bounding box and 3 keypoints on the inhaler.

Controls:
  Q   quit
  S   save current frame + label to test_captures/
"""

import cv2
import sys
import time
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR     = Path(__file__).parent
MODEL_PATH     = SCRIPT_DIR / "best.pt"
CAPTURE_DIR    = SCRIPT_DIR / "test_captures"
CAMERA_INDEX   = 0        # change to 1, 2 etc. if wrong camera opens
CONF_THRESHOLD = 0.25
IMG_SIZE       = 640

KEYPOINT_NAMES  = ["Mouthpiece", "L-corner", "Button"]
KEYPOINT_COLORS = [
    (0,   255, 100),   # KP1 – green
    (0,   180, 255),   # KP2 – cyan
    (255,  80,  80),   # KP3 – red (BGR)
]
BOX_COLOR  = (200, 200,  50)
C_WHITE    = (240, 240, 240)
C_DIM      = (110, 110, 110)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def alpha_rect(img, pt1, pt2, color, alpha=0.55, radius=8):
    ov = img.copy()
    x1, y1 = pt1;  x2, y2 = pt2;  r = radius
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(ov, (cx, cy), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def draw_detections(frame, results):
    h, w   = frame.shape[:2]
    result = results[0]
    detections = 0

    if result.boxes is None or len(result.boxes) == 0:
        return 0

    for i, box in enumerate(result.boxes):
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        detections += 1

        # Bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(frame, f"Inhaler {conf:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, BOX_COLOR, 1, cv2.LINE_AA)

        # Keypoints
        if result.keypoints is None or i >= len(result.keypoints.xy):
            continue

        kps = result.keypoints.xy[i].cpu().numpy()
        kp_vis = (result.keypoints.conf[i].cpu().numpy()
                  if result.keypoints.conf is not None
                  else [1.0, 1.0, 1.0])

        placed = []
        for k in range(min(3, len(kps))):
            px, py = int(kps[k][0]), int(kps[k][1])
            if px == 0 and py == 0:
                continue

            col = KEYPOINT_COLORS[k]
            placed.append((px, py))

            cv2.circle(frame, (px, py), 10, (0, 0, 0), 2)
            cv2.circle(frame, (px, py),  7, col, -1)
            cv2.putText(frame, KEYPOINT_NAMES[k], (px + 12, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, KEYPOINT_NAMES[k], (px + 12, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)

        for j in range(len(placed) - 1):
            cv2.line(frame, placed[j], placed[j+1], (180,180,180), 1, cv2.LINE_AA)

    return detections


def draw_hud(frame, detections, fps, saved_msg):
    h, w = frame.shape[:2]

    # Top-left info panel
    alpha_rect(frame, (6, 6), (290, 100), (15,15,15), alpha=0.62)
    det_col = (80, 220, 80) if detections > 0 else (60, 60, 220)
    det_txt = f"Inhaler DETECTED ({detections})" if detections > 0 else "No inhaler detected"
    cv2.putText(frame, det_txt,  (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, det_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_DIM,  1, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {CONF_THRESHOLD}", (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_DIM, 1, cv2.LINE_AA)

    # Top-right keypoint legend
    alpha_rect(frame, (w-192, 6), (w-6, 92), (15,15,15), alpha=0.62)
    for i, (name, col) in enumerate(zip(KEYPOINT_NAMES, KEYPOINT_COLORS)):
        y = 28 + i * 22
        cv2.circle(frame, (w-176, y-4), 6, col, -1)
        cv2.putText(frame, f"KP{i+1}: {name}", (w-163, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1, cv2.LINE_AA)

    # Bottom hint
    alpha_rect(frame, (6, h-36), (265, h-6), (15,15,15), alpha=0.55)
    cv2.putText(frame, "S: save frame    Q: quit",
                (14, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_DIM, 1, cv2.LINE_AA)

    # Save confirmation (fades after 2s)
    if saved_msg:
        cv2.putText(frame, saved_msg,
                    (14, h-50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (80,220,80), 1, cv2.LINE_AA)


def save_frame(frame, results):
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    ts         = time.strftime("%Y%m%d_%H%M%S")
    img_path   = CAPTURE_DIR / f"capture_{ts}.jpg"
    label_path = CAPTURE_DIR / f"capture_{ts}.txt"

    cv2.imwrite(str(img_path), frame)

    h, w   = frame.shape[:2]
    result = results[0]
    if result.boxes is not None and len(result.boxes) > 0:
        conf = float(result.boxes[0].conf[0])
        if conf >= CONF_THRESHOLD and result.keypoints is not None:
            x1, y1, x2, y2 = map(float, result.boxes[0].xyxy[0])
            cx = ((x1+x2)/2) / w
            cy = ((y1+y2)/2) / h
            bw = (x2-x1) / w
            bh = (y2-y1) / h
            kps    = result.keypoints.xy[0].cpu().numpy()
            kp_str = "  ".join(
                f"{float(kps[k][0])/w:.6f} {float(kps[k][1])/h:.6f} 2"
                for k in range(3)
            )
            with open(label_path, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}  {kp_str}\n")

    return f"Saved: {img_path.name}"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] best.pt not found at: {MODEL_PATH}")
        print(f"  Place best.pt in the same folder as this script.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.  pip install ultralytics")
        sys.exit(1)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded.")
    print("Controls:  S = save frame    Q = quit\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {CAMERA_INDEX}.")
        print("  Try changing CAMERA_INDEX to 1 or 2 at the top of the script.")
        sys.exit(1)

    window = "Inhaler Live Test"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    fps        = 0.0
    prev_time  = time.time()
    saved_msg  = ""
    saved_timer= 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        frame   = cv2.flip(frame, 1)
        results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)

        detections = draw_detections(frame, results)

        now       = time.time()
        fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now

        if saved_msg and now - saved_timer > 2.0:
            saved_msg = ""

        draw_hud(frame, detections, fps, saved_msg)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            saved_msg   = save_frame(frame, results)
            saved_timer = time.time()
            print(saved_msg)

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
