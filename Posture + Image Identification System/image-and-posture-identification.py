"""
Posture Detection Monitor v5
==========================
Designed for patients at ~45° to the camera (either side).

Checks (active, each worth 33.3% of posture score):
  1. Spine upright  – ear→chest angle vs vertical
                      pass: angle <= SPINE_ANGLE_THRESHOLD (10°)
  2. Chin level     – nose must sit slightly BELOW ear level (head straight,
                      not tilted up or drooping too far down)
                      pass: 0 < (nose.y - ear.y) <= MAX_CHIN_DROP
                      fail if nose.y <= ear.y  → head tilted up
                      fail if nose.y  > ear.y + MAX_CHIN_DROP → chin drooping
  3. Lateral lean   – shoulder height asymmetry vs fixed threshold

Display only (no scoring):
  4. Chest open     – reminder note only

Good posture = ALL 3 criteria must pass.
Posture score = average of all 3 individual scores (0–100%).

Session data is appended to: session_data/userid_data.csv

Controls:  Q = quit
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import os
from datetime import datetime

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

# ── Tunable constants ──────────────────────────────────────────────────────────
SMOOTH_WIN            = 12     # rolling-average window (frames)
MIN_VIS               = 0.60   # landmark visibility threshold
SPINE_ANGLE_THRESHOLD = 10.0    # degrees from vertical
LATERAL_THRESHOLD     = 0.20   # normalised shoulder-height difference

# Chin level bounds (normalised coords; larger y = lower on screen).
# Correct: nose sits slightly BELOW ear level.
#   nose.y must be > ear.y                       → strictly fails if head tilts up
#   nose.y must be <= ear.y + MAX_CHIN_DROP_RATIO * ref_len → fails if chin droops too far
# 0.15 ≈ nose can drop up to ~15% of the ear-shoulder distance below ear level.
MAX_CHIN_DROP_RATIO   = 0.15       # halved from 0.30 — stricter upper bound on drooping

# ── Output folder & file ───────────────────────────────────────────────────────
SESSION_DIR  = "session_data"
USER_ID      = "userid"          # change to the actual user ID when deploying

CSV_HEADER = (
    "session_start_time,session_end_time,session_duration,"
    "good_posture_time,bad_posture_time,"
    "average_score,peak_score,"
    "spine_upright_average_score,chin_level_average_score,no_lateral_lean_average_score"
)

# ── Colours ────────────────────────────────────────────────────────────────────
C_GOOD  = (80,  210,  80)
C_BAD   = (60,   60, 230)
C_WARN  = (40,  180, 230)
C_NOTE  = (200, 160,  40)
C_WHITE = (240, 240, 240)
C_DIM   = (110, 110, 110)

# ── Metric keys ────────────────────────────────────────────────────────────────
METRIC_KEYS = ["spine_angle", "spine_signed_angle", "nose_ear_offset", "lateral"]


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION FILE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_session_csv(start_dt: datetime, session: dict,
                     score_log: list, criterion_logs: dict) -> str:
    """
    Append one row of session data to userid_data.csv.
    Creates the file with a header if it does not yet exist.
    Returns the filepath written to.
    """
    os.makedirs(SESSION_DIR, exist_ok=True)
    filepath = os.path.join(SESSION_DIR, f"{USER_ID}_data.csv")

    end_dt   = datetime.now()
    duration = (end_dt - start_dt).total_seconds()

    good_t  = session["good_time"]
    bad_t   = session["bad_time"]

    avg = lambda lst: round(sum(lst) / len(lst), 2) if lst else 0.0

    row = ",".join(str(v) for v in [
        start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        fmt_duration(duration),
        fmt_duration(good_t),
        fmt_duration(bad_t),
        avg(score_log),
        round(max(score_log), 2) if score_log else 0.0,
        avg(criterion_logs["spine"]),
        avg(criterion_logs["chin"]),
        avg(criterion_logs["lateral"]),
    ])

    # Write header only if file is new / empty
    write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
    with open(filepath, "a", encoding="utf-8", newline="") as f:
        if write_header:
            f.write(CSV_HEADER + "\n")
        f.write(row + "\n")

    return filepath

def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"





# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def px(nx, ny, fw, fh):
    return int(nx * fw), int(ny * fh)


def detect_facing_side(lm):
    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    if ls.z != rs.z:
        return 'left' if ls.z < rs.z else 'right'
    return 'left' if ls.visibility >= rs.visibility else 'right'


def vector_angle_from_vertical(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    return np.degrees(np.arccos(np.clip(dy / length, -1.0, 1.0)))


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_metrics(lm, fw, fh):
    n  = lm[mp_pose.PoseLandmark.NOSE.value]
    le = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
    re = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    if any(p.visibility < MIN_VIS for p in [n, le, re, ls, rs]):
        return None

    mid_ear_x = (le.x + re.x) / 2
    mid_ear_y = (le.y + re.y) / 2
    mid_sh_x  = (ls.x + rs.x) / 2
    mid_sh_y  = (ls.y + rs.y) / 2

    ref_len = abs(mid_sh_y - mid_ear_y)
    if ref_len < 0.01:
        return None

    side = detect_facing_side(lm)

    # Absolute spine angle (for score gradient)
    abs_spine_angle = vector_angle_from_vertical(mid_ear_x, mid_ear_y, mid_sh_x, mid_sh_y)

    # Signed spine angle — captures which direction the chest deviates from ear.
    # Convention (after cv2.flip mirror): x increases left→right.
    #   Facing RIGHT: chest should be to the RIGHT of ear → mid_sh_x > mid_ear_x is correct.
    #                 signed = mid_sh_x - mid_ear_x  (positive = correct lean direction)
    #   Facing LEFT : chest should be to the LEFT of ear → mid_sh_x < mid_ear_x is correct.
    #                 signed = mid_ear_x - mid_sh_x  (positive = correct lean direction)
    # Positive signed angle → leaning in the acceptable direction (away from camera).
    # Zero or negative    → upright or leaning toward camera → use abs angle for threshold.
    if side == 'right':
        lateral_deviation = mid_ear_x - mid_sh_x   # positive = chest LEFT of ear = correct for right-facing
    else:
        lateral_deviation = mid_sh_x - mid_ear_x   # positive = chest RIGHT of ear = correct for left-facing

    # Convert the lateral deviation to a signed angle in degrees.
    # We reuse the magnitude from abs_spine_angle but flip it if lean is wrong direction.
    spine_signed_angle = abs_spine_angle if lateral_deviation >= 0 else -abs_spine_angle

    # nose_ear_offset: signed, normalised by ref_len.
    #   positive  → nose BELOW ears (correct direction)
    #   negative  → nose ABOVE ears (head tilted up — always fail)
    nose_ear_offset = (n.y - mid_ear_y) / ref_len

    return dict(
        spine_angle        = abs_spine_angle,
        spine_signed_angle = spine_signed_angle,
        nose_ear_offset    = nose_ear_offset,
        lateral            = abs(ls.y - rs.y) / ref_len,
        _nose_px    = px(n.x,       n.y,       fw, fh),
        _mid_ear_px = px(mid_ear_x, mid_ear_y, fw, fh),
        _mid_sh_px  = px(mid_sh_x,  mid_sh_y,  fw, fh),
        _side       = side,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════════════════════

def individual_scores(sm):
    # ── 1. Spine (directional) ────────────────────────────────────────────────
    # spine_signed_angle convention:
    #   positive → chest leaning in the CORRECT direction (away from camera)
    #   negative → chest leaning toward the camera (always fail, even slightly)
    #   zero     → perfectly vertical
    #
    # Pass condition: 0 <= signed_angle <= SPINE_ANGLE_THRESHOLD
    #   Any lean toward camera (negative) = immediate fail.
    #   Any lean away from camera beyond threshold = fail.
    #
    # Score uses the absolute angle for a smooth gradient, but is capped at 0
    # the moment the direction is wrong.
    signed = sm["spine_signed_angle"]
    abs_a  = sm["spine_angle"]

    if signed < 0:
        # Leaning toward camera — score reflects how far off, but never passes.
        spine_score = max(0.0, 1.0 - abs_a / (SPINE_ANGLE_THRESHOLD * 2)) * 50  # 0–50 only
        spine_pass  = False
    else:
        # Leaning in the correct direction (or perfectly upright).
        spine_score = max(0.0, 1.0 - abs_a / (SPINE_ANGLE_THRESHOLD * 2)) * 100
        spine_pass  = abs_a <= SPINE_ANGLE_THRESHOLD

    # ── 2. Chin level (range check) ───────────────────────────────────────────
    # nose_ear_offset is normalised by ref_len (ear→shoulder distance).
    # Pass zone: 0 < offset <= MAX_CHIN_DROP_RATIO
    #   offset <= 0              → nose at/above ear level = head tilted up  (fail)
    #   offset > MAX_CHIN_DROP_RATIO → nose too far below ears = chin drooping (fail)
    offset = sm["nose_ear_offset"]

    if offset <= 0:
        # Head tilted up: score grades from 0 (very negative) toward 100 as
        # offset approaches 0 from below. At 0 it is exactly on the boundary.
        # We treat the whole upward region as fail with a score that shows
        # how far off the person is (closer to 0 = closer to correct).
        chin_score = max(0.0, 1.0 + offset / MAX_CHIN_DROP_RATIO) * 50  # 0–50 only
        chin_pass  = False
    elif offset > MAX_CHIN_DROP_RATIO:
        # Chin drooping: score falls from 100 at the upper bound toward 0.
        excess     = offset - MAX_CHIN_DROP_RATIO
        chin_score = max(0.0, 1.0 - excess / MAX_CHIN_DROP_RATIO) * 50  # 0–50 only
        chin_pass  = False
    else:
        # Inside the good zone: score 50–100, peaking at the midpoint.
        mid        = MAX_CHIN_DROP_RATIO / 2.0
        distance   = abs(offset - mid) / mid   # 0 at centre, 1 at edges
        chin_score = 100.0 - distance * 50.0   # 50–100
        chin_pass  = True

    # ── 3. Lateral lean ───────────────────────────────────────────────────────
    lateral_score = max(0.0, 1.0 - sm["lateral"] / (LATERAL_THRESHOLD * 2)) * 100
    lateral_pass  = sm["lateral"] <= LATERAL_THRESHOLD

    overall = (spine_score + chin_score + lateral_score) / 3.0
    good    = spine_pass and chin_pass and lateral_pass

    return dict(
        spine   = dict(score=spine_score,   ok=spine_pass),
        chin    = dict(score=chin_score,    ok=chin_pass),
        lateral = dict(score=lateral_score, ok=lateral_pass),
        overall = overall,
        good    = good,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def alpha_rect(img, pt1, pt2, color, alpha=0.58, radius=10):
    ov = img.copy()
    x1, y1 = pt1;  x2, y2 = pt2;  r = radius
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(ov, (cx, cy), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def draw_check_row(img, x, y, label, ok, criterion_score):
    col  = C_GOOD if ok else C_BAD
    mark = "OK" if ok else "!!"
    cv2.rectangle(img, (x, y-14), (x+28, y+4), col, -1)
    cv2.putText(img, mark, (x+2, y+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (10,10,10), 1, cv2.LINE_AA)
    cv2.putText(img, label, (x+36, y+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"{int(criterion_score)}%", (x+270, y+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)


def draw_spine_overlay(image, metrics, good):
    col     = C_GOOD if good else C_BAD
    nose    = metrics["_nose_px"]
    mid_ear = metrics["_mid_ear_px"]
    mid_sh  = metrics["_mid_sh_px"]
    cv2.line(image, mid_ear, (mid_ear[0], mid_sh[1]), C_DIM, 1, cv2.LINE_AA)
    cv2.line(image, mid_sh,  mid_ear, col, 2, cv2.LINE_AA)
    cv2.line(image, mid_ear, nose,    col, 2, cv2.LINE_AA)
    for pt, lbl in [(mid_sh,"CHEST"),(mid_ear,"EAR"),(nose,"NOSE")]:
        cv2.circle(image, pt, 7, col, -1)
        cv2.circle(image, pt, 7, C_WHITE, 1)
        cv2.putText(image, lbl, (pt[0]+10, pt[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


def draw_hud(image, sm, result, session):
    fh, fw = image.shape[:2]
    good   = result["good"]
    col    = C_GOOD if good else C_BAD

    # ── Main panel ─────────────────────────────────────────────────────────
    panel_w = fw // 2 + 20
    alpha_rect(image, (6, 6), (panel_w, 200), (18,18,18), alpha=0.62)

    verdict = "GOOD POSTURE" if good else "ADJUST POSTURE"
    cv2.putText(image, verdict, (16, 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2, cv2.LINE_AA)
    cv2.putText(image, f"Posture Score: {int(result['overall'])}%", (16, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_WHITE, 1, cv2.LINE_AA)

    draw_check_row(image, 16, 104,
                   "Spine upright", result["spine"]["ok"], result["spine"]["score"])
    draw_check_row(image, 16, 134,
                   "Chin level  (head straight)", result["chin"]["ok"], result["chin"]["score"])
    draw_check_row(image, 16, 164,
                   "No lateral lean", result["lateral"]["ok"], result["lateral"]["score"])

    # Chest note
    alpha_rect(image, (6, 204), (panel_w, 228), (30,24,10), alpha=0.55)
    cv2.putText(image, "Please keep your chest open.",
                (14, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_NOTE, 1, cv2.LINE_AA)

    # Facing indicator
    side = sm.get("_side", "")
    if side:
        cv2.putText(image, f"Facing: {side.upper()} side",
                    (fw - 180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_WARN, 1, cv2.LINE_AA)

    # ── Session stats bottom-left (time-based) ─────────────────────────────
    good_t  = session["good_time"]
    bad_t   = session["bad_time"]
    total_t = good_t + bad_t
    good_pct = int(good_t / total_t * 100) if total_t > 0 else 0
    elapsed  = int(time.time() - session["start"])

    alpha_rect(image, (6, fh-68), (370, fh-6), (18,18,18), alpha=0.55)
    cv2.putText(image,
                f"Session {fmt_duration(elapsed)}   Good posture: {good_pct}%",
                (14, fh-44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(image,
                f"Good: {fmt_duration(good_t)}   Bad: {fmt_duration(bad_t)}",
                (14, fh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_DIM, 1, cv2.LINE_AA)

    # Bottom hint
    cv2.putText(image, "Q: quit",
                (fw//2 - 30, fh-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

start_dt = datetime.now()

histories = {k: deque(maxlen=SMOOTH_WIN) for k in METRIC_KEYS}

# Session: track time in seconds (incremented by actual frame delta)
session = dict(
    good_time = 0.0,
    bad_time  = 0.0,
    start     = time.time(),
)

# Per-second logging (snapshot taken every ~1 s)
score_log      = []
criterion_logs = {"spine": [], "chin": [], "lateral": []}
last_log_time  = time.time()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Posture Detection Monitor v5")
print(f"  Spine threshold  : {SPINE_ANGLE_THRESHOLD}° from vertical")
print(f"  Chin drop max    : {MAX_CHIN_DROP_RATIO} x ear-shoulder dist")
print(f"  Lateral threshold: {LATERAL_THRESHOLD} (normalised)")
print(f"  Session data dir : {os.path.abspath(SESSION_DIR)}")
print("  Q to quit.\n")

prev_time = time.time()

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now       = time.time()
        frame_dt  = now - prev_time   # actual seconds since last frame
        prev_time = now

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        current_result = None   # used for per-second logging

        if results.pose_landmarks:
            lm      = results.pose_landmarks.landmark
            metrics = extract_metrics(lm, fw, fh)

            # Dimmed skeleton
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(70,70,70), thickness=1, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(55,55,55), thickness=1)
            )

            if metrics is None:
                cv2.putText(image, "Low landmark visibility — adjust position",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_WARN, 2)
            else:
                for key in METRIC_KEYS:
                    histories[key].append(metrics[key])

                sm = {k: float(np.mean(histories[k])) for k in METRIC_KEYS}
                sm["_side"] = metrics["_side"]

                result = individual_scores(sm)
                current_result = result

                # Accumulate time using real frame duration
                if result["good"]:
                    session["good_time"] += frame_dt
                else:
                    session["bad_time"]  += frame_dt

                draw_spine_overlay(image, metrics, result["good"])
                draw_hud(image, sm, result, session)

        else:
            cv2.putText(image, "No person detected",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.80, C_WARN, 2)

        # ── Per-second snapshot for data log ──────────────────────────────
        if now - last_log_time >= 1.0 and current_result is not None:
            score_log.append(current_result["overall"])
            criterion_logs["spine"].append(current_result["spine"]["score"])
            criterion_logs["chin"].append(current_result["chin"]["score"])
            criterion_logs["lateral"].append(current_result["lateral"]["score"])
            last_log_time = now

        cv2.imshow("Posture Detection Monitor", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE SESSION + PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

good_t  = session["good_time"]
bad_t   = session["bad_time"]
total_t = good_t + bad_t
elapsed = time.time() - session["start"]
good_pct = (good_t / total_t * 100) if total_t > 0 else 0.0

print(f"\n── Session Summary {'─'*30}")
print(f"  Duration     : {fmt_duration(elapsed)}")
print(f"  Good posture : {fmt_duration(good_t)}  ({good_pct:.1f}%)")
print(f"  Bad posture  : {fmt_duration(bad_t)}  ({100 - good_pct:.1f}%)")

if score_log:
    saved = save_session_csv(start_dt, session, score_log, criterion_logs)
    print(f"\n  Session saved : {os.path.abspath(saved)}")
else:
    print("\n  No posture data recorded — file not saved.")