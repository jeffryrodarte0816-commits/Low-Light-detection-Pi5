"""
fusion_decision.py
==================
Decision-Level (Late) Fusion Pipeline for Jetson Orin Nano
- Arducam B0506  -> YOLO visual inference
- TOPDON TC001   -> YOLO thermal inference
- Fusion Layer   -> Merges detections via IoU + confidence voting
- Output         -> Single fused display with color-coded detection sources

Detection Box Colors:
    GREEN  = Confirmed by BOTH sensors (V+T fused)
    BLUE   = Visual only  (Arducam)
    RED    = Thermal only (TC001)

FIXES vs previous version:
  - auto_detect_cameras() unpacked correctly (returns 4 values, was unpacked as 2)
  - Camera open retry loop added (handles V4L2 device-busy after calibrate_offset.py)
  - normalize_thermal_frame() added (handles TC001 256x384 double-height quirk)
  - ASCII-only window name (no em-dash Qt crash)

Usage:
    source /home/sunnysquad/venv/bin/activate
    python3 fusion_decision.py

Controls:
    P   : Print current calibration offset + detection stats
    ESC : Quit
"""

import cv2
import numpy as np
import time
import json
import os
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
VISUAL_MODEL_PATH  = "/home/sunnysquad/yolo_project/ultralytics/runs/detect/test_camera/weights/best.pt"
THERMAL_MODEL_PATH = "/home/sunnysquad/yolo_project/ultralytics/runs/detect/test_camera/weights/best.pt"
OFFSET_FILE        = "/home/sunnysquad/yolo_project/camera_offset.json"

# ─────────────────────────────────────────────────────────────────
# HARDWARE
# ─────────────────────────────────────────────────────────────────
VIS_WIDTH    = 1920
VIS_HEIGHT   = 1080
THERM_WIDTH  = 256
THERM_HEIGHT = 192
DISP_WIDTH   = 1280
DISP_HEIGHT  = 720

# ─────────────────────────────────────────────────────────────────
# YOLO SETTINGS
# ─────────────────────────────────────────────────────────────────
YOLO_CONF    = 0.35
YOLO_IOU     = 0.45
YOLO_IMGSZ   = 640
YOLO_EVERY_N = 2

# ─────────────────────────────────────────────────────────────────
# FUSION SETTINGS
# ─────────────────────────────────────────────────────────────────
FUSION_IOU_THRESHOLD = 0.3
FUSION_CONF_BOOST    = 0.1
THERMAL_COLORMAP     = cv2.COLORMAP_INFERNO
THERMAL_ALPHA        = 0.35

SRC_VISUAL  = "V"
SRC_THERMAL = "T"
SRC_FUSED   = "V+T"

COLOR_VISUAL  = (255, 180,   0)   # Blue
COLOR_THERMAL = (0,    80, 255)   # Red
COLOR_FUSED   = (0,   220,   0)   # Green

WIN_MAIN = "Decision Fusion | Arducam + TC001 + YOLO"


# ─────────────────────────────────────────────────────────────────
# THERMAL FRAME NORMALISATION
# ─────────────────────────────────────────────────────────────────
def normalize_thermal_frame(frame):
    """
    Converts TC001 raw frame to clean BGR 256x192.
    Handles the 256x384 double-height YUYV quirk.
    """
    h = frame.shape[0]
    if h == 384:
        frame = frame[:192, :]
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


# ─────────────────────────────────────────────────────────────────
# CAMERA DETECTION — with retry on device-busy
# ─────────────────────────────────────────────────────────────────
def auto_detect_cameras(max_retries=5, retry_delay=1.5):
    """
    Scan /dev/video0..9 for Arducam and TC001.
    Retries if devices are busy (e.g. just released by calibrate_offset.py).

    Returns: vis_cap, therm_cap, vis_idx, therm_idx
    """
    for attempt in range(1, max_retries + 1):
        print(f"Scanning cameras (attempt {attempt}/{max_retries})...")
        vis_cap = therm_cap = None
        vis_idx = therm_idx = -1

        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue

            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"  /dev/video{i}: {int(w)}x{int(h)}")

            # TC001: 256 wide, 192 or 384 tall
            if w == 256 and (h == 192 or h == 384) and therm_cap is None:
                print(f"    [OK] TOPDON TC001 at video{i}")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)
                therm_cap = cap
                therm_idx = i

            # Arducam: width >= 640
            elif w >= 640 and vis_cap is None:
                print(f"    [OK] Arducam B0506 at video{i}")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  VIS_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIS_HEIGHT)
                vis_cap = cap
                vis_idx = i
            else:
                cap.release()

        if vis_cap is not None and therm_cap is not None:
            print(f"  Both cameras found.\n")
            return vis_cap, therm_cap, vis_idx, therm_idx

        # Release whatever was found before retrying
        if vis_cap:
            vis_cap.release()
        if therm_cap:
            therm_cap.release()

        if attempt < max_retries:
            print(f"  Not all cameras found — waiting {retry_delay}s before retry...")
            time.sleep(retry_delay)

    return None, None, -1, -1


# ─────────────────────────────────────────────────────────────────
# OFFSET LOADER
# ─────────────────────────────────────────────────────────────────
def load_offset(path):
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        ox = data.get("offset_x", 0)
        oy = data.get("offset_y", 0)
        print(f"[OFFSET] Loaded: x={ox}px, y={oy}px  (from {path})")
        return ox, oy
    print(f"[OFFSET] No file at {path} — defaulting to (0, 0).")
    print(f"         Run calibrate_offset.py first.")
    return 0, 0


# ─────────────────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────────────────
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────
# DETECTION EXTRACTION
# ─────────────────────────────────────────────────────────────────
def extract_detections(results, frame_w, frame_h, offset_x=0, offset_y=0):
    dets = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 += offset_x;  x2 += offset_x
            y1 += offset_y;  y2 += offset_y
            x1 = max(0, min(x1, frame_w));  x2 = max(0, min(x2, frame_w))
            y1 = max(0, min(y1, frame_h));  y2 = max(0, min(y2, frame_h))
            dets.append({
                "box_norm": [x1/frame_w, y1/frame_h, x2/frame_w, y2/frame_h],
                "conf":     float(box.conf[0]),
                "cls":      int(box.cls[0]),
                "label":    result.names[int(box.cls[0])]
            })
    return dets


# ─────────────────────────────────────────────────────────────────
# DECISION FUSION
# ─────────────────────────────────────────────────────────────────
def decision_fusion(vis_dets, therm_dets):
    fused         = []
    matched_therm = set()

    for v in vis_dets:
        best_iou  = 0
        best_tidx = -1
        for tidx, t in enumerate(therm_dets):
            if tidx in matched_therm:
                continue
            if t["cls"] != v["cls"]:
                continue
            iou = compute_iou(v["box_norm"], t["box_norm"])
            if iou > best_iou:
                best_iou  = iou
                best_tidx = tidx

        if best_iou >= FUSION_IOU_THRESHOLD and best_tidx >= 0:
            t   = therm_dets[best_tidx]
            matched_therm.add(best_tidx)
            vw  = v["conf"];  tw = t["conf"];  total = vw + tw
            avg_box = [(v["box_norm"][i]*vw + t["box_norm"][i]*tw)/total
                       for i in range(4)]
            fused.append({
                "box_norm": avg_box,
                "conf":     min(1.0, max(v["conf"], t["conf"]) + FUSION_CONF_BOOST),
                "cls":      v["cls"],
                "label":    v["label"],
                "source":   SRC_FUSED,
                "iou":      best_iou
            })
        else:
            fused.append({**v, "source": SRC_VISUAL, "iou": 0})

    for tidx, t in enumerate(therm_dets):
        if tidx not in matched_therm:
            fused.append({**t, "source": SRC_THERMAL, "iou": 0})

    return fused


# ─────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────
def draw_detections(frame, detections, disp_w, disp_h):
    for det in detections:
        x1n, y1n, x2n, y2n = det["box_norm"]
        x1 = int(x1n * disp_w);  y1 = int(y1n * disp_h)
        x2 = int(x2n * disp_w);  y2 = int(y2n * disp_h)
        src   = det["source"]
        color = COLOR_FUSED  if src == SRC_FUSED  else \
                COLOR_VISUAL if src == SRC_VISUAL else COLOR_THERMAL
        thick = 3 if src == SRC_FUSED else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        tag = f"[{src}] {det['label']} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return frame


def draw_legend(frame):
    items = [
        ("V+T  Fused",   COLOR_FUSED),
        ("V    Visual",  COLOR_VISUAL),
        ("T    Thermal", COLOR_THERMAL),
    ]
    x0, y0 = frame.shape[1] - 200, 15
    for i, (text, color) in enumerate(items):
        y = y0 + i * 28
        cv2.rectangle(frame, (x0, y), (x0 + 18, y + 18), color, -1)
        cv2.putText(frame, text, (x0 + 24, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    offset_x, offset_y = load_offset(OFFSET_FILE)

    print(f"\nLoading visual YOLO model...")
    vis_model   = YOLO(VISUAL_MODEL_PATH)
    print(f"Loading thermal YOLO model...")
    therm_model = YOLO(THERMAL_MODEL_PATH)
    print("  Models loaded.\n")

    # FIX: unpack all 4 return values from auto_detect_cameras()
    vis_cap, therm_cap, vis_idx, therm_idx = auto_detect_cameras()

    if vis_cap is None or therm_cap is None:
        print("\n[ERROR] Could not open both cameras.")
        print("  - Make sure calibrate_offset.py is fully closed first")
        print("  - Check: ls /dev/video*")
        print("  - Check: v4l2-ctl --list-devices")
        return

    print("\n" + "-"*50)
    print("  DECISION-LEVEL FUSION ACTIVE")
    print("  GREEN  = Both sensors agree (V+T fused)")
    print("  BLUE   = Visual only")
    print("  RED    = Thermal only")
    print("  P      = Print stats | ESC = Quit")
    print("-"*50 + "\n")

    frame_count   = 0
    vis_results   = None
    therm_results = None
    fps_timer     = time.time()
    fps           = 0.0
    fps_frames    = 0
    stats = {"fused": 0, "visual_only": 0, "thermal_only": 0, "frames": 0}

    while True:
        ret_v, frame_vis   = vis_cap.read()
        ret_t, frame_therm = therm_cap.read()

        if not ret_v or not ret_t:
            print("Frame read error -- retrying...")
            time.sleep(0.05)
            continue

        frame_count         += 1
        fps_frames          += 1
        stats["frames"]      = frame_count

        # Normalise thermal (handles 256x384 quirk)
        frame_therm = normalize_thermal_frame(frame_therm)

        # ── Inference every N frames ──────────────────────────────
        if frame_count % YOLO_EVERY_N == 0:
            vis_input   = cv2.resize(frame_vis, (YOLO_IMGSZ, YOLO_IMGSZ))
            vis_results = vis_model.predict(
                source=vis_input, conf=YOLO_CONF, iou=YOLO_IOU,
                imgsz=YOLO_IMGSZ, verbose=False, device=0)

            therm_gray   = cv2.cvtColor(frame_therm, cv2.COLOR_BGR2GRAY)
            therm_rgb    = cv2.applyColorMap(therm_gray, THERMAL_COLORMAP)
            therm_input  = cv2.resize(therm_rgb, (YOLO_IMGSZ, YOLO_IMGSZ))
            therm_results = therm_model.predict(
                source=therm_input, conf=YOLO_CONF, iou=YOLO_IOU,
                imgsz=YOLO_IMGSZ, verbose=False, device=0)

        # ── Extract detections ────────────────────────────────────
        vis_dets   = []
        therm_dets = []

        if vis_results is not None:
            vis_dets = extract_detections(
                vis_results, YOLO_IMGSZ, YOLO_IMGSZ)

        if therm_results is not None:
            therm_dets = extract_detections(
                therm_results, YOLO_IMGSZ, YOLO_IMGSZ,
                offset_x = -offset_x / VIS_WIDTH  * YOLO_IMGSZ,
                offset_y = -offset_y / VIS_HEIGHT * YOLO_IMGSZ)

        # ── Fusion ────────────────────────────────────────────────
        fused_dets = decision_fusion(vis_dets, therm_dets)

        for d in fused_dets:
            if   d["source"] == SRC_FUSED:   stats["fused"]        += 1
            elif d["source"] == SRC_VISUAL:  stats["visual_only"]  += 1
            else:                            stats["thermal_only"]  += 1

        # ── Build display ─────────────────────────────────────────
        therm_gray  = cv2.cvtColor(frame_therm, cv2.COLOR_BGR2GRAY)
        therm_color = cv2.applyColorMap(therm_gray, THERMAL_COLORMAP)
        therm_big   = cv2.resize(therm_color, (VIS_WIDTH, VIS_HEIGHT),
                                 interpolation=cv2.INTER_LINEAR)
        fused_bg    = cv2.addWeighted(frame_vis, 1.0, therm_big, THERMAL_ALPHA, 0.0)
        display     = cv2.resize(fused_bg, (DISP_WIDTH, DISP_HEIGHT))

        display = draw_detections(display, fused_dets, DISP_WIDTH, DISP_HEIGHT)
        display = draw_legend(display)

        # FPS counter
        now = time.time()
        if now - fps_timer >= 1.0:
            fps        = fps_frames / (now - fps_timer)
            fps_frames = 0
            fps_timer  = now
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(WIN_MAIN, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('p'):
            print(f"\n[STATS] frames={stats['frames']} | "
                  f"fused={stats['fused']} | "
                  f"visual_only={stats['visual_only']} | "
                  f"thermal_only={stats['thermal_only']}")
            print(f"[OFFSET] x={offset_x}px, y={offset_y}px\n")

    vis_cap.release()
    therm_cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
