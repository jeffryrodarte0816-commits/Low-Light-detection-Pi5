# Models to run on each sensor
## Thermal Camera
Run the Instance segmentation model- use predict where mask is used to differentiate animals overlapping each other.   
## LIDAR 
Just be used for information based so nothing image based, will be used for autonomous driving and if necessary in conjunction with sensor that's necessary.  
Need to do: Talk to jeffry in regards to lidar and ros working together, along with lidar to be used with both cameras. Focus on the camera fusion, then try to apply lidar.
## Day/Night Camera
Pose estimation model and detection model only for day/night, possibly not run the Oriented Bounding boxes(OBB) model since it's more for top-view camera(maybe not applicable towards rover from ground).

# Decision-Level(Late) fusion for both cameras  
Purpose: To best get results via accuracy, We are going to run YOLO models independently on both cameras  
Step 1: Pre-process the Thermal Feed  
Step 2: The "Overlap" Logic (Intersection over Union)  

Scripts using:  
1.A calibration script to adjust the image of both sensors to overlap  
2. Fusion script to use both YOLO on cameras individually, merging detection and give singel output  

calibrate_offset.py  
```bash
"""
calibrate_offset.py
===================
Parallax Offset Calibration Tool
Measures the vertical (and horizontal) pixel offset between your
Arducam B0506 and TOPDON TC001 when mounted vertically on the rover.

This offset is saved to camera_offset.json and loaded automatically
by fusion_decision.py so boxes from each sensor align correctly.

HOW TO USE:
1. Place a clear object (e.g. a water bottle or person) at ~3 meters.
2. Run this script.
3. Two windows appear side by side: Visual (left) | Thermal (right).
4. Click the CENTER of the object in BOTH windows.
5. The offset is calculated and saved automatically.

Controls:
    C      : Clear current clicks and retry
    S      : Save current offset manually
    ESC    : Quit without saving

Usage:
    source /home/sunnysquad/venv/bin/activate
    python3 calibrate_offset.py
"""

import cv2
import numpy as np
import json
import os

OFFSET_FILE  = "/home/sunnysquad/yolo_project/camera_offset.json"

VIS_WIDTH    = 1920
VIS_HEIGHT   = 1080
THERM_WIDTH  = 256
THERM_HEIGHT = 192
DISP_W       = 640    # Display width per camera panel
DISP_H       = 480
THERMAL_COLORMAP = cv2.COLORMAP_INFERNO

# Click state
clicks = {"visual": None, "thermal": None}
active_window = None


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def auto_detect_cameras():
    print("🔍 Scanning cameras...")
    vis_cap = therm_cap = None
    vis_idx = therm_idx = -1
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"  /dev/video{i}: {int(w)}x{int(h)}")
        if w == 256 and h == 192 and therm_cap is None:
            print(f"    ✅ TOPDON TC001 → video{i}")
            therm_cap = cap
            therm_idx = i
        elif w > 640 and vis_cap is None:
            print(f"    ✅ Arducam B0506 → video{i}")
            vis_cap = cap
            vis_cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIS_WIDTH)
            vis_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIS_HEIGHT)
            vis_idx = i
        else:
            cap.release()
    return vis_cap, therm_cap, vis_idx, therm_idx


def mouse_visual(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale display coords back to full visual resolution
        real_x = int(x * VIS_WIDTH  / DISP_W)
        real_y = int(y * VIS_HEIGHT / DISP_H)
        clicks["visual"] = (real_x, real_y)
        print(f"  📍 Visual click  → ({real_x}, {real_y}) [full res]")
        check_and_save()


def mouse_thermal(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale display coords back to full thermal resolution
        real_x = int(x * THERM_WIDTH  / DISP_W)
        real_y = int(y * THERM_HEIGHT / DISP_H)
        clicks["thermal"] = (real_x, real_y)
        print(f"  📍 Thermal click → ({real_x}, {real_y}) [full res]")
        check_and_save()


def check_and_save():
    """If both clicks exist, compute and save offset."""
    if clicks["visual"] is None or clicks["thermal"] is None:
        return

    vx, vy = clicks["visual"]
    tx, ty = clicks["thermal"]

    # Scale thermal click to visual coordinate space for comparison
    tx_scaled = int(tx * VIS_WIDTH  / THERM_WIDTH)
    ty_scaled = int(ty * VIS_HEIGHT / THERM_HEIGHT)

    # Offset = how much to shift thermal detections to align with visual
    offset_x = vx - tx_scaled
    offset_y = vy - ty_scaled

    print(f"\n✅ OFFSET CALCULATED")
    print(f"   Visual  center: ({vx}, {vy})")
    print(f"   Thermal center: ({tx_scaled}, {ty_scaled}) [scaled to visual res]")
    print(f"   Offset X: {offset_x}px")
    print(f"   Offset Y: {offset_y}px")

    save_offset(offset_x, offset_y)


def save_offset(ox, oy):
    data = {
        "offset_x":    ox,
        "offset_y":    oy,
        "vis_width":   VIS_WIDTH,
        "vis_height":  VIS_HEIGHT,
        "therm_width":  THERM_WIDTH,
        "therm_height": THERM_HEIGHT,
        "note": "offset_x/y = pixels in visual-res space to shift thermal detections"
    }
    os.makedirs(os.path.dirname(OFFSET_FILE), exist_ok=True)
    with open(OFFSET_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved to: {OFFSET_FILE}")
    print(f"   You can now run fusion_decision.py\n")


def main():
    vis_cap, therm_cap, vis_idx, therm_idx = auto_detect_cameras()
    if vis_cap is None or therm_cap is None:
        print("❌ Could not detect both cameras.")
        return

    cv2.namedWindow("VISUAL  — Click object center", cv2.WINDOW_NORMAL)
    cv2.namedWindow("THERMAL — Click object center", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VISUAL  — Click object center", DISP_W, DISP_H)
    cv2.resizeWindow("THERMAL — Click object center", DISP_W, DISP_H)
    cv2.moveWindow("VISUAL  — Click object center",   20,       100)
    cv2.moveWindow("THERMAL — Click object center",   DISP_W + 60, 100)

    cv2.setMouseCallback("VISUAL  — Click object center", mouse_visual)
    cv2.setMouseCallback("THERMAL — Click object center", mouse_thermal)

    print("\n─────────────────────────────────────────────")
    print("  PARALLAX CALIBRATION")
    print("  1. Place object at ~3 meters from rover")
    print("  2. Click CENTER of object in VISUAL window")
    print("  3. Click CENTER of object in THERMAL window")
    print("  Offset is saved automatically.")
    print("  C = Clear clicks | ESC = Quit")
    print("─────────────────────────────────────────────\n")

    while True:
        ret_v, frame_vis   = vis_cap.read()
        ret_t, frame_therm = therm_cap.read()

        if not ret_v or not ret_t:
            continue

        # Visual display
        vis_disp = cv2.resize(frame_vis, (DISP_W, DISP_H))

        # Thermal display — apply colormap
        if len(frame_therm.shape) == 2:
            therm_disp = cv2.applyColorMap(frame_therm, THERMAL_COLORMAP)
        else:
            therm_disp = frame_therm
        therm_disp = cv2.resize(therm_disp, (DISP_W, DISP_H))

        # Draw clicks if set
        if clicks["visual"]:
            cx = int(clicks["visual"][0] * DISP_W / VIS_WIDTH)
            cy = int(clicks["visual"][1] * DISP_H / VIS_HEIGHT)
            cv2.drawMarker(vis_disp, (cx, cy), (0, 255, 0),
                           cv2.MARKER_CROSS, 30, 2)
            cv2.putText(vis_disp, "CLICKED", (cx + 15, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if clicks["thermal"]:
            cx = int(clicks["thermal"][0] * DISP_W / THERM_WIDTH)
            cy = int(clicks["thermal"][1] * DISP_H / THERM_HEIGHT)
            cv2.drawMarker(therm_disp, (cx, cy), (0, 255, 0),
                           cv2.MARKER_CROSS, 30, 2)
            cv2.putText(therm_disp, "CLICKED", (cx + 15, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Instructions overlay
        instructions = [
            "1. Click center of object",
            "2. Do same in other window",
            "Offset saves automatically",
            "C = Clear | ESC = Quit"
        ]
        for i, line in enumerate(instructions):
            cv2.putText(vis_disp, line, (10, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

        # Status
        vstatus = "✓ CLICKED" if clicks["visual"]  else "— click here"
        tstatus = "✓ CLICKED" if clicks["thermal"] else "— click here"
        cv2.putText(vis_disp,   f"Visual: {vstatus}",
                    (10, DISP_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if clicks["visual"]  else (0,200,255), 2)
        cv2.putText(therm_disp, f"Thermal: {tstatus}",
                    (10, DISP_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if clicks["thermal"] else (0,200,255), 2)

        cv2.imshow("VISUAL  — Click object center", vis_disp)
        cv2.imshow("THERMAL — Click object center", therm_disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('c'):
            clicks["visual"]  = None
            clicks["thermal"] = None
            print("🔄 Clicks cleared. Click again to recalibrate.")
        elif key == ord('s') and clicks["visual"] and clicks["thermal"]:
            check_and_save()

    vis_cap.release()
    therm_cap.release()
    cv2.destroyAllWindows()
    print("✅ Calibration tool closed.")


if __name__ == "__main__":
    main()

```

fusion_detection.py  
```bash
"""
fusion_decision.py
==================
Decision-Level (Late) Fusion Pipeline for Jetson Orin Nano
- Arducam B0506  → YOLO visual inference
- TOPDON TC001   → YOLO thermal inference
- Fusion Layer   → Merges detections via IoU + confidence voting
- Output         → Single fused display with color-coded detection sources

Detection Box Colors:
    GREEN   = Confirmed by BOTH sensors (highest confidence — fused)
    BLUE    = Visual only  (Arducam)
    RED     = Thermal only (TC001)

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
# ^ Point THERMAL_MODEL_PATH to a thermally-trained model if you have one.
#   If not, the visual model on thermal frames will still detect heat blobs
#   as shapes — useful as a fallback.

OFFSET_FILE = "/home/sunnysquad/yolo_project/camera_offset.json"
# Saved by calibrate_offset.py. If not found, offset defaults to (0, 0).

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
YOLO_CONF    = 0.35   # Lower threshold — fusion layer handles false positives
YOLO_IOU     = 0.45
YOLO_IMGSZ   = 640
YOLO_EVERY_N = 2      # Run inference every N frames (raise if FPS drops)

# ─────────────────────────────────────────────────────────────────
# FUSION SETTINGS
# ─────────────────────────────────────────────────────────────────
FUSION_IOU_THRESHOLD  = 0.3   # Minimum IoU to consider two boxes as "same object"
FUSION_CONF_BOOST     = 0.1   # Confidence bonus applied to dual-confirmed detections
THERMAL_COLORMAP      = cv2.COLORMAP_INFERNO
THERMAL_ALPHA         = 0.35  # Thermal overlay transparency on fused display

# Detection source labels
SRC_VISUAL   = "V"
SRC_THERMAL  = "T"
SRC_FUSED    = "V+T"

# Colors (BGR)
COLOR_VISUAL   = (255, 180,   0)   # Blue
COLOR_THERMAL  = (0,    80, 255)   # Red
COLOR_FUSED    = (0,   220,   0)   # Green


# ─────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────

def load_offset(path):
    """Load saved parallax offset from calibrate_offset.py."""
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        ox, oy = data.get("offset_x", 0), data.get("offset_y", 0)
        print(f"📐 Loaded camera offset: x={ox}px, y={oy}px (from {path})")
        return ox, oy
    print(f"⚠️  No offset file found at {path}. Defaulting to (0, 0).")
    print(f"   Run calibrate_offset.py to measure parallax.")
    return 0, 0


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def auto_detect_cameras():
    print("🔍 Scanning cameras...")
    vis_cap = therm_cap = None
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"  /dev/video{i}: {int(w)}x{int(h)}")
        if w == 256 and h == 192 and therm_cap is None:
            print(f"    ✅ TOPDON TC001 → video{i}")
            therm_cap = cap
        elif w > 640 and vis_cap is None:
            print(f"    ✅ Arducam B0506 → video{i}")
            vis_cap = cap
            vis_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  VIS_WIDTH)
            vis_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIS_HEIGHT)
        else:
            cap.release()
    return vis_cap, therm_cap


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union between two boxes.
    Format: [x1, y1, x2, y2] in normalized [0,1] coords.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter  = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union  = areaA + areaB - inter

    return inter / union if union > 0 else 0.0


def extract_detections(results, frame_w, frame_h, offset_x=0, offset_y=0):
    """
    Extract detections from YOLO results as normalized [0,1] boxes.
    offset_x/y: shift applied to correct parallax (in pixels, at frame resolution).
    Returns list of dicts: {box_norm, conf, cls, label}
    """
    dets = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # Apply parallax offset correction
            x1 += offset_x; x2 += offset_x
            y1 += offset_y; y2 += offset_y
            # Clamp to frame bounds
            x1 = max(0, min(x1, frame_w))
            x2 = max(0, min(x2, frame_w))
            y1 = max(0, min(y1, frame_h))
            y2 = max(0, min(y2, frame_h))
            # Normalize
            dets.append({
                "box_norm": [x1/frame_w, y1/frame_h, x2/frame_w, y2/frame_h],
                "conf":     float(box.conf[0]),
                "cls":      int(box.cls[0]),
                "label":    result.names[int(box.cls[0])]
            })
    return dets


def decision_fusion(vis_dets, therm_dets):
    """
    Core fusion logic.
    
    Strategy:
    1. For each visual detection, search for a thermal detection with IoU > threshold.
    2. If found → FUSED detection (avg box, boosted confidence, marked V+T).
    3. Unmatched visual → keep as visual-only (lower display priority).
    4. Unmatched thermal → keep as thermal-only (lower display priority).
    
    Returns list of fused detections with source labels.
    """
    fused    = []
    matched_therm = set()

    for v in vis_dets:
        best_iou  = 0
        best_tidx = -1
        for tidx, t in enumerate(therm_dets):
            if tidx in matched_therm:
                continue
            # Only match same class (or allow cross-class if thermal model differs)
            if t["cls"] != v["cls"]:
                continue
            iou = compute_iou(v["box_norm"], t["box_norm"])
            if iou > best_iou:
                best_iou  = iou
                best_tidx = tidx

        if best_iou >= FUSION_IOU_THRESHOLD and best_tidx >= 0:
            # ── DUAL CONFIRMATION ───────────────────────────────
            t = therm_dets[best_tidx]
            matched_therm.add(best_tidx)

            # Average the two boxes (weighted by confidence)
            vw = v["conf"]
            tw = t["conf"]
            total = vw + tw
            avg_box = [
                (v["box_norm"][i] * vw + t["box_norm"][i] * tw) / total
                for i in range(4)
            ]
            fused_conf = min(1.0, max(v["conf"], t["conf"]) + FUSION_CONF_BOOST)

            fused.append({
                "box_norm": avg_box,
                "conf":     fused_conf,
                "cls":      v["cls"],
                "label":    v["label"],
                "source":   SRC_FUSED,
                "iou":      best_iou
            })
        else:
            # ── VISUAL ONLY ─────────────────────────────────────
            fused.append({**v, "source": SRC_VISUAL, "iou": 0})

    # ── THERMAL ONLY (unmatched) ─────────────────────────────────
    for tidx, t in enumerate(therm_dets):
        if tidx not in matched_therm:
            fused.append({**t, "source": SRC_THERMAL, "iou": 0})

    return fused


def draw_detections(frame, detections, disp_w, disp_h):
    """Draw fused detections on display frame with source-coded colors."""
    for det in detections:
        x1n, y1n, x2n, y2n = det["box_norm"]
        x1 = int(x1n * disp_w); y1 = int(y1n * disp_h)
        x2 = int(x2n * disp_w); y2 = int(y2n * disp_h)

        src    = det["source"]
        conf   = det["conf"]
        label  = det["label"]
        color  = COLOR_FUSED if src == SRC_FUSED else \
                 COLOR_VISUAL if src == SRC_VISUAL else COLOR_THERMAL

        # Box (thicker for fused)
        thickness = 3 if src == SRC_FUSED else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label
        tag = f"[{src}] {label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return frame


def draw_legend(frame):
    """Draw source legend in top-right corner."""
    items = [
        ("V+T  Fused",    COLOR_FUSED),
        ("V    Visual",   COLOR_VISUAL),
        ("T    Thermal",  COLOR_THERMAL),
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
    # Load parallax offset
    offset_x, offset_y = load_offset(OFFSET_FILE)

    # Load models
    print(f"\n🤖 Loading visual YOLO model...")
    vis_model   = YOLO(VISUAL_MODEL_PATH)
    print(f"🤖 Loading thermal YOLO model...")
    therm_model = YOLO(THERMAL_MODEL_PATH)
    print("   ✅ Models loaded.\n")

    # Open cameras
    vis_cap, therm_cap = auto_detect_cameras()
    if vis_cap is None or therm_cap is None:
        print("\n❌ Could not identify both cameras. Check USB connections.")
        print("   Run: v4l2-ctl --list-devices")
        return

    print("\n─────────────────────────────────────────────")
    print("  DECISION-LEVEL FUSION ACTIVE")
    print("  GREEN  = Both sensors agree (V+T fused)")
    print("  BLUE   = Visual only")
    print("  RED    = Thermal only")
    print("  P      = Print stats | ESC = Quit")
    print("─────────────────────────────────────────────\n")

    frame_count   = 0
    vis_results   = None
    therm_results = None
    fps_timer     = time.time()
    fps           = 0.0

    # Stats counters
    stats = {"fused": 0, "visual_only": 0, "thermal_only": 0, "frames": 0}

    while True:
        ret_v, frame_vis   = vis_cap.read()
        ret_t, frame_therm = therm_cap.read()

        if not ret_v or not ret_t:
            print("⚠️  Frame read error — retrying...")
            time.sleep(0.05)
            continue

        frame_count    += 1
        stats["frames"] = frame_count

        # ── Inference every N frames ──────────────────────────────
        if frame_count % YOLO_EVERY_N == 0:
            # Visual inference (Arducam)
            vis_input    = cv2.resize(frame_vis, (YOLO_IMGSZ, YOLO_IMGSZ))
            vis_results  = vis_model.predict(
                source=vis_input, conf=YOLO_CONF, iou=YOLO_IOU,
                imgsz=YOLO_IMGSZ, verbose=False, device=0)

            # Thermal inference (TC001 — apply colormap first so model sees RGB)
            if len(frame_therm.shape) == 2:
                therm_rgb = cv2.applyColorMap(frame_therm, THERMAL_COLORMAP)
            else:
                therm_rgb = frame_therm
            therm_input   = cv2.resize(therm_rgb, (YOLO_IMGSZ, YOLO_IMGSZ))
            therm_results = therm_model.predict(
                source=therm_input, conf=YOLO_CONF, iou=YOLO_IOU,
                imgsz=YOLO_IMGSZ, verbose=False, device=0)

        # ── Extract detections ────────────────────────────────────
        vis_dets = []
        therm_dets_raw = []

        if vis_results is not None:
            vis_dets = extract_detections(
                vis_results, YOLO_IMGSZ, YOLO_IMGSZ)

        if therm_results is not None:
            # Apply inverse offset to thermal boxes to align into visual space
            therm_dets_raw = extract_detections(
                therm_results, YOLO_IMGSZ, YOLO_IMGSZ,
                offset_x = -offset_x / VIS_WIDTH  * YOLO_IMGSZ,
                offset_y = -offset_y / VIS_HEIGHT * YOLO_IMGSZ)

        # ── Decision-Level Fusion ─────────────────────────────────
        fused_dets = decision_fusion(vis_dets, therm_dets_raw)

        # Update stats
        for d in fused_dets:
            if d["source"] == SRC_FUSED:   stats["fused"]        += 1
            elif d["source"] == SRC_VISUAL: stats["visual_only"]  += 1
            else:                           stats["thermal_only"]  += 1

        # ── Build display frame ───────────────────────────────────
        # Overlay thermal on visual for context, then draw boxes on top
        if len(frame_therm.shape) == 2:
            therm_color = cv2.applyColorMap(frame_therm, THERMAL_COLORMAP)
        else:
            therm_color = frame_therm

        therm_big = cv2.resize(therm_color, (VIS_WIDTH, VIS_HEIGHT),
                               interpolation=cv2.INTER_LINEAR)
        fused_bg  = cv2.addWeighted(frame_vis, 1.0, therm_big, THERMAL_ALPHA, 0.0)
        display   = cv2.resize(fused_bg, (DISP_WIDTH, DISP_HEIGHT))

        # Draw detections
        display = draw_detections(display, fused_dets, DISP_WIDTH, DISP_HEIGHT)
        display = draw_legend(display)

        # FPS
        now = time.time()
        if now - fps_timer >= 1.0:
            fps         = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer   = now
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Decision Fusion | Arducam + TC001 + YOLO", display)

        # ── Controls ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('p'):
            print(f"\n📊 STATS → frames={stats['frames']} | "
                  f"fused={stats['fused']} | "
                  f"visual_only={stats['visual_only']} | "
                  f"thermal_only={stats['thermal_only']}")
            print(f"📐 OFFSET → x={offset_x}px, y={offset_y}px\n")

    vis_cap.release()
    therm_cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Done.")


if __name__ == "__main__":
    main()

```
