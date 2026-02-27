"""
calibrate_offset.py
===================
Parallax Offset Calibration Tool
Measures the vertical (and horizontal) pixel offset between your
Arducam B0506 and TOPDON TC001 when mounted vertically on the rover.

FIXES vs previous version:
  - Window names use ASCII only (em-dash caused Qt null-pointer crash)
  - TC001 256x384 YUYV handled: crops to top 192 rows after conversion
  - Arducam detected at 640x480 default (not just 1920x1080)

HOW TO USE:
  1. Place a clear object (e.g. water bottle) at ~3 meters from rover.
  2. Run this script.
  3. Two windows appear: Visual (left) | Thermal (right).
  4. Click the CENTER of the same object in BOTH windows.
  5. Offset is calculated and saved automatically.

Controls:
    C   : Clear clicks and retry
    S   : Save current offset manually
    ESC : Quit without saving

Usage:
    source /home/sunnysquad/venv/bin/activate
    python3 calibrate_offset.py
"""

import cv2
import numpy as np
import json
import os

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
OFFSET_FILE  = "/home/sunnysquad/yolo_project/camera_offset.json"

VIS_WIDTH    = 1920
VIS_HEIGHT   = 1080
THERM_WIDTH  = 256
THERM_HEIGHT = 192
DISP_W       = 640
DISP_H       = 480
THERMAL_COLORMAP = cv2.COLORMAP_INFERNO

# ── FIX 1: pure ASCII window names — no em-dash, no unicode ──────
WIN_VISUAL  = "VISUAL  - Click object center"
WIN_THERMAL = "THERMAL - Click object center"

# Click state
clicks = {"visual": None, "thermal": None}


# ─────────────────────────────────────────────────────────────────
# CAMERA DETECTION
# ─────────────────────────────────────────────────────────────────
def auto_detect_cameras():
    print("Scanning cameras...")
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

        # Arducam: width >= 640 (catches 640x480 default AND 1920x1080)
        elif w >= 640 and vis_cap is None:
            print(f"    [OK] Arducam B0506 at video{i}")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  VIS_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIS_HEIGHT)
            vis_cap = cap
            vis_idx = i
        else:
            cap.release()

    return vis_cap, therm_cap, vis_idx, therm_idx


# ─────────────────────────────────────────────────────────────────
# FIX 2: TC001 FRAME NORMALISATION
# Handles YUYV 256x192 and the quirky 256x384 double-height variant
# ─────────────────────────────────────────────────────────────────
def normalize_thermal_frame(frame):
    """
    Convert TC001 raw frame to a clean BGR 256x192 image.
    - h==384: crop top 192 rows (real thermal data is in top half)
    - Single channel: convert to BGR
    - Returns BGR image ready for colormap application
    """
    h, w = frame.shape[:2]

    # Double-height quirk: crop to top 192 rows
    if h == 384:
        frame = frame[:192, :]

    # Single-channel (true grayscale from YUYV Y-plane)
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Already 3-channel BGR (OpenCV auto-converted YUYV)
    return frame


# ─────────────────────────────────────────────────────────────────
# MOUSE CALLBACKS
# ─────────────────────────────────────────────────────────────────
def mouse_visual(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        real_x = int(x * VIS_WIDTH  / DISP_W)
        real_y = int(y * VIS_HEIGHT / DISP_H)
        clicks["visual"] = (real_x, real_y)
        print(f"  Visual click  -> ({real_x}, {real_y}) [full res]")
        check_and_save()


def mouse_thermal(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        real_x = int(x * THERM_WIDTH  / DISP_W)
        real_y = int(y * THERM_HEIGHT / DISP_H)
        clicks["thermal"] = (real_x, real_y)
        print(f"  Thermal click -> ({real_x}, {real_y}) [full res]")
        check_and_save()


# ─────────────────────────────────────────────────────────────────
# OFFSET CALCULATION & SAVE
# ─────────────────────────────────────────────────────────────────
def check_and_save():
    if clicks["visual"] is None or clicks["thermal"] is None:
        return

    vx, vy = clicks["visual"]
    tx, ty = clicks["thermal"]

    # Scale thermal click into visual coordinate space
    tx_scaled = int(tx * VIS_WIDTH  / THERM_WIDTH)
    ty_scaled = int(ty * VIS_HEIGHT / THERM_HEIGHT)

    offset_x = vx - tx_scaled
    offset_y = vy - ty_scaled

    print(f"\n[OFFSET CALCULATED]")
    print(f"  Visual  center: ({vx}, {vy})")
    print(f"  Thermal center: ({tx_scaled}, {ty_scaled}) [scaled to visual res]")
    print(f"  Offset X: {offset_x}px")
    print(f"  Offset Y: {offset_y}px")

    save_offset(offset_x, offset_y)


def save_offset(ox, oy):
    data = {
        "offset_x":     ox,
        "offset_y":     oy,
        "vis_width":    VIS_WIDTH,
        "vis_height":   VIS_HEIGHT,
        "therm_width":  THERM_WIDTH,
        "therm_height": THERM_HEIGHT,
        "note": "offset_x/y = pixels in visual-res space to shift thermal detections"
    }
    os.makedirs(os.path.dirname(OFFSET_FILE), exist_ok=True)
    with open(OFFSET_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[SAVED] -> {OFFSET_FILE}")
    print(f"  You can now run fusion_decision.py\n")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    vis_cap, therm_cap, vis_idx, therm_idx = auto_detect_cameras()

    if vis_cap is None:
        print("[ERROR] Arducam B0506 not found. Check USB connection.")
        return
    if therm_cap is None:
        print("[ERROR] TOPDON TC001 not found. Check USB connection.")
        return

    # ── Create windows FIRST, then set callbacks ─────────────────
    # Window names are pure ASCII to avoid Qt null-pointer crash
    cv2.namedWindow(WIN_VISUAL,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_THERMAL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_VISUAL,  DISP_W, DISP_H)
    cv2.resizeWindow(WIN_THERMAL, DISP_W, DISP_H)
    cv2.moveWindow(WIN_VISUAL,   20,          100)
    cv2.moveWindow(WIN_THERMAL,  DISP_W + 60, 100)

    # ── Set mouse callbacks AFTER windows are confirmed open ──────
    cv2.setMouseCallback(WIN_VISUAL,  mouse_visual)
    cv2.setMouseCallback(WIN_THERMAL, mouse_thermal)

    print("\n" + "-"*50)
    print("  PARALLAX CALIBRATION")
    print("  1. Place object at ~3 meters from rover")
    print("  2. Click CENTER of object in VISUAL window")
    print("  3. Click CENTER of object in THERMAL window")
    print("  Offset saves automatically.")
    print("  C = Clear clicks | S = Save manually | ESC = Quit")
    print("-"*50 + "\n")

    while True:
        ret_v, frame_vis   = vis_cap.read()
        ret_t, frame_therm = therm_cap.read()

        if not ret_v or not ret_t:
            continue

        # Normalise thermal frame (handles 256x384 quirk + gray->BGR)
        frame_therm = normalize_thermal_frame(frame_therm)

        # ── Visual display ────────────────────────────────────────
        vis_disp = cv2.resize(frame_vis, (DISP_W, DISP_H))

        # ── Thermal display ───────────────────────────────────────
        therm_gray = cv2.cvtColor(frame_therm, cv2.COLOR_BGR2GRAY)
        therm_disp = cv2.applyColorMap(therm_gray, THERMAL_COLORMAP)
        therm_disp = cv2.resize(therm_disp, (DISP_W, DISP_H))

        # ── Draw click markers ────────────────────────────────────
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

        # ── Instructions overlay ──────────────────────────────────
        instructions = [
            "1. Click center of object in each window",
            "2. Offset saves automatically",
            "C = Clear  |  S = Save  |  ESC = Quit"
        ]
        for i, line in enumerate(instructions):
            cv2.putText(vis_disp, line, (10, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

        # ── Status bar ────────────────────────────────────────────
        vstatus = "[CLICKED]" if clicks["visual"]  else "click here"
        tstatus = "[CLICKED]" if clicks["thermal"] else "click here"
        vc = (0, 255, 0) if clicks["visual"]  else (0, 200, 255)
        tc = (0, 255, 0) if clicks["thermal"] else (0, 200, 255)
        cv2.putText(vis_disp,   f"Visual: {vstatus}",
                    (10, DISP_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vc, 2)
        cv2.putText(therm_disp, f"Thermal: {tstatus}",
                    (10, DISP_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tc, 2)

        cv2.imshow(WIN_VISUAL,  vis_disp)
        cv2.imshow(WIN_THERMAL, therm_disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:    # ESC
            break
        elif key == ord('c'):
            clicks["visual"]  = None
            clicks["thermal"] = None
            print("Clicks cleared. Click again to recalibrate.")
        elif key == ord('s'):
            if clicks["visual"] and clicks["thermal"]:
                check_and_save()
            else:
                print("Need both clicks before saving.")

    vis_cap.release()
    therm_cap.release()
    cv2.destroyAllWindows()
    print("Calibration tool closed.")


if __name__ == "__main__":
    main()
