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
