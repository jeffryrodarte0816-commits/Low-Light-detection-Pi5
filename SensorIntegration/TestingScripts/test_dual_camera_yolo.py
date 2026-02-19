"""
dual_camera_yolo.py
===================
Runs YOLO independently on both cameras simultaneously.
No sensor fusion — each camera gets its own window and its own inference.

Windows:
    "YOLO | Arducam B0506"   — visual camera detections
    "YOLO | TOPDON TC001"    — thermal camera detections (with colormap)

Controls:
    P   : Print FPS + detection counts per camera
    ESC : Quit

Usage:
    source /home/sunnysquad/venv/bin/activate
    python3 dual_camera_yolo.py

Fixes applied vs original:
    1. auto_detect_cameras: Arducam check changed from w > 640 → w >= 640
       (default reported width is exactly 640, so the old check silently
       dropped it into the else-branch and released the capture).
    2. Arducam target resolution changed to 640×480 @ 15 fps (MJPG).
       Requesting 1920×1080 caused the V4L2 driver to reject the mode,
       resulting in black / no frames.
    3. VIS_WIDTH / VIS_HEIGHT constants updated to match 640×480.
    4. DISP_WIDTH / DISP_HEIGHT kept at 640×480 for the visual window
       (no upscaling needed) and 256×192 for thermal, side-by-side layout
       adjusted accordingly.
"""

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────
# MODEL PATH
# ─────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/sunnysquad/yolo_project/ultralytics/runs/detect/test_camera/weights/best.pt"

# ─────────────────────────────────────────────────────────────────
# HARDWARE
# ─────────────────────────────────────────────────────────────────
# FIX #2: Arducam runs at 640×480 @ 15 fps — do NOT request 1920×1080
VIS_WIDTH    = 640
VIS_HEIGHT   = 480
VIS_FPS      = 15

THERM_WIDTH  = 256
THERM_HEIGHT = 192

# Display window sizes (each camera gets its own window)
VIS_DISP_WIDTH   = 640
VIS_DISP_HEIGHT  = 480
THERM_DISP_WIDTH = 512   # upscale thermal 2× so it's easier to see
THERM_DISP_HEIGHT = 384

# ─────────────────────────────────────────────────────────────────
# YOLO SETTINGS
# ─────────────────────────────────────────────────────────────────
YOLO_CONF    = 0.4
YOLO_IOU     = 0.45
YOLO_IMGSZ   = 640
YOLO_EVERY_N = 2      # Run inference every N frames

# ─────────────────────────────────────────────────────────────────
# THERMAL COLORMAP
# Applied before YOLO so the model sees RGB-like input from TC001
# Options: cv2.COLORMAP_INFERNO, COLORMAP_JET, COLORMAP_HOT
# ─────────────────────────────────────────────────────────────────
THERMAL_COLORMAP = cv2.COLORMAP_INFERNO

# Detection box color (BGR)
BOX_COLOR   = (0, 255, 0)
LABEL_COLOR = (0, 0, 0)


# ─────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────

def open_arducam(index=0):
    """
    Open Arducam B0506 at /dev/video<index> and configure it for
    640×480 @ 15 fps with MJPG so the driver accepts the mode.
    Returns the configured VideoCapture, or None on failure.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    # Set MJPG first — required before changing resolution on most UVC devices
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  VIS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIS_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, VIS_FPS)
    # Verify
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"    Arducam configured: {int(w)}x{int(h)} @ {fps:.0f}fps")
    return cap


def auto_detect_cameras():
    """
    Scan /dev/video0..5 and identify Arducam vs TOPDON by resolution.

    FIX #1: Changed `w > 640` to `w >= 640` for the Arducam branch.
    The Arducam's default reported width at open-time is exactly 640,
    which the old strict `> 640` check silently rejected, releasing
    the capture and leaving vis_cap as None.
    """
    print("🔍 Scanning cameras...")
    vis_cap   = None
    therm_cap = None

    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"  /dev/video{i}: {int(w)}x{int(h)}")

        if w == 256 and h == 192 and therm_cap is None:
            print(f"    ✅ TOPDON TC001  → video{i}")
            therm_cap = cap

        # FIX #1: was `w > 640`, must be `w >= 640`
        elif w >= 640 and vis_cap is None:
            print(f"    ✅ Arducam B0506 → video{i}")
            cap.release()  # release the probe cap, reopen with proper settings
            vis_cap = open_arducam(i)
            if vis_cap is None:
                print(f"    ⚠️  Failed to re-open Arducam at video{i}")
        else:
            cap.release()

    # Hard fallback: if scan missed the Arducam (e.g. default res wasn't
    # reported correctly), try video0 directly since you confirmed it's there.
    if vis_cap is None:
        print("  ⚠️  Scan didn't catch Arducam — forcing video0...")
        vis_cap = open_arducam(0)
        if vis_cap is not None:
            print("    ✅ Arducam B0506 → video0 (forced)")

    return vis_cap, therm_cap


def draw_detections(frame, results, disp_w, disp_h, infer_w, infer_h):
    """
    Draw YOLO boxes on the display frame.
    Scales coords from inference resolution → display resolution.
    """
    scale_x = disp_w / infer_w
    scale_y = disp_h / infer_h

    count = 0
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)

            conf  = float(box.conf[0])
            label = result.names[int(box.cls[0])]
            tag   = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), BOX_COLOR, -1)
            cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, LABEL_COLOR, 2)
            count += 1

    return frame, count


def draw_hud(frame, label, fps, det_count):
    """Draw camera label, FPS, and detection count on frame."""
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Detections: {det_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


# ─────────────────────────────────────────────────────────────────
# CAMERA WORKER THREAD
# Each camera runs in its own thread so neither blocks the other
# ─────────────────────────────────────────────────────────────────

class CameraWorker(threading.Thread):
    def __init__(self, cap, model, window_name, is_thermal=False,
                 cam_width=640, cam_height=480,
                 disp_width=640, disp_height=480):
        super().__init__(daemon=True)
        self.cap         = cap
        self.model       = model
        self.window_name = window_name
        self.is_thermal  = is_thermal
        self.cam_width   = cam_width
        self.cam_height  = cam_height
        self.disp_width  = disp_width
        self.disp_height = disp_height

        # Shared state for main thread to read
        self.latest_frame  = None
        self.det_count     = 0
        self.fps           = 0.0
        self.lock          = threading.Lock()
        self.running       = True

    def stop(self):
        self.running = False

    def run(self):
        frame_count = 0
        results     = None
        fps_timer   = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1

            # ── Thermal colormap ──────────────────────────────────
            if self.is_thermal:
                if len(frame.shape) == 2:
                    frame = cv2.applyColorMap(frame, THERMAL_COLORMAP)
                else:
                    frame = cv2.applyColorMap(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), THERMAL_COLORMAP)

            # ── YOLO inference ────────────────────────────────────
            if frame_count % YOLO_EVERY_N == 0:
                yolo_input = cv2.resize(frame, (YOLO_IMGSZ, YOLO_IMGSZ))
                results    = self.model.predict(
                    source  = yolo_input,
                    conf    = YOLO_CONF,
                    iou     = YOLO_IOU,
                    imgsz   = YOLO_IMGSZ,
                    verbose = False,
                    device  = 0
                )

            # ── Build display frame ───────────────────────────────
            display = cv2.resize(frame, (self.disp_width, self.disp_height))
            count   = 0

            if results is not None:
                display, count = draw_detections(
                    display, results,
                    self.disp_width, self.disp_height,
                    YOLO_IMGSZ, YOLO_IMGSZ
                )

            # ── FPS (computed over the last second) ───────────────
            now = time.time()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps_val     = frame_count / elapsed
                frame_count = 0
                fps_timer   = now
                with self.lock:
                    self.fps = fps_val

            with self.lock:
                self.det_count    = count
                self.latest_frame = display.copy()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    # ── Camera discovery ──────────────────────────────────────────
    vis_cap, therm_cap = auto_detect_cameras()
    if vis_cap is None or therm_cap is None:
        missing = []
        if vis_cap   is None: missing.append("Arducam B0506")
        if therm_cap is None: missing.append("TOPDON TC001")
        print(f"\n❌ Could not open: {', '.join(missing)}")
        print("   Check USB connections or run: v4l2-ctl --list-devices")
        if vis_cap:   vis_cap.release()
        if therm_cap: therm_cap.release()
        return

    # ── Load models (one per camera to avoid sharing state) ───────
    print(f"\n🤖 Loading YOLO models (x2)...")
    vis_model   = YOLO(MODEL_PATH)
    therm_model = YOLO(MODEL_PATH)
    print("   ✅ Models loaded.\n")

    # ── Create windows ────────────────────────────────────────────
    cv2.namedWindow("YOLO | Arducam B0506", cv2.WINDOW_NORMAL)
    cv2.namedWindow("YOLO | TOPDON TC001",  cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO | Arducam B0506", VIS_DISP_WIDTH,   VIS_DISP_HEIGHT)
    cv2.resizeWindow("YOLO | TOPDON TC001",  THERM_DISP_WIDTH, THERM_DISP_HEIGHT)
    # Position windows side by side
    cv2.moveWindow("YOLO | Arducam B0506",  20,                    100)
    cv2.moveWindow("YOLO | TOPDON TC001",   VIS_DISP_WIDTH + 60,   100)

    # ── Start camera worker threads ───────────────────────────────
    vis_worker = CameraWorker(
        cap         = vis_cap,
        model       = vis_model,
        window_name = "YOLO | Arducam B0506",
        is_thermal  = False,
        cam_width   = VIS_WIDTH,
        cam_height  = VIS_HEIGHT,
        disp_width  = VIS_DISP_WIDTH,
        disp_height = VIS_DISP_HEIGHT,
    )
    therm_worker = CameraWorker(
        cap         = therm_cap,
        model       = therm_model,
        window_name = "YOLO | TOPDON TC001",
        is_thermal  = True,
        cam_width   = THERM_WIDTH,
        cam_height  = THERM_HEIGHT,
        disp_width  = THERM_DISP_WIDTH,
        disp_height = THERM_DISP_HEIGHT,
    )

    vis_worker.start()
    therm_worker.start()

    print("─────────────────────────────────────────────")
    print("  DUAL CAMERA YOLO RUNNING")
    print(f"  Left window  → Arducam B0506 ({VIS_WIDTH}×{VIS_HEIGHT} @ {VIS_FPS}fps)")
    print(f"  Right window → TOPDON TC001  ({THERM_WIDTH}×{THERM_HEIGHT})")
    print("  P = Print stats | ESC = Quit")
    print("─────────────────────────────────────────────\n")

    # ── Main display loop (runs on main thread for cv2.imshow) ────
    while True:
        with vis_worker.lock:
            vis_frame = vis_worker.latest_frame
            vis_fps   = vis_worker.fps
            vis_dets  = vis_worker.det_count

        with therm_worker.lock:
            therm_frame = therm_worker.latest_frame
            therm_fps   = therm_worker.fps
            therm_dets  = therm_worker.det_count

        if vis_frame is not None:
            vis_display = draw_hud(vis_frame.copy(), "Arducam B0506", vis_fps, vis_dets)
            cv2.imshow("YOLO | Arducam B0506", vis_display)

        if therm_frame is not None:
            therm_display = draw_hud(therm_frame.copy(), "TOPDON TC001", therm_fps, therm_dets)
            cv2.imshow("YOLO | TOPDON TC001", therm_display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        elif key == ord('p'):
            print(f"\n📊 Arducam  → FPS: {vis_fps:.1f}  | Detections: {vis_dets}")
            print(f"📊 TC001    → FPS: {therm_fps:.1f}  | Detections: {therm_dets}\n")

    # ── Cleanup ───────────────────────────────────────────────────
    print("\nShutting down...")
    vis_worker.stop()
    therm_worker.stop()
    vis_worker.join(timeout=2)
    therm_worker.join(timeout=2)
    vis_cap.release()
    therm_cap.release()
    cv2.destroyAllWindows()
    print("✅ Done.")


if __name__ == "__main__":
    main()
