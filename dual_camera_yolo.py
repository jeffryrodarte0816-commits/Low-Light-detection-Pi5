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
VIS_WIDTH    = 1920
VIS_HEIGHT   = 1080
THERM_WIDTH  = 256
THERM_HEIGHT = 192

DISP_WIDTH   = 960    # Each window width
DISP_HEIGHT  = 540    # Each window height

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

def auto_detect_cameras():
    """Scan /dev/video0..5 and identify Arducam vs TOPDON by resolution."""
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
        elif w > 640 and vis_cap is None:
            print(f"    ✅ Arducam B0506 → video{i}")
            vis_cap = cap
            vis_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  VIS_WIDTH)
            vis_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIS_HEIGHT)
            vis_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        else:
            cap.release()

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
                 cam_width=1920, cam_height=1080):
        super().__init__(daemon=True)
        self.cap         = cap
        self.model       = model
        self.window_name = window_name
        self.is_thermal  = is_thermal
        self.cam_width   = cam_width
        self.cam_height  = cam_height

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
            display = cv2.resize(frame, (DISP_WIDTH, DISP_HEIGHT))
            count   = 0

            if results is not None:
                display, count = draw_detections(
                    display, results,
                    DISP_WIDTH, DISP_HEIGHT,
                    YOLO_IMGSZ, YOLO_IMGSZ
                )

            # ── FPS ───────────────────────────────────────────────
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_val      = frame_count / (now - fps_timer)
                frame_count  = 0
                fps_timer    = now
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
        print("\n❌ Could not identify both cameras.")
        print("   Check USB connections or run: v4l2-ctl --list-devices")
        return

    # ── Load models (one per camera to avoid sharing state) ───────
    print(f"\n🤖 Loading YOLO models (x2)...")
    vis_model   = YOLO(MODEL_PATH)
    therm_model = YOLO(MODEL_PATH)
    print("   ✅ Models loaded.\n")

    # ── Create windows ────────────────────────────────────────────
    cv2.namedWindow("YOLO | Arducam B0506",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("YOLO | TOPDON TC001",   cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO | Arducam B0506", DISP_WIDTH, DISP_HEIGHT)
    cv2.resizeWindow("YOLO | TOPDON TC001",  DISP_WIDTH, DISP_HEIGHT)
    # Position windows side by side
    cv2.moveWindow("YOLO | Arducam B0506",  20,             100)
    cv2.moveWindow("YOLO | TOPDON TC001",   DISP_WIDTH + 60, 100)

    # ── Start camera worker threads ───────────────────────────────
    vis_worker = CameraWorker(
        cap         = vis_cap,
        model       = vis_model,
        window_name = "YOLO | Arducam B0506",
        is_thermal  = False,
        cam_width   = VIS_WIDTH,
        cam_height  = VIS_HEIGHT
    )
    therm_worker = CameraWorker(
        cap         = therm_cap,
        model       = therm_model,
        window_name = "YOLO | TOPDON TC001",
        is_thermal  = True,
        cam_width   = THERM_WIDTH,
        cam_height  = THERM_HEIGHT
    )

    vis_worker.start()
    therm_worker.start()

    print("─────────────────────────────────────────────")
    print("  DUAL CAMERA YOLO RUNNING")
    print("  Left window  → Arducam B0506 (visual)")
    print("  Right window → TOPDON TC001  (thermal)")
    print("  P = Print stats | ESC = Quit")
    print("─────────────────────────────────────────────\n")

    # ── Main display loop (runs on main thread for cv2.imshow) ────
    while True:
        with vis_worker.lock:
            vis_frame   = vis_worker.latest_frame
            vis_fps     = vis_worker.fps
            vis_dets    = vis_worker.det_count

        with therm_worker.lock:
            therm_frame = therm_worker.latest_frame
            therm_fps   = therm_worker.fps
            therm_dets  = therm_worker.det_count

        if vis_frame is not None:
            vis_display = draw_hud(
                vis_frame.copy(), "Arducam B0506", vis_fps, vis_dets)
            cv2.imshow("YOLO | Arducam B0506", vis_display)

        if therm_frame is not None:
            therm_display = draw_hud(
                therm_frame.copy(), "TOPDON TC001", therm_fps, therm_dets)
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
