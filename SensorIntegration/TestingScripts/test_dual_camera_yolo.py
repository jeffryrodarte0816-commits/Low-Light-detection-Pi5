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

──────────────────────────────────────────────────────────────────
TROUBLESHOOTING — run this first if cameras aren't found:
    v4l2-ctl --list-devices
Then update ARDUCAM_INDEX and TOPDON_INDEX below to match.
──────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────
# ★ SET THESE TO MATCH YOUR SYSTEM ★
# Run `v4l2-ctl --list-devices` to confirm which /dev/videoN each is.
# ─────────────────────────────────────────────────────────────────
ARDUCAM_INDEX = 0   # /dev/video0
TOPDON_INDEX  = 2   # /dev/video2  ✅ confirmed

# ─────────────────────────────────────────────────────────────────
# MODEL PATH
# ─────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/sunnysquad/yolo_project/ultralytics/runs/detect/test_camera/weights/best.pt"

# ─────────────────────────────────────────────────────────────────
# HARDWARE RESOLUTION
# ─────────────────────────────────────────────────────────────────
VIS_WIDTH    = 640
VIS_HEIGHT   = 480
VIS_FPS      = 15

THERM_WIDTH  = 256
THERM_HEIGHT = 192

# Display window sizes
VIS_DISP_WIDTH    = 640
VIS_DISP_HEIGHT   = 480
THERM_DISP_WIDTH  = 512   # 2× upscale so thermal is easier to see
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
# ─────────────────────────────────────────────────────────────────
THERMAL_COLORMAP = cv2.COLORMAP_INFERNO

# Detection box colors (BGR)
BOX_COLOR   = (0, 255, 0)
LABEL_COLOR = (0, 0, 0)


# ─────────────────────────────────────────────────────────────────
# CAMERA OPEN HELPERS
# ─────────────────────────────────────────────────────────────────

def scan_cameras():
    """Print all available /dev/video* devices and their reported sizes.
    Useful for confirming ARDUCAM_INDEX and TOPDON_INDEX."""
    print("─── Available V4L2 devices ───────────────────")
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  /dev/video{i}  →  {int(w)}×{int(h)} @ {fps:.0f}fps")
            cap.release()
    print("──────────────────────────────────────────────")


def open_arducam(index):
    """
    Open Arducam B0506 at /dev/video<index>.
    Sets MJPG format first (required before resolution change on UVC devices),
    then 640×480 @ 15fps.
    Returns configured VideoCapture or None.
    """
    print(f"  Opening Arducam  at /dev/video{index} ...")
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"    ❌ Could not open /dev/video{index}")
        return None

    # MJPG must be set BEFORE resolution — many UVC drivers require this order
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  VIS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIS_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, VIS_FPS)

    # Warmup — discard the first few frames; some cameras return garbage initially
    for _ in range(5):
        cap.read()

    # Verify a real frame comes through
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"    ❌ Arducam opened but cannot read frames from video{index}")
        cap.release()
        return None

    w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"    ✅ Arducam B0506  →  {int(w)}×{int(h)} @ {fps:.0f}fps  (frame shape: {frame.shape})")
    return cap


def open_topdon(index):
    """
    Open TOPDON TC001 at /dev/video<index>.
    The TC001 streams raw 16-bit thermal data packed as YUYV at 256×192.
    We do NOT request a specific resolution — let the driver report what it has.
    Returns configured VideoCapture or None.

    NOTE: The TC001 often reports 640×480 when queried with cap.get() even
    though it actually delivers 256×192 frames. This is a driver quirk.
    We trust the actual frame shape instead of the reported properties.
    """
    print(f"  Opening TC001    at /dev/video{index} ...")
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"    ❌ Could not open /dev/video{index}")
        return None

    # Do NOT force a resolution — the TC001 driver ignores set() calls and
    # returns garbage if you try to change its native mode.
    # Just set a buffer size hint to keep latency low.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Warmup reads — TC001 sometimes needs several frames to stabilise
    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"    ❌ TC001 opened but cannot read frames from video{index}")
        cap.release()
        return None

    w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"    ✅ TOPDON TC001   →  reported {int(w)}×{int(h)}, actual frame shape: {frame.shape}")
    return cap


# ─────────────────────────────────────────────────────────────────
# DRAWING UTILITIES
# ─────────────────────────────────────────────────────────────────

def draw_detections(frame, results, disp_w, disp_h, infer_w, infer_h):
    """Draw YOLO boxes scaled from inference resolution → display resolution."""
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
    """Overlay camera name, FPS, and detection count."""
    cv2.putText(frame, label,               (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",   (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Dets: {det_count}",(10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


# ─────────────────────────────────────────────────────────────────
# CAMERA WORKER THREAD
# ─────────────────────────────────────────────────────────────────

class CameraWorker(threading.Thread):
    def __init__(self, cap, model, window_name, is_thermal=False,
                 disp_width=640, disp_height=480):
        super().__init__(daemon=True)
        self.cap         = cap
        self.model       = model
        self.window_name = window_name
        self.is_thermal  = is_thermal
        self.disp_width  = disp_width
        self.disp_height = disp_height

        self.latest_frame = None
        self.det_count    = 0
        self.fps          = 0.0
        self.lock         = threading.Lock()
        self.running      = True

    def stop(self):
        self.running = False

    def run(self):
        frame_count = 0
        results     = None
        fps_timer   = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            frame_count += 1

            # ── Thermal: convert to colour before YOLO ────────────
            if self.is_thermal:
                # TC001 delivers BGR (from YUYV conversion by V4L2).
                # Convert to gray first to strip any chroma noise,
                # then apply a colourmap so YOLO gets an RGB-like image.
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.applyColorMap(gray, THERMAL_COLORMAP)

            # ── YOLO inference every N frames ─────────────────────
            if frame_count % YOLO_EVERY_N == 0:
                yolo_input = cv2.resize(frame, (YOLO_IMGSZ, YOLO_IMGSZ))
                results = self.model.predict(
                    source  = yolo_input,
                    conf    = YOLO_CONF,
                    iou     = YOLO_IOU,
                    imgsz   = YOLO_IMGSZ,
                    verbose = False,
                    device  = 0,
                )

            # ── Build display frame ───────────────────────────────
            display = cv2.resize(frame, (self.disp_width, self.disp_height))
            count   = 0
            if results is not None:
                display, count = draw_detections(
                    display, results,
                    self.disp_width, self.disp_height,
                    YOLO_IMGSZ, YOLO_IMGSZ,
                )

            # ── Rolling FPS counter ───────────────────────────────
            now     = time.time()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                with self.lock:
                    self.fps = frame_count / elapsed
                frame_count = 0
                fps_timer   = now

            with self.lock:
                self.det_count    = count
                self.latest_frame = display.copy()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    # Print all available devices first — useful for debugging
    scan_cameras()

    # ── Open cameras by known index ───────────────────────────────
    vis_cap   = open_arducam(ARDUCAM_INDEX)
    therm_cap = open_topdon(TOPDON_INDEX)

    # TC001 sometimes registers two nodes (e.g. video2=metadata, video3=image).
    # If the primary index failed, automatically try the next one.
    if therm_cap is None:
        fallback = TOPDON_INDEX + 1
        print(f"\n  ⚠️  TC001 not on video{TOPDON_INDEX}, trying video{fallback} ...")
        therm_cap = open_topdon(fallback)
        if therm_cap is not None:
            print(f"  ✅ TC001 found at video{fallback}")

    # Last resort: scan everything except the Arducam index
    if therm_cap is None:
        print("\n  ⚠️  Still no TC001 — scanning all remaining nodes...")
        for i in range(10):
            if i in (ARDUCAM_INDEX, TOPDON_INDEX, TOPDON_INDEX + 1):
                continue
            candidate = open_topdon(i)
            if candidate is not None:
                therm_cap = candidate
                print(f"  ✅ TC001 found at video{i} — set TOPDON_INDEX={i} to skip scan next time.")
                break

    if vis_cap is None or therm_cap is None:
        missing = []
        if vis_cap   is None: missing.append("Arducam B0506")
        if therm_cap is None: missing.append("TOPDON TC001")
        print(f"\n❌ Could not open: {', '.join(missing)}")
        print("   Run `v4l2-ctl --list-devices` and update ARDUCAM_INDEX / TOPDON_INDEX.")
        if vis_cap:   vis_cap.release()
        if therm_cap: therm_cap.release()
        return

    # ── Load YOLO models ──────────────────────────────────────────
    print(f"\n🤖 Loading YOLO models (x2)...")
    vis_model   = YOLO(MODEL_PATH)
    therm_model = YOLO(MODEL_PATH)
    print("   ✅ Models loaded.\n")

    # ── Create display windows ────────────────────────────────────
    cv2.namedWindow("YOLO | Arducam B0506", cv2.WINDOW_NORMAL)
    cv2.namedWindow("YOLO | TOPDON TC001",  cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO | Arducam B0506", VIS_DISP_WIDTH,    VIS_DISP_HEIGHT)
    cv2.resizeWindow("YOLO | TOPDON TC001",  THERM_DISP_WIDTH,  THERM_DISP_HEIGHT)
    cv2.moveWindow("YOLO | Arducam B0506",   20,                100)
    cv2.moveWindow("YOLO | TOPDON TC001",    VIS_DISP_WIDTH + 60, 100)

    # ── Start worker threads ──────────────────────────────────────
    vis_worker = CameraWorker(
        cap         = vis_cap,
        model       = vis_model,
        window_name = "YOLO | Arducam B0506",
        is_thermal  = False,
        disp_width  = VIS_DISP_WIDTH,
        disp_height = VIS_DISP_HEIGHT,
    )
    therm_worker = CameraWorker(
        cap         = therm_cap,
        model       = therm_model,
        window_name = "YOLO | TOPDON TC001",
        is_thermal  = True,
        disp_width  = THERM_DISP_WIDTH,
        disp_height = THERM_DISP_HEIGHT,
    )

    vis_worker.start()
    therm_worker.start()

    print("─────────────────────────────────────────────")
    print("  DUAL CAMERA YOLO RUNNING")
    print(f"  Left  → Arducam B0506  (video{ARDUCAM_INDEX}, {VIS_WIDTH}×{VIS_HEIGHT} @ {VIS_FPS}fps)")
    print(f"  Right → TOPDON TC001   (video{TOPDON_INDEX}, {THERM_WIDTH}×{THERM_HEIGHT})")
    print("  P = Print stats | ESC = Quit")
    print("─────────────────────────────────────────────\n")

    # ── Main display loop ─────────────────────────────────────────
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
            cv2.imshow("YOLO | Arducam B0506",
                       draw_hud(vis_frame.copy(), "Arducam B0506", vis_fps, vis_dets))

        if therm_frame is not None:
            cv2.imshow("YOLO | TOPDON TC001",
                       draw_hud(therm_frame.copy(), "TOPDON TC001", therm_fps, therm_dets))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        elif key == ord('p'):
            print(f"\n📊 Arducam  → FPS: {vis_fps:.1f}  Detections: {vis_dets}")
            print(f"📊 TC001    → FPS: {therm_fps:.1f}  Detections: {therm_dets}\n")

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
