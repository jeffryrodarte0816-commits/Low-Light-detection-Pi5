# Sensor Integration (Starting out with tc001 and day/night arducam)
https://github.com/leswright1977/PyThermalCamera/blob/main/README.md  
Use the Homography method to integrate both sensors. Need opencv and numpy.
First connect the tc001, then verify if dev/video0, then connect arducam and verify if dev/video2.  
We ran the "test_arducam.py" to verify the arducam is video2 which it was.  
To verify the imports used within a specific script:(Example below)
```bash
python3 -c "import cv2, numpy, argparse, time, io, sys; print('✅ Success! All libraries are installed.')"
```
## Sensor alignment
If you mount them vertically (Arducam on top, Topdon on bottom, for example), your calibration process becomes easier:  
Lock the X-Axis: In the Python script, your saved_x (horizontal shift) should be almost exactly 0 (or whatever centers the image). You generally won't need to touch this again.  
Tune the Y-Axis: You only have to worry about saved_y.  
The "Sweet Spot": Calibrate the saved_y variable using an object at your average target distance (e.g., 3 meters).  
Anything closer than 3m will look slightly misaligned vertically (ghost image appears lower).  
Anything further than 3m will look slightly misaligned vertically (ghost image appears higher).  
But in both cases, the Left/Right alignment is perfect, so your rover tracks straight.  
Code below that includes the scripts above:  
```bash
import cv2
import numpy as np
import time

# --- HARDWARE CONFIGURATION ---
# Jetson Orin Nano often creates multiple /dev/videoX nodes per camera.
# Use 'v4l2-ctl --list-devices' in terminal to find the real indices.

# Arducam (Visual) - 1080p
VIS_WIDTH = 1920
VIS_HEIGHT = 1080

# TOPDON TC001 (Thermal) - Native 256x192
THERM_WIDTH = 256
THERM_HEIGHT = 192

# --- CALIBRATION DEFAULTS ---
# Update these after you run the calibration mode once
saved_scale = 1.0    # Zoom level
saved_x = 0          # Horizontal shift
saved_y = 0          # Vertical shift
saved_alpha = 0.5    # Transparency

def open_camera(index, width, height):
    # Enforce V4L2 backend for Jetson compatibility
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    
    # Force MJPG if available (lower USB bandwidth usage)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def main():
    global saved_scale, saved_x, saved_y, saved_alpha
    
    # 1. Device Discovery Helper
    # We try to open a few indices to see what's what
    print("Attempting to connect to cameras...")
    vis_cap = None
    therm_cap = None

    # Try indices 0 through 4
    for i in range(4):
        temp_cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if temp_cap.isOpened():
            w = temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Index {i}: Found Camera with res {w}x{h}")
            
            # Simple logic to auto-detect based on resolution
            # TOPDON is small (256x192), Arducam is big (1920x1080)
            if w == 256 and h == 192 and therm_cap is None:
                print(f"  -> Identified as TOPDON Thermal (Index {i})")
                therm_cap = temp_cap
            elif w > 640 and vis_cap is None:
                print(f"  -> Identified as Arducam Visual (Index {i})")
                vis_cap = temp_cap
                # Re-set resolution to 1080p in case it defaulted low
                vis_cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIS_WIDTH)
                vis_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIS_HEIGHT)
            else:
                temp_cap.release()
    
    if vis_cap is None or therm_cap is None:
        print("\nERROR: Could not auto-identify both cameras.")
        print("Please check USB connections or manually set IDs in code.")
        return

    print("\n--- CONTROLS ---")
    print("WASD  : Move Thermal Layer")
    print("Q / E : Scale Thermal Layer")
    print("+ / - : Adjust Transparency")
    print("P     : Print Current Calibration Values")
    print("ESC   : Quit")

    while True:
        # 2. Read Frames
        ret_v, frame_vis = vis_cap.read()
        ret_t, frame_therm = therm_cap.read()

        if not ret_v or not ret_t:
            print("Frame error.")
            time.sleep(0.1)
            continue

        # 3. Process Thermal Image
        # Resize to match Visual Dimensions strictly for overlay
        # (We use INTER_NEAREST for speed, or INTER_LINEAR for smoothness)
        therm_scaled = cv2.resize(frame_therm, (VIS_WIDTH, VIS_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # 4. Apply Affine Transform (Calibration)
        # Center point for scaling
        cx, cy = VIS_WIDTH // 2, VIS_HEIGHT // 2
        
        # Build the Matrix
        # [ Scale,  0,    Shift_X + (1-Scale)*Center_X ]
        # [ 0,      Scale, Shift_Y + (1-Scale)*Center_Y ]
        M = np.float32([
            [saved_scale, 0, saved_x + (1 - saved_scale) * cx],
            [0, saved_scale, saved_y + (1 - saved_scale) * cy]
        ])
        
        therm_aligned = cv2.warpAffine(therm_scaled, M, (VIS_WIDTH, VIS_HEIGHT))

        # 5. Fusion
        # If TOPDON output is grayscale, convert to BGR first
        if len(therm_aligned.shape) == 2:
            therm_aligned = cv2.cvtColor(therm_aligned, cv2.COLOR_GRAY2BGR)
            # Optional: Apply false color if it's raw grayscale
            therm_aligned = cv2.applyColorMap(therm_aligned, cv2.COLORMAP_JET)

        fused = cv2.addWeighted(frame_vis, 1.0, therm_aligned, saved_alpha, 0.0)

        # 6. Display
        # Resize for display if 1080p is too big for your monitor
        display_img = cv2.resize(fused, (1280, 720))
        cv2.imshow("Jetson Fusion", display_img)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
        elif key == ord('w'): saved_y -= 10
        elif key == ord('s'): saved_y += 10
        elif key == ord('a'): saved_x -= 10
        elif key == ord('d'): saved_x += 10
        elif key == ord('q'): saved_scale -= 0.05
        elif key == ord('e'): saved_scale += 0.05
        elif key == ord('='): saved_alpha = min(saved_alpha + 0.1, 1.0)
        elif key == ord('-'): saved_alpha = max(saved_alpha - 0.1, 0.0)
        elif key == ord('p'):
            print(f"CALIBRATION: Scale={saved_scale:.2f}, X={saved_x}, Y={saved_y}")

    vis_cap.release()
    therm_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```
