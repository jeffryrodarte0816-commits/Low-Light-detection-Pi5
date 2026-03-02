"""
ColormapComparison.py

Purpose to compare colormaps visually with live thermal camera and see which colormaps can be optimal to use to then apply towards the "ConfidenceComparisonColormap.py"

"""
import cv2
import numpy as np

# 1. Initialize Camera (use 0 for USB thermal or GStreamer pipeline for Jetson)
cap = cv2.VideoCapture(0)

# 2. Define the four colormaps you want to compare
# These are the ones we identified from your earlier images
cmaps = [
    ("PARULA", cv2.COLORMAP_PARULA),
    ("CIVIDIS", cv2.COLORMAP_CIVIDIS),
    ("TWILIGHT", cv2.COLORMAP_TWILIGHT),
    ("TURBO", cv2.COLORMAP_TURBO)
]

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale if the camera provides 3-channel BGR
    # Note: Many raw thermal sensors provide grayscale natively
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    processed_frames = []
    for name, code in cmaps:
        # Apply colormap
        colored = cv2.applyColorMap(gray, code)
        # Add label text to the frame
        cv2.putText(colored, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        processed_frames.append(colored)

    # 3. Create a 2x2 grid (Tiling)
    top_row = np.concatenate((processed_frames[0], processed_frames[1]), axis=1)
    bottom_row = np.concatenate((processed_frames[2], processed_frames[3]), axis=1)
    grid = np.concatenate((top_row, bottom_row), axis=0)

    # 4. Show the combined results
    cv2.imshow('Thermal Colormap Comparison', grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
