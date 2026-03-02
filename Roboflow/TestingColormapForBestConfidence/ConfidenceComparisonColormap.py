"""
ConfidenceComparisonColormap.py

Purpose to compare confidence level of colormaps by applying colormap to raw thermal data, runs YOLO after 
"""
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
# Using your updated training path from test_camera2
MODEL_PATH   = "/home/sunnysquad/yolo_project/ultralytics/runs/detect/test_grayscale/weights/best.pt"
DATA_DIR     = "/home/sunnysquad/yolo_project/ultralytics/datasets/coco8-grayscale/images/val/"
RESULTS_DIR  = "evaluation_results/"
CONF_THRESH  = 0.25

COLORMAPS = {
    "INFERNO": cv2.COLORMAP_INFERNO,
    "TWILIGHT":     cv2.COLORMAP_TWILIGHT,
    "GRAYSCALE":     cv2.COLORMAP_GRAYSCALE
}

# ─────────────────────────────────────────────────────────────────
# EVALUATION ENGINE
# ─────────────────────────────────────────────────────────────────
def run_evaluation():
    model = YOLO(MODEL_PATH)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    performance_data = []

    image_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.jpg', '.png'))]
    print(f"Starting evaluation on {len(image_files)} images...")

    for img_name in image_files:
        raw_path = os.path.join(DATA_DIR, img_name)
        img_gray = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        
        if img_gray is None: continue

        for cmap_name, cmap_code in COLORMAPS.items():
            # 1. Apply colormap (Creates a 3-channel BGR image)
            color_img = cv2.applyColorMap(img_gray, cmap_code)
            
            # 2. FIX: Convert back to Grayscale (Reduces 3 channels to 1)
            # This allows the model to see the enhanced contrast while staying in 1-channel format.
            input_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            
            # 3. Run YOLO Inference on the 1-channel result
            results = model.predict(source=input_img, conf=CONF_THRESH, verbose=False)
            
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                label = results[0].names[cls]
                
                performance_data.append({
                    "Image": img_name,
                    "Colormap": cmap_name,
                    "Label": label,
                    "Confidence": conf
                })

    # ─────────────────────────────────────────────────────────────
    # DATA ANALYSIS & VISUALIZATION
    # ─────────────────────────────────────────────────────────────
    if not performance_data:
        print("No detections found with current confidence threshold.")
        return

    df = pd.DataFrame(performance_data)
    df.to_csv(os.path.join(RESULTS_DIR, "detailed_metrics.csv"), index=False)
    
    summary = df.groupby("Colormap")["Confidence"].mean().sort_values(ascending=False)
    print("\n--- Summary: Mean Confidence Scores ---")
    print(summary)

    plt.figure(figsize=(10, 6))
    summary.plot(kind='bar', color=['orange', 'red', 'blue'])
    plt.title("YOLO Confidence Comparison (Grayscale-Trained Model)")
    plt.ylabel("Mean Confidence Score")
    plt.xlabel("Colormap Palette (Converted to Gray for Inference)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, "confidence_comparison.png"))
    print(f"\nEvaluation complete. Results saved in {RESULTS_DIR}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
    else:
        run_evaluation()
