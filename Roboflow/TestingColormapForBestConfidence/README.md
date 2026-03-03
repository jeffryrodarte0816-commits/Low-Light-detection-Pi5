# Testing colormaps to see which gives best confidence level when using YOLO  
Images are converted into raw format in the computer(jetson/rpi5) such as grayscale which is 8-bit, while the YOLO can use the colormapping that gives a better edge,thus leading to better results in confidence level of model, the colormapping that gives best confidence value is considered best colormap to use in terms of accuracy and detection.  
Steps to test:  
1. train a grayscale(8-bit) model within ultralytics
2. run the model to get a best.pt
3. apply the best.pt directory within the "ConfidenceComparisonColormap.py", inside the model_path(line 17)
4. the " data_dir= " is where we put within the folder with all the pictures that are .jpeg and should hold grayscale images  
Colormaps mean confidence value:
TWILIGHT    0.785581
PINK    0.829959
VIRIDIS    0.717258
JET        0.702301

# Task to complete
Already completed testing ultralytics example coco8_grayscale.yaml which I made a model and trained it, ran the model onto the ConfidenceComparisonColormap.py and resulted in colormap_PINK giving best confidence so theoretically PINK is best colormap for thermal in terms of using YOLO for detection.  
Need now to test dataset for this specific project, using only grayscale images, test colormaps with those grayscale images and see if best colormap would be different.  
If the PINK results in best colormap for our dataset, then colormap = PINK for fusion_decision.py and calibrate_offset.py within the SensorIntegration folder.  
