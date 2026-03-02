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
