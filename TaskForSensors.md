# Models to run on each sensor
## Thermal Camera
Run the Instance segmentation model- use predict where mask is used to differentiate animals overlapping each other.   
## LIDAR 
Just be used for information based so nothing image based, will be used for autonomous driving and if necessary in conjunction with sensor that's necessary.  
Need to do: Talk to jeffry in regards to lidar and ros working together, along with lidar to be used with both cameras. Focus on the camera fusion, then try to apply lidar.
## Day/Night Camera
Pose estimation model and detection model only for day/night, possibly not run the Oriented Bounding boxes(OBB) model since it's more for top-view camera(maybe not applicable towards rover from ground).
# Sensor alignment (Arducam Day/Night vertically aligned above the TopDon tc001)
If you mount them vertically (Arducam on top, Topdon on bottom, for example), your calibration process becomes easier:  
Lock the X-Axis: In the Python script, your saved_x (horizontal shift) should be almost exactly 0 (or whatever centers the image). You generally won't need to touch this again.  
Tune the Y-Axis: You only have to worry about saved_y.  
The "Sweet Spot": Calibrate the saved_y variable using an object at your average target distance (e.g., 3 meters).  
Anything closer than 3m will look slightly misaligned vertically (ghost image appears lower).  
Anything further than 3m will look slightly misaligned vertically (ghost image appears higher).  
But in both cases, the Left/Right alignment is perfect, so your rover tracks straight.  

# Decision-Level(Late) fusion for both cameras  
Purpose: To best get results via accuracy, We are going to run YOLO models independently on both cameras  
Step 1: Pre-process the Thermal Feed  
Step 2: The "Overlap" Logic (Intersection over Union)  

Scripts using:  
1.A calibration script to adjust the image of both sensors to overlap  
2. Fusion script to use both YOLO on cameras individually, merging detection and give singel output  
Process to follow:  
```bash
source /home/sunnysquad/venv/bin/activate
python3 calibrate_offset.py
python3 fusion_decision.py
```
## Results and variables to look for:
### Results:
Results from bounding boxes:  
Box Color: Green  
Meaning: (thick)Both sensors agree — highest confidence, fused box  

Box Color:  Blue  
Meaning: Arducam only detected it  

Box Color: Red  
Meaning: TC001 only detected it (heat signature, no visual match) 
### Variables to look for:  
#### within calibrate_offset.py:  
Look for offset values-  
```bash
Offset X: 12px
Offset Y: 87px
```
Offset x: should be very close to 0 (±20px) since your cameras are vertically stacked — there's almost no horizontal parallax  
Offset y: should be a positive number reflecting the physical vertical gap between your two lenses. A typical 3–5cm mount separation at 3 meters gives roughly 60–120px in visual-res space  
  
Erros that can occur:  
1.offset_x is large (>50px) — your cameras aren't truly vertically aligned, or you clicked off-center. Run C to clear and reclicks  
2.offset_y is 0 or negative — you may have clicked in the wrong order, or the thermal window click didn't register properly  
3.Numbers wildly change between calibration attempts — your target object is too small or too far. Use something large and thermally distinct (a person, a warm bottle) at exactly 3 meters  

#### within fusion_decision.py:  
##### For Yolo confidence threshold:  
```bash
YOLO_CONF = 0.35
```
1.Too low → lots of ghost red/blue boxes, noisy display  
2.Too high → real animals missed entirely, green boxes disappear  
3.Watch for: if you're seeing 10+ red/blue boxes on an empty scene, raise to 0.45. If known objects aren't being detected at all, lower to 0.25  

##### For Fusion IoU threshold:  
```bash
FUSION_IOU_THRESHOLD = 0.3
```
This is the most important value in the whole pipeline. It controls how much the visual and thermal boxes must overlap to be called the same object.    
1.Too low (e.g. 0.1) → unrelated detections get merged into false green boxes  
2.Too high (e.g. 0.6) → legitimate dual detections never fuse, everything stays red/blue  
3.Watch for: if you're standing in front of the rover and your detection is blue AND red simultaneously instead of green, your IoU is too high or your offset calibration is off. Start by re-running calibrate_offset.py first, then lower this to 0.2 if still not fusing  

##### For Confidence boost for fused detections:  
```bash
FUSION_CONF_BOOST = 0.1
```
Adds 0.1 to confidence when both sensors agree. Harmless in most cases — only matters if you're logging confidence scores for downstream decision-making on the rover. Leave it unless you're piping scores into navigation logic.  

##### For Inference cadence:  
```bash
YOLO_EVERY_N = 2
```
1.At 2, YOLO runs every other frame. Boxes persist on the off-frame from the last inference result  
2.Watch for: if detections are "flickering" or jumping around, raise to 3 or 4 — this smooths the display because stale boxes are held longer. If your rover is moving fast and boxes lag behind moving animals, lower to 1 (every frame) but expect FPS to drop ~40%  

##### For Thermal overlay transparency:  
```bash
THERMAL_ALPHA = 0.35 
```
Use the 'P' key, to get live diagnostic data:  
```bash
fused=45 | visual_only=120 | thermal_only=8
```
Pattern:fused is very low  
Represents: both others are highOffset is wrong   
Fix: boxes aren't overlappingRe-run calibrate_offset.py  

Pattern: thermal_only is always 0  
Represents:TC001 YOLO isn't detecting anything  
Fix: Lower YOLO_CONF, check colormap is being applied before inference  

Pattern: visual_only dominates entirely  
Represents: Normal for bright daylight — thermal adds less. Fine  
Fix: No fix needed  

Pattern: fused is very high but you see false boxes  
Represents: FUSION_IOU_THRESHOLD too low  
Fix: Raise to 0.4 or 0.5  

Pattern: All three are near 0  
Represents: Both models failing — wrong model path or cameras swapped  
Fix: Verify MODEL_PATH and re-check v4l2-ctl --list-devices    



