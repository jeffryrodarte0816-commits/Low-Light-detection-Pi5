# Models to run on each sensor
## Thermal Camera
Run the Instance segmentation model- use predict where mask is used to differentiate animals overlapping each other.   
## LIDAR 
Just be used for information based so nothing image based, will be used for autonomous driving and if necessary in conjunction with sensor that's necessary.
## Day/Night Camera
Pose estimation model and detection model only for day/night, possibly run as well the Oriented Bounding boxes(OBB) model since it's more for top-view camera(maybe not applicable towards rover from ground). 
