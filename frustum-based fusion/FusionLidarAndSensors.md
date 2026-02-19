# Frustrum-based fusion
This fusion is what autnonomous cars like Tesla implemented when they started out.
## Task needed to implement this type of fusion
1. Have the data sent from the YOLO(coming from the cameras) into the software named RViz(robot visualizor simulator)
2. Confirm if hypersen provide a ROS driver/node
3. Hardest thing is the extrinsic calibration(where LIDAR is positioned relative to your Arducam in 3D space (x, y, z, roll, pitch, yaw)).
