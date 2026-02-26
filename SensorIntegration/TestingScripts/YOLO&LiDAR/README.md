# Each python script and purpose
Once the ImageOverlap testing scripts work as intented, then we update that code working onto these files sicne these python scripts are now implementing the ROS on top of Image overlap  
1. Drivers	"hps3d_lidar_node.py"-Starts the LiDAR stream.
2. Detection	"fusion_decision.py"-Finds the animal and aligns Thermal + Visual.
3. Fusion	"lidar_fusion_node.py"-Takes the 2D "Fused Box" and calculates the 3D Distance.

If these scripts work as intented, then copy these python scripts and paste onto the frustrum-based fusion folder
