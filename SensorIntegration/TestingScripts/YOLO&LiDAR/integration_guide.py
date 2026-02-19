# ═══════════════════════════════════════════════════════════════════════════════
# ROS LIDAR INTEGRATION GUIDE
# DFR0728 (HPS-3D160-U) + Arducam B0506 + TOPDON TC001 + YOLO
# Jetson Orin Nano | ROS Noetic | Python 3
# ═══════════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────────
# STEP 1 — SDK LIBRARY SETUP
# ───────────────────────────────────────────────────────────────────────────────

# Clone the Hypersen SDK (confirmed working with DFR0728):
#   git clone https://github.com/DFRobotdl/DFR0728_HPS3D_SDK-HPS3D160_SDK-V1.8.git
#
# Copy the shared library to the system library path:
#   sudo cp DFR0728_HPS3D_SDK-HPS3D160_SDK-V1.8/V1.8/lib/linux/libhps3d.so /usr/local/lib/
#   sudo ldconfig
#
# Set USB device permissions (add to /etc/udev/rules.d/ to make permanent):
#   sudo chmod 777 /dev/ttyACM0
#
# Or create a udev rule (permanent fix):
#   echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", MODE="0666"' \
#        | sudo tee /etc/udev/rules.d/99-hps3d.rules
#   sudo udevadm control --reload-rules


# ───────────────────────────────────────────────────────────────────────────────
# STEP 2 — ROS PACKAGE SETUP
# ───────────────────────────────────────────────────────────────────────────────

# Create catkin workspace and package (skip if you already have one):
#
#   mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src
#   catkin_create_pkg sensor_fusion rospy std_msgs sensor_msgs geometry_msgs \
#                     visualization_msgs tf2_ros message_filters vision_msgs
#
#   cd ~/catkin_ws && catkin_make
#   source devel/setup.bash
#
# Copy the two node scripts:
#   cp hps3d_lidar_node.py    ~/catkin_ws/src/sensor_fusion/scripts/
#   cp lidar_fusion_node.py   ~/catkin_ws/src/sensor_fusion/scripts/
#   chmod +x ~/catkin_ws/src/sensor_fusion/scripts/*.py
#
# Install vision_msgs (needed by fusion node):
#   sudo apt install ros-noetic-vision-msgs


# ───────────────────────────────────────────────────────────────────────────────
# STEP 3 — BRIDGE PATCH FOR fusion_decision.py
# ───────────────────────────────────────────────────────────────────────────────
#
# Add these lines to your existing fusion_decision.py so it publishes detections
# to ROS topics that lidar_fusion_node.py can subscribe to.
# This is the MINIMUM change needed — no restructuring required.

# ---------- ADD AT TOP of fusion_decision.py (after existing imports) ----------

import rospy
from std_msgs.msg import String as ROSString
import json as _json

def _init_ros_publishers():
    """Call once in main() before the capture loop."""
    rospy.init_node("fusion_camera_node", anonymous=False, disable_signals=True)
    cam_pub   = rospy.Publisher("/camera/detections",  ROSString, queue_size=5)
    therm_pub = rospy.Publisher("/thermal/detections", ROSString, queue_size=5)
    return cam_pub, therm_pub

def _publish_to_ros(cam_pub, therm_pub, vis_dets, therm_dets_raw):
    """
    Call inside the main() capture loop after extract_detections().
    Converts detection dicts to JSON and publishes to ROS.
    """
    if not rospy.is_shutdown():
        cam_pub.publish(ROSString(data=_json.dumps(vis_dets)))
        therm_pub.publish(ROSString(data=_json.dumps(therm_dets_raw)))

# ---------- CHANGES inside main() of fusion_decision.py ----------------------
#
# Near the top of main(), add:
#
#     cam_pub, therm_pub = _init_ros_publishers()
#
# Inside the while loop, after extract_detections() calls, add:
#
#     _publish_to_ros(cam_pub, therm_pub, vis_dets, therm_dets_raw)
#
# That is the only change. Your existing display loop is untouched.


# ───────────────────────────────────────────────────────────────────────────────
# STEP 4 — TF TRANSFORM (CRITICAL — measure your actual offsets!)
# ───────────────────────────────────────────────────────────────────────────────
#
# You must tell ROS how the LiDAR is physically positioned relative to the camera.
# Measure these offsets in metres from the camera optical center to the LiDAR:
#
#   x = sideways offset (+ right)
#   y = vertical offset (+ down)
#   z = forward offset  (+ forward)
#   Rotation is usually 0 0 0 if they face the same direction.
#
# Example — LiDAR is 6cm to the right, 2cm below, same forward plane:
#
#   rosrun tf static_transform_publisher 0.06 0.02 0 0 0 0 camera_link lidar_link 100
#
# Add this to your launch file so it starts automatically:
#
#   <node pkg="tf" type="static_transform_publisher" name="lidar_to_cam_tf"
#         args="0.06 0.02 0 0 0 0 camera_link lidar_link 100" />


# ───────────────────────────────────────────────────────────────────────────────
# STEP 5 — CAMERA INTRINSICS  (calibrate with ros camera_calibration package)
# ───────────────────────────────────────────────────────────────────────────────
#
# The fusion node needs your Arducam's focal length and principal point.
# Default values (fx=fy=1500, cx=960, cy=540) are approximate for 1920x1080.
#
# To get real calibration values:
#   rosrun camera_calibration cameracalibrator.py --size 9x6 --square 0.025 \
#          image:=/arducam/image_raw camera:=/arducam
#
# Then set these rosparam values in your launch file:
#   <param name="/lidar_fusion_node/fx"    value="1487.3" />
#   <param name="/lidar_fusion_node/fy"    value="1489.1" />
#   <param name="/lidar_fusion_node/cx"    value="963.4"  />
#   <param name="/lidar_fusion_node/cy"    value="541.8"  />


# ───────────────────────────────────────────────────────────────────────────────
# STEP 6 — LAUNCH FILE  (sensor_fusion.launch)
# ───────────────────────────────────────────────────────────────────────────────

LAUNCH_FILE = """
<launch>

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- LiDAR Driver Node (DFR0728 / HPS-3D160-U)                  -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  <node pkg="sensor_fusion" type="hps3d_lidar_node.py"
        name="hps3d_lidar_node" output="screen">
    <param name="device_port"   value="/dev/ttyACM0" />
    <param name="frame_id"      value="lidar_link" />
    <param name="hdr_mode"      value="1" />     <!-- 1=simple, 2=auto, 3=super -->
    <param name="min_dist_m"    value="0.25" />
    <param name="max_dist_m"    value="8.0" />
    <param name="sdk_lib_path"  value="/usr/local/lib/libhps3d.so" />
  </node>

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- Static TF: LiDAR frame → Camera frame                       -->
  <!-- EDIT these values to match your physical sensor placement!  -->
  <!-- args: x y z yaw pitch roll parent child rate(Hz)            -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  <node pkg="tf" type="static_transform_publisher"
        name="lidar_to_cam_tf"
        args="0.06 0.02 0 0 0 0 camera_link lidar_link 100" />

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- Fusion Node                                                  -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  <node pkg="sensor_fusion" type="lidar_fusion_node.py"
        name="lidar_fusion_node" output="screen">

    <!-- Camera intrinsics — replace with your calibration values  -->
    <param name="fx"     value="1500.0" />
    <param name="fy"     value="1500.0" />
    <param name="cx"     value="960.0"  />
    <param name="cy"     value="540.0"  />
    <param name="img_w"  value="1920"   />
    <param name="img_h"  value="1080"   />

    <!-- Topic names -->
    <param name="camera_topic"   value="/camera/detections"  />
    <param name="thermal_topic"  value="/thermal/detections" />
    <param name="lidar_topic"    value="/lidar/points"       />
    <param name="camera_frame"   value="camera_link"         />
    <param name="lidar_frame"    value="lidar_link"          />

    <!-- Fusion thresholds -->
    <param name="iou_overlap_thresh"  value="0.15" />   <!-- min 2D box overlap to check depth -->
    <param name="depth_diff_thresh_m" value="0.08" />   <!-- min depth gap to call front/behind -->
    <param name="min_lidar_pts"       value="3"    />   <!-- min points needed to assign depth -->
    <param name="sync_slop"           value="0.05" />   <!-- time sync tolerance (seconds) -->
  </node>

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- RViz (optional — opens visualization automatically)         -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- <node pkg="rviz" type="rviz" name="rviz"
             args="-d $(find sensor_fusion)/rviz/fusion.rviz" /> -->

</launch>
"""


# ───────────────────────────────────────────────────────────────────────────────
# STEP 7 — RVIZ SETUP  (what to add in RViz after launch)
# ───────────────────────────────────────────────────────────────────────────────
#
# Open RViz:   rosrun rviz rviz
#
# Add these displays (Displays panel → Add):
#
#  1. PointCloud2
#       Topic: /lidar/points
#       Color: z-axis (creates a depth-colored heatmap)
#       Style: Points,  Size: 3px
#
#  2. MarkerArray
#       Topic: /fusion/markers
#       (Shows 3D boxes for each detected object, colored by source.
#        Yellow lines connect front/behind occlusion pairs.)
#
#  3. Image  (optional — see raw camera)
#       Topic: /arducam/image_raw  (or wherever your camera publishes)
#
#  4. TF    (optional — shows frame axes)
#
# Set Fixed Frame to: camera_link


# ───────────────────────────────────────────────────────────────────────────────
# STEP 8 — RUNNING THE SYSTEM
# ───────────────────────────────────────────────────────────────────────────────
#
#  Terminal 1 — ROS core:
#      roscore
#
#  Terminal 2 — Camera fusion node (your existing script + bridge patch):
#      source ~/catkin_ws/devel/setup.bash
#      python3 fusion_decision.py
#
#  Terminal 3 — Full sensor stack:
#      source ~/catkin_ws/devel/setup.bash
#      roslaunch sensor_fusion sensor_fusion.launch
#
#  Terminal 4 — Monitor occlusion output:
#      rostopic echo /fusion/occlusion_pairs
#
#  Terminal 5 — Monitor fused detections with depth:
#      rostopic echo /fusion/detections


# ───────────────────────────────────────────────────────────────────────────────
# TOPIC MAP (full system)
# ───────────────────────────────────────────────────────────────────────────────
#
#  fusion_decision.py   →  /camera/detections      (JSON string, vis YOLO boxes)
#  fusion_decision.py   →  /thermal/detections     (JSON string, thermal YOLO boxes)
#  hps3d_lidar_node.py  →  /lidar/points           (PointCloud2, 160×60 ordered)
#  hps3d_lidar_node.py  →  /lidar/depth_image      (Image 16UC1, mm)
#  hps3d_lidar_node.py  →  /lidar/status           (JSON diagnostics)
#  lidar_fusion_node.py →  /fusion/detections      (JSON, all objects + depth)
#  lidar_fusion_node.py →  /fusion/occlusion_pairs (JSON, front/behind pairs)
#  lidar_fusion_node.py →  /fusion/markers         (MarkerArray for RViz)


# ───────────────────────────────────────────────────────────────────────────────
# KNOWN GOTCHAS
# ───────────────────────────────────────────────────────────────────────────────
#
#  1. TF transform MUST be set before fusion node starts, or all frames will be
#     dropped. Launch lidar_to_cam_tf node first (it's in the launch file above).
#
#  2. ApproximateTimeSynchronizer only works if all 3 publishers stamp their
#     messages. fusion_decision.py uses cv2 and has no ROS clock — the bridge
#     patch above uses rospy.Time.now() implicitly via publisher. This is fine
#     since all 3 will be stamped within the same machine clock.
#
#  3. libhps3d.so is 32-bit on some SDK versions. Check with:
#       file /usr/local/lib/libhps3d.so
#     It should say "ELF 64-bit" on Jetson (ARM64). If it says 32-bit,
#     download the ARM64 build from the DFRobotdl GitHub repo.
#
#  4. If /dev/ttyACM0 doesn't appear after USB plug-in, check:
#       dmesg | grep tty
#     The device may be at /dev/ttyACM1 — set ~device_port accordingly.
#
#  5. The LiDAR's 76°×32° FOV is narrower than the camera. Objects near the
#     edges of the camera frame will have no LiDAR depth (depth_m = null).
#     This is expected — the fusion node handles it gracefully.
