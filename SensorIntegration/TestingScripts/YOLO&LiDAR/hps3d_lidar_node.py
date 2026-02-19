#!/usr/bin/env python3
"""
hps3d_lidar_node.py
====================
ROS Driver Node for DFR0728 / HPS-3D160-U Solid-State LiDAR
- Wraps the Hypersen HPS3D SDK via ctypes (libhps3d.so)
- Publishes: /lidar/points     → sensor_msgs/PointCloud2  (160×60 ordered cloud)
             /lidar/depth_image → sensor_msgs/Image        (raw depth map, 16UC1 mm)
             /lidar/status     → std_msgs/String           (JSON diagnostics)

Hardware:  DFR0728 / HPS-3D160-U
Interface: USB → /dev/ttyACM0  (appears as CDC-ACM serial device)
SDK:       libhps3d.so  (Hypersen SDK V1.8, place in /usr/local/lib/)

Usage:
    sudo chmod 777 /dev/ttyACM0
    rosrun <your_package> hps3d_lidar_node.py

Parameters (rosparam):
    ~device_port   : USB device path   (default: /dev/ttyACM0)
    ~frame_id      : TF frame name     (default: lidar_link)
    ~publish_rate  : Hz cap (0=max)    (default: 0)
    ~hdr_mode      : 0=off 1=simple 2=auto 3=super (default: 1)
    ~min_dist_m    : clip near points  (default: 0.25)
    ~max_dist_m    : clip far points   (default: 12.0)
    ~sdk_lib_path  : path to .so       (default: /usr/local/lib/libhps3d.so)
"""

import rospy
import ctypes
import ctypes.util
import os
import sys
import json
import struct
import threading
import numpy as np
from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pc2

# ─────────────────────────────────────────────────────────────────────────────
# HPS3D SENSOR CONSTANTS  (from api.h / SDK manual)
# ─────────────────────────────────────────────────────────────────────────────
LIDAR_WIDTH     = 160
LIDAR_HEIGHT    = 60
LIDAR_PIXELS    = LIDAR_WIDTH * LIDAR_HEIGHT   # 9600 pixels

# Packet types
PACKET_FULL_DEPTH   = 0x04
PACKET_SIMPLE_DEPTH = 0x03

# Run modes
RUN_IDLE         = 0x00
RUN_SINGLE_SHOT  = 0x01
RUN_CONTINUOUS   = 0x02

# HDR modes
HDR_DISABLE  = 0x00
SIMPLE_HDR   = 0x01
AUTO_HDR     = 0x02
SUPER_HDR    = 0x03

# Return codes
RET_OK               = 0x00
RET_ERROR            = 0xFF

# Distance sentinel values (mm, from SDK docs)
DIST_OUT_OF_RANGE    = 65400   # too far / no reflection
DIST_TOO_CLOSE       = 65500   # closer than 0.25m
DIST_LOW_SIGNAL      = 65300   # weak signal
DIST_INVALID         = 65530   # sensor internally invalid

INVALID_DIST_THRESH  = 65000   # anything >= this is invalid

# ─────────────────────────────────────────────────────────────────────────────
# SDK CTYPES WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class HPS3DFullDepthData(ctypes.Structure):
    """
    Maps to the FullDepthDataTypeDef struct in api.h.
    depth_data: flat array of uint16 distances in mm (row-major, 160×60)
    point_cloud_data: flat array of float32 XYZ triplets (9600 × 3)
    """
    _fields_ = [
        ("distance_average", ctypes.c_uint16),
        ("distance_min",     ctypes.c_uint16),
        ("saturation_count", ctypes.c_uint16),
        ("depth_data",       ctypes.c_uint16 * LIDAR_PIXELS),
        # Point cloud: x0,y0,z0, x1,y1,z1, ...
        ("point_cloud_data", ctypes.c_float  * (LIDAR_PIXELS * 3)),
    ]


class HPS3DSDK:
    """
    Thin ctypes wrapper around libhps3d.so.
    Loads the shared library and exposes the API calls needed by this node.
    """

    def __init__(self, lib_path: str):
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise RuntimeError(
                f"Cannot load HPS3D SDK library at '{lib_path}': {e}\n"
                f"Make sure you ran:  sudo ldconfig  after copying libhps3d.so "
                f"to /usr/local/lib/"
            )
        self._setup_signatures()
        rospy.loginfo(f"[LiDAR] SDK library loaded from {lib_path}")

    def _setup_signatures(self):
        """Assign argument/return types for each SDK function."""
        lib = self._lib

        # int HPS3D_Connect(const char *device_name, uint8_t *device_addr)
        lib.HPS3D_Connect.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8)]
        lib.HPS3D_Connect.restype  = ctypes.c_int

        # int HPS3D_ConfigInit(uint8_t device_addr, uint8_t *roi_num)
        lib.HPS3D_ConfigInit.argtypes = [ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint8)]
        lib.HPS3D_ConfigInit.restype  = ctypes.c_int

        # int HPS3D_SetRunMode(uint8_t device_addr, uint8_t mode)
        lib.HPS3D_SetRunMode.argtypes = [ctypes.c_uint8, ctypes.c_uint8]
        lib.HPS3D_SetRunMode.restype  = ctypes.c_int

        # int HPS3D_SetPacketType(uint8_t device_addr, uint8_t type)
        lib.HPS3D_SetPacketType.argtypes = [ctypes.c_uint8, ctypes.c_uint8]
        lib.HPS3D_SetPacketType.restype  = ctypes.c_int

        # int HPS3D_SetHDRMode(uint8_t device_addr, uint8_t mode)
        lib.HPS3D_SetHDRMode.argtypes = [ctypes.c_uint8, ctypes.c_uint8]
        lib.HPS3D_SetHDRMode.restype  = ctypes.c_int

        # int HPS3D_SetPointCloudEn(uint8_t device_addr, bool enable)
        lib.HPS3D_SetPointCloudEn.argtypes = [ctypes.c_uint8, ctypes.c_bool]
        lib.HPS3D_SetPointCloudEn.restype  = ctypes.c_int

        # int HPS3D_SetOpticalEnable(uint8_t device_addr, bool enable)
        lib.HPS3D_SetOpticalEnable.argtypes = [ctypes.c_uint8, ctypes.c_bool]
        lib.HPS3D_SetOpticalEnable.restype  = ctypes.c_int

        # int HPS3D_SingleMeasurement(uint8_t device_addr,
        #                             FullDepthDataTypeDef *data)
        lib.HPS3D_SingleMeasurement.argtypes = [
            ctypes.c_uint8,
            ctypes.POINTER(HPS3DFullDepthData)
        ]
        lib.HPS3D_SingleMeasurement.restype = ctypes.c_int

        # int HPS3D_Disconnect(uint8_t device_addr)
        lib.HPS3D_Disconnect.argtypes = [ctypes.c_uint8]
        lib.HPS3D_Disconnect.restype  = ctypes.c_int

    # ── Public API calls ──────────────────────────────────────────────────────

    def connect(self, device_port: str):
        addr = ctypes.c_uint8(0)
        ret  = self._lib.HPS3D_Connect(device_port.encode(), ctypes.byref(addr))
        if ret != RET_OK:
            raise RuntimeError(f"HPS3D_Connect failed (ret={ret:#04x}). "
                               f"Check USB cable and device permissions.")
        return addr.value

    def config_init(self, device_addr: int):
        roi_num = ctypes.c_uint8(0)
        ret = self._lib.HPS3D_ConfigInit(device_addr, ctypes.byref(roi_num))
        if ret != RET_OK:
            raise RuntimeError(f"HPS3D_ConfigInit failed (ret={ret:#04x})")
        return roi_num.value

    def set_run_mode(self, device_addr: int, mode: int):
        ret = self._lib.HPS3D_SetRunMode(device_addr, mode)
        if ret != RET_OK:
            rospy.logwarn(f"[LiDAR] HPS3D_SetRunMode({mode}) returned {ret:#04x}")

    def set_packet_type(self, device_addr: int, ptype: int):
        ret = self._lib.HPS3D_SetPacketType(device_addr, ptype)
        if ret != RET_OK:
            rospy.logwarn(f"[LiDAR] HPS3D_SetPacketType returned {ret:#04x}")

    def set_hdr_mode(self, device_addr: int, mode: int):
        ret = self._lib.HPS3D_SetHDRMode(device_addr, mode)
        if ret != RET_OK:
            rospy.logwarn(f"[LiDAR] HPS3D_SetHDRMode returned {ret:#04x}")

    def set_point_cloud_en(self, device_addr: int, enable: bool):
        ret = self._lib.HPS3D_SetPointCloudEn(device_addr, enable)
        if ret != RET_OK:
            rospy.logwarn(f"[LiDAR] HPS3D_SetPointCloudEn returned {ret:#04x}")

    def set_optical_enable(self, device_addr: int, enable: bool):
        ret = self._lib.HPS3D_SetOpticalEnable(device_addr, enable)
        if ret != RET_OK:
            rospy.logwarn(f"[LiDAR] HPS3D_SetOpticalEnable returned {ret:#04x}")

    def single_measurement(self, device_addr: int) -> HPS3DFullDepthData:
        data = HPS3DFullDepthData()
        ret  = self._lib.HPS3D_SingleMeasurement(device_addr, ctypes.byref(data))
        if ret != RET_OK:
            raise RuntimeError(f"HPS3D_SingleMeasurement failed (ret={ret:#04x})")
        return data

    def disconnect(self, device_addr: int):
        self._lib.HPS3D_Disconnect(device_addr)


# ─────────────────────────────────────────────────────────────────────────────
# ROS NODE
# ─────────────────────────────────────────────────────────────────────────────

class HPS3DLidarNode:

    def __init__(self):
        rospy.init_node("hps3d_lidar_node", anonymous=False)

        # ── Parameters ────────────────────────────────────────────────────────
        self.device_port  = rospy.get_param("~device_port",  "/dev/ttyACM0")
        self.frame_id     = rospy.get_param("~frame_id",     "lidar_link")
        self.publish_rate = rospy.get_param("~publish_rate", 0)
        self.hdr_mode     = rospy.get_param("~hdr_mode",     SIMPLE_HDR)
        self.min_dist_m   = rospy.get_param("~min_dist_m",   0.25)
        self.max_dist_m   = rospy.get_param("~max_dist_m",   12.0)
        self.sdk_lib_path = rospy.get_param("~sdk_lib_path", "/usr/local/lib/libhps3d.so")

        self.min_dist_mm = int(self.min_dist_m * 1000)
        self.max_dist_mm = int(self.max_dist_m * 1000)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_cloud  = rospy.Publisher("/lidar/points",      PointCloud2, queue_size=5)
        self.pub_depth  = rospy.Publisher("/lidar/depth_image", Image,       queue_size=5)
        self.pub_status = rospy.Publisher("/lidar/status",      String,      queue_size=5)

        # ── SDK init ──────────────────────────────────────────────────────────
        rospy.loginfo(f"[LiDAR] Connecting to DFR0728 on {self.device_port} ...")
        self.sdk          = HPS3DSDK(self.sdk_lib_path)
        self.device_addr  = self.sdk.connect(self.device_port)
        rospy.loginfo(f"[LiDAR] Connected. Device address = {self.device_addr}")

        roi_num = self.sdk.config_init(self.device_addr)
        rospy.loginfo(f"[LiDAR] ConfigInit complete. ROI regions = {roi_num}")

        # Configure sensor
        self.sdk.set_packet_type(self.device_addr, PACKET_FULL_DEPTH)
        self.sdk.set_hdr_mode(self.device_addr, self.hdr_mode)
        self.sdk.set_optical_enable(self.device_addr, True)   # enables XYZ point cloud
        self.sdk.set_point_cloud_en(self.device_addr, True)
        self.sdk.set_run_mode(self.device_addr, RUN_IDLE)     # start idle; loop drives it

        # ── Stats ─────────────────────────────────────────────────────────────
        self._frame_count  = 0
        self._error_count  = 0
        self._lock         = threading.Lock()

        rospy.loginfo("[LiDAR] Node ready. Publishing on /lidar/points and /lidar/depth_image")
        rospy.on_shutdown(self._shutdown)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def spin(self):
        rate = rospy.Rate(self.publish_rate) if self.publish_rate > 0 else None

        while not rospy.is_shutdown():
            try:
                data = self.sdk.single_measurement(self.device_addr)
                stamp = rospy.Time.now()

                depth_array = np.array(data.depth_data, dtype=np.uint16)   # (9600,)
                xyz_array   = np.array(data.point_cloud_data, dtype=np.float32)  # (28800,)

                self._publish_point_cloud(depth_array, xyz_array, stamp)
                self._publish_depth_image(depth_array, stamp)
                self._publish_status(data, stamp)

                self._frame_count += 1

            except RuntimeError as e:
                self._error_count += 1
                rospy.logwarn_throttle(5.0, f"[LiDAR] Measurement error: {e}")

            if rate:
                rate.sleep()

    # ── PointCloud2 publisher ─────────────────────────────────────────────────

    def _publish_point_cloud(self, depth_mm: np.ndarray,
                              xyz_flat: np.ndarray,
                              stamp: rospy.Time):
        """
        Build a sensor_msgs/PointCloud2 from the SDK's XYZ output.

        The SDK returns point cloud in metres (x,y,z per pixel).
        We filter out invalid depth pixels before publishing.
        The cloud is kept ordered (width=160, height=60) so downstream
        nodes can do 2D pixel-to-point lookups.
        """
        xyz = xyz_flat.reshape(LIDAR_HEIGHT, LIDAR_WIDTH, 3)   # (60,160,3) metres

        # Build validity mask — True where depth is a real measurement
        valid = (depth_mm < INVALID_DIST_THRESH) & \
                (depth_mm >= self.min_dist_mm)   & \
                (depth_mm <= self.max_dist_mm)
        valid_2d = valid.reshape(LIDAR_HEIGHT, LIDAR_WIDTH)

        # For invalid pixels, push XYZ to NaN so RViz renders them as empty
        xyz[~valid_2d] = np.nan

        # Pack into PointCloud2 (ordered, so width/height meaningful)
        header = Header(stamp=stamp, frame_id=self.frame_id)
        fields = [
            PointField("x", 0,  PointField.FLOAT32, 1),
            PointField("y", 4,  PointField.FLOAT32, 1),
            PointField("z", 8,  PointField.FLOAT32, 1),
            PointField("d", 12, PointField.UINT16,  1),  # raw depth mm
        ]

        # Interleave depth into point rows
        depth_2d = depth_mm.reshape(LIDAR_HEIGHT, LIDAR_WIDTH).astype(np.uint16)

        points = []
        for r in range(LIDAR_HEIGHT):
            for c in range(LIDAR_WIDTH):
                x, y, z = xyz[r, c]
                d = int(depth_2d[r, c])
                points.append((float(x), float(y), float(z), d))

        cloud_msg = pc2.create_cloud(header, fields, points)
        cloud_msg.width    = LIDAR_WIDTH
        cloud_msg.height   = LIDAR_HEIGHT
        cloud_msg.is_dense = False   # contains NaN for invalid pixels

        self.pub_cloud.publish(cloud_msg)

    # ── Depth image publisher ─────────────────────────────────────────────────

    def _publish_depth_image(self, depth_mm: np.ndarray, stamp: rospy.Time):
        """
        Publish the raw 160×60 depth map as a 16-bit mono image (mm units).
        Encoding: 16UC1 — compatible with RViz depth display.
        Invalid pixels are left at their raw sentinel values (65300–65530).
        """
        img_data = depth_mm.reshape(LIDAR_HEIGHT, LIDAR_WIDTH).astype(np.uint16)

        msg              = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height       = LIDAR_HEIGHT
        msg.width        = LIDAR_WIDTH
        msg.encoding     = "16UC1"
        msg.is_bigendian = False
        msg.step         = LIDAR_WIDTH * 2   # 2 bytes per pixel
        msg.data         = img_data.tobytes()

        self.pub_depth.publish(msg)

    # ── Status publisher ──────────────────────────────────────────────────────

    def _publish_status(self, data: HPS3DFullDepthData, stamp: rospy.Time):
        """Publish a JSON diagnostics string every frame for monitoring."""
        status = {
            "stamp":            stamp.to_sec(),
            "frame_count":      self._frame_count,
            "error_count":      self._error_count,
            "distance_avg_mm":  int(data.distance_average),
            "distance_min_mm":  int(data.distance_min),
            "saturation_count": int(data.saturation_count),
            "device_addr":      self.device_addr,
        }
        self.pub_status.publish(String(data=json.dumps(status)))

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _shutdown(self):
        rospy.loginfo("[LiDAR] Shutting down — setting sensor to IDLE.")
        try:
            self.sdk.set_run_mode(self.device_addr, RUN_IDLE)
            self.sdk.disconnect(self.device_addr)
        except Exception as e:
            rospy.logwarn(f"[LiDAR] Shutdown error (non-fatal): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        node = HPS3DLidarNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
    except RuntimeError as e:
        rospy.logfatal(f"[LiDAR] Fatal startup error: {e}")
        sys.exit(1)
