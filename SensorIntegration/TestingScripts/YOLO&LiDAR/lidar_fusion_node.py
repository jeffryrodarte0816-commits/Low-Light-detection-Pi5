#!/usr/bin/env python3
"""
lidar_fusion_node.py
=====================
ROS Fusion Node — Combines YOLO camera detections with HPS-3D LiDAR point cloud
to resolve object occlusion (determines which objects are in front of/behind others).

Subscribes:
    /camera/detections       → vision_msgs/Detection2DArray  (Arducam YOLO boxes)
    /thermal/detections      → vision_msgs/Detection2DArray  (TOPDON YOLO boxes)
    /lidar/points            → sensor_msgs/PointCloud2       (HPS-3D160-U cloud)

Publishes:
    /fusion/detections       → std_msgs/String   (JSON — fused objects with depth + occlusion)
    /fusion/markers          → visualization_msgs/MarkerArray  (RViz 3D bounding boxes)
    /fusion/occlusion_pairs  → std_msgs/String   (JSON — which object is behind which)

Pipeline per frame:
    1. Time-synchronize all 3 input streams (ApproximateTimeSynchronizer)
    2. Project LiDAR points into camera image frame using TF transform + camera intrinsics
    3. For each YOLO detection box, gather all LiDAR points that project into it
    4. Compute median depth of each detection → assign Z to each 2D box
    5. Detect overlapping 2D boxes (IoU > threshold)
    6. For overlapping pairs → rank by Z depth to determine front/back
    7. Publish results + RViz markers

Coordinate frames:
    camera_link   : Arducam optical frame (set in your URDF or static TF)
    lidar_link    : HPS-3D LiDAR frame    (set in your URDF or static TF)
    map / odom    : world frame (optional)

    The critical transform is:  lidar_link → camera_link
    You MUST broadcast this via a static_transform_publisher or URDF.
    Example (if LiDAR is 5cm to the right of camera, same height, same yaw):
        rosrun tf static_transform_publisher 0.05 0 0 0 0 0 camera_link lidar_link 100

Camera intrinsics (set via rosparam):
    ~fx, ~fy   : focal lengths in pixels
    ~cx, ~cy   : principal point in pixels
    ~img_w     : image width  (Arducam: 1920)
    ~img_h     : image height (Arducam: 1080)
    These must match the camera you are projecting into.

Detection message format (published on /camera/detections and /thermal/detections):
    This node expects vision_msgs/Detection2DArray.
    If your YOLO nodes publish a custom message type, set ~use_custom_msg True
    and see the _parse_custom_detections() stub below to adapt the parser.

Usage:
    rosrun <your_package> lidar_fusion_node.py

Controls (via /fusion/command std_msgs/String):
    "reset_stats"  → clear occlusion counters
"""

import rospy
import numpy as np
import json
import tf2_ros
import tf2_geometry_msgs
import threading
from collections import defaultdict

from std_msgs.msg           import String, Header, ColorRGBA
from sensor_msgs.msg        import PointCloud2
from geometry_msgs.msg      import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2

# ApproximateTimeSynchronizer lets us sync topics that don't share exact stamps
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Vision messages for Detection2D (install: sudo apt install ros-<distro>-vision-msgs)
try:
    from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
    VISION_MSGS_AVAILABLE = True
except ImportError:
    VISION_MSGS_AVAILABLE = False
    rospy.logwarn("[Fusion] vision_msgs not found — falling back to JSON string topics. "
                  "Install with: sudo apt install ros-<distro>-vision-msgs")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# HPS-3D160-U LiDAR field of view (degrees) → used for sanity-checking projections
LIDAR_HFOV_DEG = 76.0
LIDAR_VFOV_DEG = 32.0

# Occlusion classification labels
OCC_FRONT     = "FRONT"   # this object is in front of the other
OCC_BEHIND    = "BEHIND"  # this object is behind the other
OCC_COPLANAR  = "COPLANAR"

# Detection source labels (mirrored from fusion_decision.py)
SRC_VISUAL   = "V"
SRC_THERMAL  = "T"
SRC_FUSED    = "V+T"
SRC_LIDAR    = "L"

# RViz marker colors (RGBA 0–1)
COLOR_FUSED   = ColorRGBA(0.0, 0.86, 0.0, 0.85)   # green
COLOR_VISUAL  = ColorRGBA(1.0, 0.71, 0.0, 0.75)   # amber/blue
COLOR_THERMAL = ColorRGBA(1.0, 0.31, 0.0, 0.75)   # red
COLOR_OCCLUDED = ColorRGBA(0.5, 0.5, 1.0, 0.6)    # light blue → occluded objects


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: IoU for 2D normalized boxes [x1,y1,x2,y2]
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(a, b):
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FUSION NODE
# ─────────────────────────────────────────────────────────────────────────────

class LidarFusionNode:

    def __init__(self):
        rospy.init_node("lidar_fusion_node", anonymous=False)

        # ── Camera intrinsics ─────────────────────────────────────────────────
        # Default values approximate for Arducam B0506 at 1920×1080.
        # Override with rosparam for your exact calibration.
        self.fx    = rospy.get_param("~fx",    1500.0)
        self.fy    = rospy.get_param("~fy",    1500.0)
        self.cx    = rospy.get_param("~cx",    960.0)
        self.cy    = rospy.get_param("~cy",    540.0)
        self.img_w = rospy.get_param("~img_w", 1920)
        self.img_h = rospy.get_param("~img_h", 1080)

        # ── Topic configuration ───────────────────────────────────────────────
        self.camera_topic  = rospy.get_param("~camera_topic",  "/camera/detections")
        self.thermal_topic = rospy.get_param("~thermal_topic", "/thermal/detections")
        self.lidar_topic   = rospy.get_param("~lidar_topic",   "/lidar/points")
        self.camera_frame  = rospy.get_param("~camera_frame",  "camera_link")
        self.lidar_frame   = rospy.get_param("~lidar_frame",   "lidar_link")

        # ── Fusion thresholds ─────────────────────────────────────────────────
        self.iou_overlap_thresh   = rospy.get_param("~iou_overlap_thresh",  0.15)
        self.depth_diff_thresh_m  = rospy.get_param("~depth_diff_thresh_m", 0.08)
        self.min_lidar_pts_in_box = rospy.get_param("~min_lidar_pts",       3)
        self.sync_slop_s          = rospy.get_param("~sync_slop",           0.05)

        # ── TF listener ───────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer(rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_fused    = rospy.Publisher("/fusion/detections",      String,      queue_size=5)
        self.pub_markers  = rospy.Publisher("/fusion/markers",         MarkerArray, queue_size=5)
        self.pub_occ      = rospy.Publisher("/fusion/occlusion_pairs", String,      queue_size=5)

        # ── Command subscriber ────────────────────────────────────────────────
        rospy.Subscriber("/fusion/command", String, self._cmd_callback)

        # ── Synchronized subscribers ──────────────────────────────────────────
        self._setup_synchronized_subscribers()

        # ── Stats ─────────────────────────────────────────────────────────────
        self._stats = defaultdict(int)
        self._lock  = threading.Lock()

        rospy.loginfo("[Fusion] Node ready. Waiting for synchronized sensor data...")

    # ── Subscriber setup ──────────────────────────────────────────────────────

    def _setup_synchronized_subscribers(self):
        """
        ApproximateTimeSynchronizer aligns all 3 streams by timestamp.
        slop = max allowed time gap in seconds between matched messages.
        """
        if VISION_MSGS_AVAILABLE:
            cam_sub   = Subscriber(self.camera_topic,  Detection2DArray)
            therm_sub = Subscriber(self.thermal_topic, Detection2DArray)
        else:
            # Fall back to String topics if vision_msgs unavailable
            cam_sub   = Subscriber(self.camera_topic,  String)
            therm_sub = Subscriber(self.thermal_topic, String)

        lidar_sub = Subscriber(self.lidar_topic, PointCloud2)

        sync = ApproximateTimeSynchronizer(
            [cam_sub, therm_sub, lidar_sub],
            queue_size=10,
            slop=self.sync_slop_s
        )
        sync.registerCallback(self._synchronized_callback)

    # ── Main synchronized callback ────────────────────────────────────────────

    def _synchronized_callback(self, cam_msg, therm_msg, lidar_msg):
        """
        Called once per synchronized triplet.
        All messages have matching timestamps (within slop).
        """
        stamp = lidar_msg.header.stamp

        # 1. Parse camera detections into normalized dicts
        cam_dets   = self._parse_detections(cam_msg,   SRC_VISUAL)
        therm_dets = self._parse_detections(therm_msg, SRC_THERMAL)

        # 2. Decision-level fusion (camera-only, mirrors fusion_decision.py logic)
        fused_dets = self._camera_fusion(cam_dets, therm_dets)

        if not fused_dets:
            return   # nothing to do this frame

        # 3. Get LiDAR→camera transform
        transform = self._get_transform(self.lidar_frame, self.camera_frame, stamp)
        if transform is None:
            rospy.logwarn_throttle(5.0,
                "[Fusion] No TF transform lidar→camera. Check static_transform_publisher.")
            return

        # 4. Project LiDAR points into camera frame
        lidar_in_cam = self._transform_lidar_to_camera(lidar_msg, transform)
        if lidar_in_cam is None or len(lidar_in_cam) == 0:
            return

        # 5. Assign depth to each detection box
        self._assign_lidar_depth(fused_dets, lidar_in_cam)

        # 6. Resolve occlusion pairs
        occlusion_pairs = self._resolve_occlusion(fused_dets)

        # 7. Publish everything
        self._publish_fused(fused_dets, stamp)
        self._publish_occlusion(occlusion_pairs, stamp)
        self._publish_markers(fused_dets, occlusion_pairs, stamp)

        with self._lock:
            self._stats["frames"] += 1
            self._stats["occlusion_pairs"] += len(occlusion_pairs)

    # ── Detection parsing ─────────────────────────────────────────────────────

    def _parse_detections(self, msg, source: str) -> list:
        """
        Parse camera detection message into internal dict format.

        Dict fields:
            box_norm : [x1,y1,x2,y2] normalized 0–1 in camera image
            conf     : float confidence 0–1
            cls      : int class id
            label    : str class name
            source   : SRC_VISUAL | SRC_THERMAL | SRC_FUSED
            depth_m  : float | None  (assigned later from LiDAR)
            lidar_pts: int           (number of LiDAR points inside box)
        """
        dets = []
        if VISION_MSGS_AVAILABLE and isinstance(msg, Detection2DArray):
            for det in msg.detections:
                bb    = det.bbox
                cx_px = bb.center.x
                cy_px = bb.center.y
                w_px  = bb.size_x
                h_px  = bb.size_y
                x1 = (cx_px - w_px / 2) / self.img_w
                y1 = (cy_px - h_px / 2) / self.img_h
                x2 = (cx_px + w_px / 2) / self.img_w
                y2 = (cy_px + h_px / 2) / self.img_h
                conf = det.results[0].score if det.results else 0.0
                cls  = int(det.results[0].id) if det.results else 0
                label = str(cls)
                dets.append({
                    "box_norm":  [max(0,x1), max(0,y1), min(1,x2), min(1,y2)],
                    "conf":      float(conf),
                    "cls":       cls,
                    "label":     label,
                    "source":    source,
                    "depth_m":   None,
                    "lidar_pts": 0,
                })
        elif isinstance(msg, String):
            # Fallback: expect JSON string with list of detection dicts
            # matching the format from fusion_decision.py
            dets = self._parse_custom_detections(msg.data, source)
        return dets

    def _parse_custom_detections(self, json_str: str, source: str) -> list:
        """
        Adapter for JSON string detection topics (from fusion_decision.py).
        The JSON should be a list of dicts with keys:
            box_norm, conf, cls, label
        Publish this from your fusion_decision.py by adding a ROS publisher
        that calls json.dumps(fused_dets) on each frame.
        """
        dets = []
        try:
            raw = json.loads(json_str)
            for d in raw:
                dets.append({
                    "box_norm":  d.get("box_norm",  [0,0,1,1]),
                    "conf":      d.get("conf",       0.0),
                    "cls":       d.get("cls",         0),
                    "label":     d.get("label",       "?"),
                    "source":    source,
                    "depth_m":   None,
                    "lidar_pts": 0,
                })
        except (json.JSONDecodeError, KeyError) as e:
            rospy.logwarn_throttle(5.0, f"[Fusion] Detection parse error: {e}")
        return dets

    # ── Camera-level decision fusion (same logic as fusion_decision.py) ────────

    def _camera_fusion(self, vis_dets: list, therm_dets: list,
                        iou_thresh: float = 0.30) -> list:
        """
        Merges visual + thermal detections (same algorithm as fusion_decision.py).
        Adds LiDAR depth later in _assign_lidar_depth().
        """
        fused = []
        matched = set()

        for v in vis_dets:
            best_iou, best_idx = 0, -1
            for ti, t in enumerate(therm_dets):
                if ti in matched or t["cls"] != v["cls"]:
                    continue
                iou = compute_iou(v["box_norm"], t["box_norm"])
                if iou > best_iou:
                    best_iou, best_idx = iou, ti

            if best_iou >= iou_thresh and best_idx >= 0:
                t  = therm_dets[best_idx]
                matched.add(best_idx)
                vw, tw = v["conf"], t["conf"]
                total  = vw + tw
                avg_box = [(v["box_norm"][i]*vw + t["box_norm"][i]*tw)/total
                           for i in range(4)]
                fused.append({
                    "box_norm":  avg_box,
                    "conf":      min(1.0, max(vw, tw) + 0.10),
                    "cls":       v["cls"],
                    "label":     v["label"],
                    "source":    SRC_FUSED,
                    "depth_m":   None,
                    "lidar_pts": 0,
                })
            else:
                fused.append({**v, "depth_m": None, "lidar_pts": 0})

        for ti, t in enumerate(therm_dets):
            if ti not in matched:
                fused.append({**t, "depth_m": None, "lidar_pts": 0})

        return fused

    # ── TF transform ──────────────────────────────────────────────────────────

    def _get_transform(self, source_frame: str, target_frame: str, stamp: rospy.Time):
        """
        Look up the rigid transform from source_frame to target_frame.
        Returns a 4×4 numpy homogeneous matrix, or None on failure.
        """
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp,
                timeout=rospy.Duration(0.05)
            )
            t = tf_stamped.transform.translation
            q = tf_stamped.transform.rotation
            return self._tf_to_matrix(t, q)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

    @staticmethod
    def _tf_to_matrix(t, q) -> np.ndarray:
        """Convert tf translation + quaternion into a 4×4 homogeneous matrix."""
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),        1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),        2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
        ])
        M = np.eye(4)
        M[:3, :3] = R
        M[0, 3]   = t.x
        M[1, 3]   = t.y
        M[2, 3]   = t.z
        return M

    # ── LiDAR projection ──────────────────────────────────────────────────────

    def _transform_lidar_to_camera(self, lidar_msg: PointCloud2,
                                    transform: np.ndarray) -> np.ndarray:
        """
        Read XYZ points from PointCloud2, apply TF transform,
        then project into the camera image plane.

        Returns array of shape (N, 5):
            [u_pixel, v_pixel, z_cam_m, x_cam_m, y_cam_m]
        Only points that project inside the image and have positive Z are kept.
        """
        # Read XYZ from PointCloud2 (use field names from hps3d_lidar_node.py)
        pts_lidar = []
        for p in pc2.read_points(lidar_msg, field_names=("x","y","z"),
                                  skip_nans=True):
            pts_lidar.append(p)

        if not pts_lidar:
            return None

        pts = np.array(pts_lidar, dtype=np.float32)   # (N, 3)
        # Homogeneous coords (N, 4)
        ones = np.ones((len(pts), 1), dtype=np.float32)
        pts_h = np.hstack([pts, ones])                # (N, 4)

        # Transform into camera frame
        pts_cam = (transform @ pts_h.T).T[:, :3]      # (N, 3)

        # Keep only points in front of the camera (z > 0)
        mask = pts_cam[:, 2] > 0.01
        pts_cam = pts_cam[mask]
        if len(pts_cam) == 0:
            return None

        # Project onto image plane using pinhole model
        x_cam = pts_cam[:, 0]
        y_cam = pts_cam[:, 1]
        z_cam = pts_cam[:, 2]

        u = (x_cam / z_cam * self.fx + self.cx)
        v = (y_cam / z_cam * self.fy + self.cy)

        # Keep only points inside image bounds
        in_img = (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)
        u = u[in_img]; v = v[in_img]
        z_cam = z_cam[in_img]
        x_cam = x_cam[in_img]
        y_cam = y_cam[in_img]

        return np.stack([u, v, z_cam, x_cam, y_cam], axis=1)   # (M, 5)

    # ── Depth assignment ──────────────────────────────────────────────────────

    def _assign_lidar_depth(self, dets: list, lidar_in_cam: np.ndarray):
        """
        For each detection bounding box, collect all LiDAR points that project
        into it and compute the median depth (robust to outliers).
        Mutates each det dict in place: sets depth_m and lidar_pts.
        """
        if lidar_in_cam is None or len(lidar_in_cam) == 0:
            return

        u_pts  = lidar_in_cam[:, 0]
        v_pts  = lidar_in_cam[:, 1]
        z_pts  = lidar_in_cam[:, 2]

        for det in dets:
            x1n, y1n, x2n, y2n = det["box_norm"]
            x1_px = x1n * self.img_w
            y1_px = y1n * self.img_h
            x2_px = x2n * self.img_w
            y2_px = y2n * self.img_h

            # Slightly shrink the box to avoid grabbing background points on edges
            margin_x = (x2_px - x1_px) * 0.10
            margin_y = (y2_px - y1_px) * 0.10

            in_box = (
                (u_pts >= x1_px + margin_x) & (u_pts <= x2_px - margin_x) &
                (v_pts >= y1_px + margin_y) & (v_pts <= y2_px - margin_y)
            )

            depths_in_box = z_pts[in_box]
            det["lidar_pts"] = int(np.sum(in_box))

            if det["lidar_pts"] >= self.min_lidar_pts_in_box:
                # Use median — more robust than mean for cluttered scenes
                det["depth_m"] = float(np.median(depths_in_box))
            else:
                det["depth_m"] = None   # not enough LiDAR coverage

    # ── Occlusion resolution ──────────────────────────────────────────────────

    def _resolve_occlusion(self, dets: list) -> list:
        """
        For every pair of detections with overlapping 2D boxes and known depth:
        - If depth differs by more than threshold → front/back relationship
        - Otherwise → coplanar (treat as same plane)

        Returns list of occlusion dicts:
            {
              "front_idx"  : int,  (index into dets)
              "behind_idx" : int,
              "front_label": str,
              "behind_label": str,
              "front_depth_m" : float,
              "behind_depth_m": float,
              "depth_diff_m"  : float,
              "overlap_iou"   : float
            }
        """
        pairs = []
        n = len(dets)

        for i in range(n):
            for j in range(i + 1, n):
                di = dets[i]
                dj = dets[j]

                # Both need depth information
                if di["depth_m"] is None or dj["depth_m"] is None:
                    continue

                iou = compute_iou(di["box_norm"], dj["box_norm"])
                if iou < self.iou_overlap_thresh:
                    continue

                depth_diff = abs(di["depth_m"] - dj["depth_m"])

                if depth_diff < self.depth_diff_thresh_m:
                    relationship = OCC_COPLANAR
                    front_idx  = i
                    behind_idx = j
                elif di["depth_m"] < dj["depth_m"]:
                    relationship = OCC_FRONT
                    front_idx    = i
                    behind_idx   = j
                else:
                    relationship = OCC_FRONT
                    front_idx    = j
                    behind_idx   = i

                pairs.append({
                    "front_idx":    front_idx,
                    "behind_idx":   behind_idx,
                    "front_label":  dets[front_idx]["label"],
                    "behind_label": dets[behind_idx]["label"],
                    "front_depth_m":  dets[front_idx]["depth_m"],
                    "behind_depth_m": dets[behind_idx]["depth_m"],
                    "depth_diff_m":   depth_diff,
                    "overlap_iou":    round(iou, 3),
                    "relationship":   relationship,
                })

        return pairs

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_fused(self, dets: list, stamp: rospy.Time):
        """Publish enriched detection list as JSON string."""
        output = []
        for d in dets:
            output.append({
                "box_norm":  d["box_norm"],
                "conf":      round(d["conf"], 3),
                "cls":       d["cls"],
                "label":     d["label"],
                "source":    d["source"],
                "depth_m":   round(d["depth_m"], 3) if d["depth_m"] is not None else None,
                "lidar_pts": d["lidar_pts"],
            })
        self.pub_fused.publish(String(data=json.dumps({
            "stamp":      stamp.to_sec(),
            "detections": output,
        })))

    def _publish_occlusion(self, pairs: list, stamp: rospy.Time):
        """Publish occlusion pair analysis as JSON."""
        self.pub_occ.publish(String(data=json.dumps({
            "stamp": stamp.to_sec(),
            "pairs": pairs,
        })))

    def _publish_markers(self, dets: list, pairs: list, stamp: rospy.Time):
        """
        Publish RViz MarkerArray:
        - CUBE_LIST   → one cube per detection (scaled by confidence, colored by source)
        - TEXT_VIEW_FACING → label + depth above each box
        - LINE_LIST   → arrow-like line from front to back for each occlusion pair
        """
        marker_array = MarkerArray()
        mid = 0  # marker id counter

        # ── Occluded index set (for coloring) ─────────────────────────────────
        behind_indices = {p["behind_idx"] for p in pairs}

        for idx, det in enumerate(dets):
            if det["depth_m"] is None:
                continue

            cx_n = (det["box_norm"][0] + det["box_norm"][2]) / 2
            cy_n = (det["box_norm"][1] + det["box_norm"][3]) / 2
            w_n  = det["box_norm"][2] - det["box_norm"][0]
            h_n  = det["box_norm"][3] - det["box_norm"][1]

            # Approximate 3D position using depth + camera geometry
            z    = det["depth_m"]
            x3d  = (cx_n * self.img_w - self.cx) / self.fx * z
            y3d  = (cy_n * self.img_h - self.cy) / self.fy * z
            w3d  = w_n  * self.img_w / self.fx * z
            h3d  = h_n  * self.img_h / self.fy * z

            # Choose color
            if idx in behind_indices:
                color = COLOR_OCCLUDED
            elif det["source"] == SRC_FUSED:
                color = COLOR_FUSED
            elif det["source"] == SRC_THERMAL:
                color = COLOR_THERMAL
            else:
                color = COLOR_VISUAL

            # ── Box marker ────────────────────────────────────────────────────
            box_marker           = Marker()
            box_marker.header    = Header(stamp=stamp, frame_id=self.camera_frame)
            box_marker.ns        = "detections"
            box_marker.id        = mid; mid += 1
            box_marker.type      = Marker.CUBE
            box_marker.action    = Marker.ADD
            box_marker.pose.position.x = x3d
            box_marker.pose.position.y = y3d
            box_marker.pose.position.z = z
            box_marker.pose.orientation.w = 1.0
            box_marker.scale     = Vector3(w3d, h3d, 0.05)
            box_marker.color     = color
            box_marker.lifetime  = rospy.Duration(0.2)
            marker_array.markers.append(box_marker)

            # ── Text label ────────────────────────────────────────────────────
            txt              = Marker()
            txt.header       = Header(stamp=stamp, frame_id=self.camera_frame)
            txt.ns           = "labels"
            txt.id           = mid; mid += 1
            txt.type         = Marker.TEXT_VIEW_FACING
            txt.action       = Marker.ADD
            txt.pose.position.x = x3d
            txt.pose.position.y = y3d - h3d / 2 - 0.05
            txt.pose.position.z = z
            txt.pose.orientation.w = 1.0
            txt.scale.z      = 0.12
            txt.color        = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            behind_tag = " [BEHIND]" if idx in behind_indices else ""
            txt.text         = (f"[{det['source']}] {det['label']} "
                                f"| {det['depth_m']:.2f}m "
                                f"| {det['lidar_pts']}pts{behind_tag}")
            txt.lifetime     = rospy.Duration(0.2)
            marker_array.markers.append(txt)

        # ── Occlusion pair arrows ─────────────────────────────────────────────
        for pair in pairs:
            fi  = pair["front_idx"]
            bi  = pair["behind_idx"]
            fd  = dets[fi]["depth_m"]
            bd  = dets[bi]["depth_m"]
            fcx = (dets[fi]["box_norm"][0] + dets[fi]["box_norm"][2]) / 2
            fcy = (dets[fi]["box_norm"][1] + dets[fi]["box_norm"][3]) / 2
            bcx = (dets[bi]["box_norm"][0] + dets[bi]["box_norm"][2]) / 2
            bcy = (dets[bi]["box_norm"][1] + dets[bi]["box_norm"][3]) / 2

            fx3 = (fcx * self.img_w - self.cx) / self.fx * fd
            fy3 = (fcy * self.img_h - self.cy) / self.fy * fd
            bx3 = (bcx * self.img_w - self.cx) / self.fx * bd
            by3 = (bcy * self.img_h - self.cy) / self.fy * bd

            arr              = Marker()
            arr.header       = Header(stamp=stamp, frame_id=self.camera_frame)
            arr.ns           = "occlusion_lines"
            arr.id           = mid; mid += 1
            arr.type         = Marker.LINE_LIST
            arr.action       = Marker.ADD
            arr.scale.x      = 0.02
            arr.color        = ColorRGBA(1.0, 1.0, 0.0, 0.9)   # yellow
            arr.points       = [
                Point(fx3, fy3, fd),
                Point(bx3, by3, bd),
            ]
            arr.lifetime     = rospy.Duration(0.2)
            marker_array.markers.append(arr)

        self.pub_markers.publish(marker_array)

    # ── Command handler ───────────────────────────────────────────────────────

    def _cmd_callback(self, msg: String):
        cmd = msg.data.strip()
        if cmd == "reset_stats":
            with self._lock:
                self._stats.clear()
            rospy.loginfo("[Fusion] Stats reset.")
        elif cmd == "print_stats":
            with self._lock:
                rospy.loginfo(f"[Fusion] Stats: {dict(self._stats)}")
        else:
            rospy.logwarn(f"[Fusion] Unknown command: '{cmd}'")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        node = LidarFusionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
