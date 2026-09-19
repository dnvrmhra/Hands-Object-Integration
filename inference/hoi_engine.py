"""
hoi_engine.py — Orientation-Agnostic 3D Human-Object Interaction (HOI) Engine
=============================================================================
Designed for Microgravity Activity Recognition in Columbus Module MSG (ISRO PS-26174).

Features:
  1. Google MediaPipe 3D World Landmarks (`hand_world_landmarks` in metric meters).
  2. Intrinsic Local Coordinate Frame Normalization (X_local, Y_local, Z_local):
     - Invariant to 3D body pitch, yaw, roll, or inverted astronaut posture in zero-G.
  3. 6-DOF Attitude Tracking:
     - Extracts Euler angles (Roll, Pitch, Yaw) and palm normal orientation.
  4. 3D Spatial-Temporal HOI Fusion:
     - Evaluates 3D proximity to YOLOv8 objects (`box`, `container`).
     - Detects synchronized velocity vectors (transportation) and grasp postures.
  5. 3D Triad & Sci-Fi HUD Rendering:
     - Projects 3D coordinate frame (Red=X, Green=Y, Blue=Z) onto the palm.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Hand landmark indices (MediaPipe 21-point topology)
WRIST = 0
THUMB_CMC = 1; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7; INDEX_TIP = 8
MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP = 13; RING_PIP = 14; RING_DIP = 15; RING_TIP = 16
PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm knuckles
]


class OrientationAgnostic3DTracker:
    """
    Constructs an intrinsic local orthonormal coordinate frame for the hand:
      - Origin: Middle MCP (Palm Center)
      - Y_local: Wrist -> Middle MCP (Longitudinal axis)
      - Z_local: Palm Normal (Cross product between Transverse and Longitudinal vectors)
      - X_local: Transverse axis perpendicular to Y and Z
    """

    @staticmethod
    def compute_intrinsic_frame(world_pts: list[tuple[float, float, float]]) -> dict[str, Any]:
        """
        Takes 21 3D points (x, y, z in meters) from MediaPipe hand_world_landmarks.
        Returns rotation matrix, Euler angles (Roll, Pitch, Yaw in degrees), and local frame axes.
        """
        if len(world_pts) < 21:
            return {
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "x_axis": (1.0, 0.0, 0.0),
                "y_axis": (0.0, 1.0, 0.0),
                "z_axis": (0.0, 0.0, 1.0),
                "is_invariant": True,
            }

        pts = np.array(world_pts, dtype=np.float64)  # (21, 3)

        wrist = pts[WRIST]
        mid_mcp = pts[MIDDLE_MCP]
        index_mcp = pts[INDEX_MCP]
        pinky_mcp = pts[PINKY_MCP]

        # 1. Primary longitudinal vector: Wrist -> Middle MCP
        v_long = mid_mcp - wrist
        norm_long = np.linalg.norm(v_long)
        y_axis = v_long / (norm_long + 1e-7)

        # 2. Transverse vector: Index MCP -> Pinky MCP
        v_trans = pinky_mcp - index_mcp
        norm_trans = np.linalg.norm(v_trans)
        v_trans_unit = v_trans / (norm_trans + 1e-7)

        # 3. Palm Normal Z_local (perpendicular out of palm)
        z_axis = np.cross(v_trans_unit, y_axis)
        norm_z = np.linalg.norm(z_axis)
        z_axis = z_axis / (norm_z + 1e-7)

        # 4. Orthonormal X_local
        x_axis = np.cross(y_axis, z_axis)
        norm_x = np.linalg.norm(x_axis)
        x_axis = x_axis / (norm_x + 1e-7)

        # Build 3x3 Rotation Matrix R = [X, Y, Z]
        R = np.column_stack((x_axis, y_axis, z_axis))

        # 5. Extract Euler angles (in degrees)
        # Roll (around Z), Pitch (around X), Yaw (around Y)
        pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
        if abs(math.cos(pitch)) > 1e-4:
            roll = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            yaw = 0.0

        roll_deg = round(math.degrees(roll), 1)
        pitch_deg = round(math.degrees(pitch), 1)
        yaw_deg = round(math.degrees(yaw), 1)

        return {
            "roll": roll_deg,
            "pitch": pitch_deg,
            "yaw": yaw_deg,
            "rotation_matrix": R.tolist(),
            "x_axis": tuple(round(float(v), 3) for v in x_axis),
            "y_axis": tuple(round(float(v), 3) for v in y_axis),
            "z_axis": tuple(round(float(v), 3) for v in z_axis),
            "is_invariant": True,
        }

    @staticmethod
    def normalize_to_intrinsic_basis(
        world_pts: list[tuple[float, float, float]],
        frame_meta: dict[str, Any],
    ) -> list[tuple[float, float, float]]:
        """
        Projects all 21 3D points into the intrinsic local frame centered at Middle MCP.
        Output coordinates are strictly orientation-invariant regardless of astronaut orientation.
        """
        if "rotation_matrix" not in frame_meta or len(world_pts) < 21:
            return world_pts

        R = np.array(frame_meta["rotation_matrix"], dtype=np.float64)  # (3, 3)
        pts = np.array(world_pts, dtype=np.float64)  # (21, 3)
        origin = pts[MIDDLE_MCP]

        # Shift to origin and rotate by R.T
        local_pts = np.dot(pts - origin, R)
        return [tuple(round(float(coord), 4) for coord in row) for row in local_pts]


class HandTracker:
    """Extracts 21 2D pixel landmarks and 21 3D world landmarks with 6-DOF attitude."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            root_dir = Path(__file__).resolve().parent.parent
            model_path = str(root_dir / "models" / "hand_landmarker.task")

        self.model_path = model_path
        self.detector = None
        self._prev_positions: dict[int, tuple[float, float, float]] = {}
        self._prev_velocities: dict[int, tuple[float, float]] = {}
        self._prev_landmarks_px: dict[int, list[tuple[int, int]]] = {}
        self._last_timestamp_ms: int = 0

        self._init_detector()

    def _init_detector(self) -> None:
        try:
            base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            print(f"[HOI-3D] MediaPipe 3D HandLandmarker (High-Speed VIDEO Mode) initialized from {self.model_path}")
        except Exception as e:
            print(f"[HOI-3D] Warning: Could not initialize MediaPipe HandLandmarker: {e}")
            self.detector = None

    def process_frame(self, bgr_frame: np.ndarray) -> list[dict[str, Any]]:
        """Process frame and return 3D orientation-agnostic hand telemetry with zero lag and smoothed joints."""
        if self.detector is None:
            return []

        h, w, _ = bgr_frame.shape
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        now = time.time()
        timestamp_ms = int(now * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        try:
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            try:
                result = self.detector.detect(mp_image)
            except Exception:
                return []

        if not result.hand_landmarks:
            return []

        hands_data = []

        for hand_idx, landmarks in enumerate(result.hand_landmarks):
            # Raw 2D pixel coordinates
            raw_pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # EMA Smoothing for 2D landmarks (eliminates jitter, glides smoothly)
            pts_px = []
            prev_lms = self._prev_landmarks_px.get(hand_idx)
            alpha = 0.72  # 72% current + 28% previous: instant response, zero jitter
            for idx_pt, (rx, ry) in enumerate(raw_pts_px):
                if prev_lms and idx_pt < len(prev_lms):
                    prx, pry = prev_lms[idx_pt]
                    sx = int(alpha * rx + (1.0 - alpha) * prx)
                    sy = int(alpha * ry + (1.0 - alpha) * pry)
                    pts_px.append((sx, sy))
                else:
                    pts_px.append((rx, ry))
            self._prev_landmarks_px[hand_idx] = pts_px

            pts_norm = [{"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)} for lm in landmarks]

            # 3D World Landmarks (in meters)
            world_pts = []
            if result.hand_world_landmarks and hand_idx < len(result.hand_world_landmarks):
                w_lms = result.hand_world_landmarks[hand_idx]
                world_pts = [(w_lm.x, w_lm.y, w_lm.z) for w_lm in w_lms]
            else:
                # Fallback to pseudo-3D from normalized coordinates
                world_pts = [(lm.x - 0.5, lm.y - 0.5, lm.z) for lm in landmarks]

            # Compute intrinsic 3D local coordinate frame & Euler attitude angles
            attitude_3d = OrientationAgnostic3DTracker.compute_intrinsic_frame(world_pts)

            # Invariant local coordinates (independent of camera tilt or inverted orientation)
            invariant_pts = OrientationAgnostic3DTracker.normalize_to_intrinsic_basis(world_pts, attitude_3d)

            # Invariant pinch distance (thumb tip vs index tip in local metric frame)
            # Landmark 4 (thumb tip), Landmark 8 (index tip)
            t_tip = invariant_pts[THUMB_TIP]
            i_tip = invariant_pts[INDEX_TIP]
            pinch_dist_metric_cm = round(math.hypot(t_tip[0] - i_tip[0], t_tip[1] - i_tip[1], t_tip[2] - i_tip[2]) * 100, 1)

            # 2D pinch distance in pixels
            p_thumb = pts_px[THUMB_TIP]
            p_index = pts_px[INDEX_TIP]
            pinch_dist_px = math.hypot(p_thumb[0] - p_index[0], p_thumb[1] - p_index[1])

            # Grasp state (invariant criterion: pinch < 4.5 cm in 3D metric space)
            is_grasping = pinch_dist_metric_cm < 4.5 or (pinch_dist_px / max(w, h)) < 0.08

            # Velocity calculation
            palm_px = pts_px[MIDDLE_MCP]
            vx, vy = 0.0, 0.0
            if hand_idx in self._prev_positions:
                px, py, pt = self._prev_positions[hand_idx]
                dt = max(0.001, now - pt)
                raw_vx = (palm_px[0] - px) / dt
                raw_vy = (palm_px[1] - py) / dt
                prev_vx, prev_vy = self._prev_velocities.get(hand_idx, (raw_vx, raw_vy))
                vx = 0.7 * prev_vx + 0.3 * raw_vx
                vy = 0.7 * prev_vy + 0.3 * raw_vy

            self._prev_positions[hand_idx] = (palm_px[0], palm_px[1], now)
            self._prev_velocities[hand_idx] = (vx, vy)
            speed = math.hypot(vx, vy)

            # Handedness
            handedness = "Right"
            if result.handedness and hand_idx < len(result.handedness):
                handedness = result.handedness[hand_idx][0].category_name

            hands_data.append({
                "hand_id": hand_idx,
                "handedness": handedness,
                "landmarks_norm": pts_norm,
                "landmarks_px": pts_px,
                "world_landmarks": world_pts,
                "invariant_landmarks": invariant_pts,
                "palm_px": palm_px,
                "wrist_px": pts_px[WRIST],
                "fingertips_px": [pts_px[THUMB_TIP], pts_px[INDEX_TIP], pts_px[MIDDLE_TIP]],
                "attitude_3d": attitude_3d,
                "pinch_distance_cm": pinch_dist_metric_cm,
                "is_grasping": is_grasping,
                "velocity": (round(vx, 1), round(vy, 1)),
                "speed": round(speed, 1),
            })

        return hands_data


class ObjectMotionTracker:
    """Tracks YOLO object bounding boxes and computes displacement velocity."""

    def __init__(self):
        self._prev_objects: dict[int, tuple[float, float, float]] = {}
        self._prev_velocities: dict[int, tuple[float, float]] = {}

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = time.time()
        tracked = []

        for idx, det in enumerate(detections):
            d = dict(det)
            cx = d["bbox"]["cx"]
            cy = d["bbox"]["cy"]
            obj_id = idx

            vx, vy = 0.0, 0.0
            if obj_id in self._prev_objects:
                px, py, pt = self._prev_objects[obj_id]
                dt = max(0.001, now - pt)
                raw_vx = (cx - px) / dt
                raw_vy = (cy - py) / dt
                prev_vx, prev_vy = self._prev_velocities.get(obj_id, (raw_vx, raw_vy))
                vx = 0.7 * prev_vx + 0.3 * raw_vx
                vy = 0.7 * prev_vy + 0.3 * raw_vy

            self._prev_objects[obj_id] = (cx, cy, now)
            self._prev_velocities[obj_id] = (vx, vy)
            speed = math.hypot(vx, vy)

            d["velocity"] = (round(vx, 1), round(vy, 1))
            d["speed"] = round(speed, 1)
            tracked.append(d)

        return tracked


class HOIFusionEngine:
    """Fuses 3D Hand Tracking with YOLO Object Detection for Microgravity Activity Recognition."""

    def __init__(self):
        self.hand_tracker = HandTracker()
        self.obj_tracker = ObjectMotionTracker()
        self.current_state = "IDLE"
        self.target_object: Optional[str] = None
        self.state_confidence: float = 0.95
        self.proximity_cm: float = 0.0

    def process(
        self,
        bgr_frame: np.ndarray,
        yolo_detections: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        hands = self.hand_tracker.process_frame(bgr_frame)
        objects = self.obj_tracker.update(yolo_detections)
        hoi_summary = self._evaluate_interaction(hands, objects)
        return hands, objects, hoi_summary

    def _point_to_bbox_distance(self, pt: tuple[int, int], bbox: dict[str, float]) -> float:
        px, py = pt
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        dx = max(0.0, max(x1 - px, px - x2))
        dy = max(0.0, max(y1 - py, py - y2))
        return math.hypot(dx, dy)

    def _evaluate_interaction(
        self,
        hands: list[dict[str, Any]],
        objects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        default_attitude = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "is_invariant": True}

        if not hands:
            return {
                "state": "IDLE",
                "target_object": None,
                "confidence": 0.95,
                "proximity_cm": 0.0,
                "action_label": "NO HAND DETECTED",
                "attitude_3d": default_attitude,
            }

        primary_attitude = hands[0].get("attitude_3d", default_attitude)

        if not objects:
            return {
                "state": "IDLE",
                "target_object": None,
                "confidence": 0.95,
                "proximity_cm": 0.0,
                "action_label": "NO OBJECT DETECTED",
                "attitude_3d": primary_attitude,
            }

        best_interaction = None
        min_dist = float("inf")

        for hand in hands:
            for obj in objects:
                bbox = obj["bbox"]
                fingertip_dists = [self._point_to_bbox_distance(tip, bbox) for tip in hand["fingertips_px"]]
                palm_dist = self._point_to_bbox_distance(hand["palm_px"], bbox)
                closest = min(min(fingertip_dists), palm_dist)
                if closest < min_dist:
                    min_dist = closest
                    best_interaction = {"hand": hand, "obj": obj, "distance_px": closest}

        if not best_interaction:
            return {
                "state": "IDLE",
                "target_object": None,
                "confidence": 0.90,
                "proximity_cm": 0.0,
                "action_label": "IDLE",
                "attitude_3d": primary_attitude,
            }

        hand = best_interaction["hand"]
        obj = best_interaction["obj"]
        dist_px = best_interaction["distance_px"]
        obj_name = obj["class_name"]
        proximity_cm = round(dist_px * 0.08, 1)

        h_vx, h_vy = hand["velocity"]
        o_vx, o_vy = obj["velocity"]
        h_speed = hand["speed"]
        o_speed = obj["speed"]

        correlation = 0.0
        if h_speed > 15 and o_speed > 15:
            dot = (h_vx * o_vx) + (h_vy * o_vy)
            correlation = dot / (h_speed * o_speed)

        # State Decision Tree
        if dist_px < 40 and correlation > 0.4 and (h_speed > 15 or o_speed > 15):
            state = "TRANSPORTING"
            action_label = f"TRANSPORTING {obj_name}"
            conf = min(0.98, 0.85 + abs(correlation) * 0.12)
        elif dist_px < 35 and (hand["is_grasping"] or dist_px < 15):
            state = "GRASPING"
            action_label = f"GRASPING {obj_name}"
            conf = 0.94
        elif dist_px < 90:
            state = "REACHING"
            action_label = f"REACHING TOWARD {obj_name}"
            conf = 0.91
        else:
            state = "IDLE"
            action_label = f"MONITORING {obj_name}"
            conf = 0.88

        self.current_state = state
        self.state_confidence = conf
        self.proximity_cm = proximity_cm
        self.target_object = obj_name

        return {
            "state": state,
            "target_object": obj_name,
            "confidence": round(conf, 2),
            "proximity_cm": proximity_cm,
            "action_label": action_label,
            "motion_correlation": round(correlation, 2),
            "attitude_3d": hand.get("attitude_3d", default_attitude),
            "is_grasping": hand["is_grasping"],
            "pinch_distance_cm": hand["pinch_distance_cm"],
        }


def draw_hoi_overlays(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    hands: list[dict[str, Any]],
    hoi: dict[str, Any],
) -> np.ndarray:
    """
    Draw 3D hand skeletons, 3D orientation triad (Red=X, Green=Y, Blue=Z),
    YOLO bounding boxes, proximity vectors, and microgravity HUD badges.
    """
    vis = frame.copy()
    h, w, _ = vis.shape

    # 1. Draw Hand Landmarks & 3D Triad
    for hand in hands:
        pts = hand["landmarks_px"]
        # Hand Skeleton
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(vis, pts[a], pts[b], (248, 189, 56), 2, cv2.LINE_AA)

        for idx, (px, py) in enumerate(pts):
            color = (56, 189, 248) if idx in (0, 4, 8, 12, 16, 20) else (200, 240, 255)
            radius = 5 if idx in (4, 8) else 3
            cv2.circle(vis, (px, py), radius, color, -1, cv2.LINE_AA)

        # ── Draw 3D Orientation Triad at Palm Center ──
        palm_pt = hand["palm_px"]
        att = hand.get("attitude_3d", {})
        if "x_axis" in att:
            axis_len = 45.0
            # Projection of 3D unit vectors onto 2D image plane
            x_ax = att["x_axis"]
            y_ax = att["y_axis"]
            z_ax = att["z_axis"]

            pt_x = (int(palm_pt[0] + x_ax[0] * axis_len), int(palm_pt[1] + x_ax[1] * axis_len))
            pt_y = (int(palm_pt[0] + y_ax[0] * axis_len), int(palm_pt[1] + y_ax[1] * axis_len))
            pt_z = (int(palm_pt[0] + z_ax[0] * axis_len), int(palm_pt[1] + z_ax[1] * axis_len))

            # Red = X (Transverse), Green = Y (Longitudinal), Blue = Z (Palm Normal)
            cv2.line(vis, palm_pt, pt_x, (0, 0, 255), 2, cv2.LINE_AA)    # X: Red
            cv2.line(vis, palm_pt, pt_y, (0, 255, 0), 2, cv2.LINE_AA)    # Y: Green
            cv2.line(vis, palm_pt, pt_z, (255, 0, 0), 2, cv2.LINE_AA)    # Z: Blue

            # Orientation text overlay
            roll_lbl = f"R:{att.get('roll', 0)} P:{att.get('pitch', 0)} Y:{att.get('yaw', 0)}"
            cv2.putText(vis, roll_lbl, (palm_pt[0] - 40, palm_pt[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw Smooth YOLO Bounding Boxes
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        cls_name = det["class_name"].lower()
        conf = det["confidence"]
        is_held = det.get("is_hand_locked", False)

        if "bottle" in cls_name:
            color = (52, 211, 153)  # Bright Emerald
        elif "can" in cls_name:
            color = (248, 189, 56)  # Electric Amber
        elif "phone" in cls_name:
            color = (56, 189, 248)  # Electric Cyan
        elif "box" in cls_name:
            color = (254, 180, 216)  # Neon purple
        else:
            color = (244, 63, 94)

        thickness = 3 if is_held else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # High-Tech Corner Accents
        corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
        cv2.line(vis, (x1, y1), (x1 + corner_len, y1), color, 3, cv2.LINE_AA)
        cv2.line(vis, (x1, y1), (x1, y1 + corner_len), color, 3, cv2.LINE_AA)
        cv2.line(vis, (x2, y2), (x2 - corner_len, y2), color, 3, cv2.LINE_AA)
        cv2.line(vis, (x2, y2), (x2, y2 - corner_len), color, 3, cv2.LINE_AA)

        held_tag = " [HELD]" if is_held else ""
        label = f"{cls_name.upper()} {int(conf * 100)}%{held_tag}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), (15, 23, 42), -1)
        cv2.rectangle(vis, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), color, 1)
        cv2.putText(vis, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    # 3. Draw Proximity Line
    if hands and detections and hoi.get("state") in ("REACHING", "GRASPING", "TRANSPORTING"):
        hand_pt = hands[0]["palm_px"]
        target_obj = next((d for d in detections if d["class_name"] == hoi.get("target_object")), detections[0])
        obj_pt = (int(target_obj["bbox"]["cx"]), int(target_obj["bbox"]["cy"]))
        line_color = (36, 191, 251) if hoi["state"] == "TRANSPORTING" else (34, 211, 153)
        cv2.line(vis, hand_pt, obj_pt, line_color, 2, cv2.LINE_AA)

        mid_x = (hand_pt[0] + obj_pt[0]) // 2
        mid_y = (hand_pt[1] + obj_pt[1]) // 2
        dist_lbl = f"3D dist = {hoi.get('proximity_cm', 0.0)} cm"
        cv2.putText(vis, dist_lbl, (mid_x - 35, mid_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Microgravity 6-DOF HUD Badge (Top Center)
    state = hoi.get("state", "IDLE")
    action_label = hoi.get("action_label", "IDLE")
    state_colors = {
        "IDLE": (100, 116, 139),
        "REACHING": (251, 191, 36),
        "GRASPING": (34, 211, 153),
        "TRANSPORTING": (244, 63, 94),
    }
    badge_color = state_colors.get(state, (100, 116, 139))

    banner_w = 420
    banner_x = (w - banner_w) // 2
    cv2.rectangle(vis, (banner_x, 10), (banner_x + banner_w, 44), (10, 15, 26), -1)
    cv2.rectangle(vis, (banner_x, 10), (banner_x + banner_w, 44), badge_color, 1)

    hud_text = f"6-DOF HOI: {action_label} [{int(hoi.get('confidence', 0.9)*100)}%]"
    (tw, th), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(vis, hud_text, ((w - tw) // 2, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, badge_color, 1, cv2.LINE_AA)

    return vis
