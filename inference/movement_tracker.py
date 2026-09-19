"""
movement_tracker.py — Can & Bottle Movement & Relocation Tracking Engine
========================================================================
Detects when a person's hand reaches for, grasps, transports, and places down
cans and bottles, tracking the continuous physical trajectory and distance moved (in cm).

States:
  - AT_REST: Can/bottle resting at starting position A.
  - GRASPED: Hand closed on can/bottle.
  - TRANSPORTING: Hand actively moving can/bottle across the frame (trajectory vector active).
  - PLACED: Hand released can/bottle at destination B. Relocation event logged.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import cv2
import numpy as np

# Target object classes to track for movement
CAN_BOTTLE_CLASSES = {"bottle", "can", "cup", "container", "box", "phone", "cell phone"}


class CanBottleMovementTracker:
    def __init__(self):
        # Current active movement state
        self.state: str = "AT_REST"  # AT_REST | GRASPED | TRANSPORTING | PLACED
        self.active_object_name: Optional[str] = None
        self.start_point: Optional[tuple[int, int]] = None
        self.current_point: Optional[tuple[int, int]] = None
        self.end_point: Optional[tuple[int, int]] = None
        self.pickup_time: float = 0.0
        self.place_time: float = 0.0
        self.total_distance_cm: float = 0.0
        self.current_speed_cm_s: float = 0.0

        # Flash timer for placement confirmation
        self.placed_flash_until: float = 0.0

        # Event history
        self.movement_history: list[dict[str, Any]] = []
        self.event_counter: int = 0

        # Internal tracking memory
        self._last_obj_pos: Optional[tuple[int, int]] = None
        self._last_timestamp: float = time.time()
        self._stationary_since: float = time.time()

    def update(
        self,
        detections: list[dict[str, Any]],
        hands: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Update state machine with current frame's detections and hands.
        Returns movement telemetry summary dictionary.
        """
        now = time.time()
        dt = max(0.001, now - self._last_timestamp)

        # Filter detections for cans, bottles, and containers
        target_objs = [
            d for d in detections
            if any(c in d["class_name"].lower() for c in CAN_BOTTLE_CLASSES)
        ]

        # Determine primary target object (highest confidence)
        primary_obj = target_objs[0] if target_objs else None
        primary_hand = hands[0] if hands else None

        if primary_obj is not None:
            obj_cx = int(primary_obj["bbox"]["cx"])
            obj_cy = int(primary_obj["bbox"]["cy"])
            obj_name = primary_obj["class_name"].upper()
            if "CUP" in obj_name:
                obj_name = "CAN"
            self.current_point = (obj_cx, obj_cy)

            # Compute object displacement speed
            if self._last_obj_pos is not None:
                dx = obj_cx - self._last_obj_pos[0]
                dy = obj_cy - self._last_obj_pos[1]
                disp_px = math.hypot(dx, dy)
                speed_px_s = disp_px / dt
                self.current_speed_cm_s = round((speed_px_s * 0.08), 1)  # 1 px ≈ 0.08 cm

                if disp_px > 6:
                    self._stationary_since = now
            else:
                self.current_speed_cm_s = 0.0

            self._last_obj_pos = (obj_cx, obj_cy)

            # Measure distance from hand palm to can/bottle bounding box perimeter
            hand_dist = float("inf")
            is_hand_grasping = False
            if primary_hand is not None:
                palm_x, palm_y = primary_hand["palm_px"]
                b = primary_obj["bbox"]
                dx = max(0.0, b["x1"] - palm_x, palm_x - b["x2"])
                dy = max(0.0, b["y1"] - palm_y, palm_y - b["y2"])
                hand_dist = math.hypot(dx, dy)
                is_hand_grasping = primary_hand.get("is_grasping", False) or hand_dist < 35

            # ── State Machine Logic ──
            if self.state in ("AT_REST", "PLACED"):
                # Transition to GRASPED when hand comes in contact
                if hand_dist < 50 and is_hand_grasping:
                    self.state = "GRASPED"
                    self.active_object_name = obj_name
                    self.start_point = (obj_cx, obj_cy)
                    self.pickup_time = now

            elif self.state == "GRASPED":
                # Transition to TRANSPORTING if displacement begins
                if self.start_point is not None:
                    dist_from_start = math.hypot(obj_cx - self.start_point[0], obj_cy - self.start_point[1])
                    if dist_from_start > 15:
                        self.state = "TRANSPORTING"
                        self.total_distance_cm = round(dist_from_start * 0.08, 1)

            elif self.state == "TRANSPORTING":
                # Actively moving
                if self.start_point is not None:
                    dist_from_start = math.hypot(obj_cx - self.start_point[0], obj_cy - self.start_point[1])
                    self.total_distance_cm = round(dist_from_start * 0.08, 1)

                # Check for placement: hand released or moved away, or object stationary for > 0.8s
                is_released = hand_dist > 75 or (not is_hand_grasping and hand_dist > 45)
                is_settled = (now - self._stationary_since) > 0.8 and self.total_distance_cm > 5.0

                if is_released or is_settled:
                    self.state = "PLACED"
                    self.end_point = (obj_cx, obj_cy)
                    self.place_time = now
                    duration = round(max(0.5, self.place_time - self.pickup_time), 1)
                    self.placed_flash_until = now + 4.0  # flash confirmation badge for 4 seconds

                    # Log completed relocation event
                    self.event_counter += 1
                    event_data = {
                        "id": self.event_counter,
                        "object_name": self.active_object_name or "CAN/BOTTLE",
                        "start_pos": self.start_point,
                        "end_pos": self.end_point,
                        "distance_cm": self.total_distance_cm,
                        "duration_s": duration,
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                    self.movement_history.insert(0, event_data)
                    # Keep latest 10 events
                    if len(self.movement_history) > 10:
                        self.movement_history.pop()

        else:
            # No can or bottle detected in frame
            if now > self.placed_flash_until and self.state == "PLACED":
                self.state = "AT_REST"
            self.current_speed_cm_s = 0.0

        self._last_timestamp = now

        return {
            "state": self.state,
            "active_object": self.active_object_name,
            "start_point": self.start_point,
            "current_point": self.current_point,
            "end_point": self.end_point,
            "distance_cm": self.total_distance_cm,
            "speed_cm_s": self.current_speed_cm_s,
            "is_placed_recently": now < self.placed_flash_until,
            "history": self.movement_history,
        }

    def draw_movement_overlay(self, vis: np.ndarray) -> np.ndarray:
        """Render trajectory lines, distance badges, and relocation alerts onto the frame."""
        h, w, _ = vis.shape
        now = time.time()

        # 1. Draw Active Trajectory Vector (Start -> Current)
        if self.state == "TRANSPORTING" and self.start_point and self.current_point:
            sx, sy = self.start_point
            cx, cy = self.current_point

            # Dotted/dashed trajectory line
            cv2.line(vis, (sx, sy), (cx, cy), (0, 240, 255), 3, cv2.LINE_AA)  # Yellow/Gold vector
            cv2.circle(vis, (sx, sy), 7, (0, 165, 255), -1, cv2.LINE_AA)     # Start anchor (Orange)
            cv2.putText(vis, "ORIGIN A", (sx - 25, sy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

            # Arrow head pointing to current position
            cv2.arrowedLine(vis, (sx, sy), (cx, cy), (0, 255, 255), 3, tipLength=0.15)

            # Trajectory distance & speed badge at midpoint
            mx = (sx + cx) // 2
            my = (sy + cy) // 2
            lbl = f"MOVING: {self.total_distance_cm} cm | {self.current_speed_cm_s} cm/s"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (mx - tw // 2 - 6, my - 20), (mx + tw // 2 + 6, my + 6), (10, 15, 26), -1)
            cv2.rectangle(vis, (mx - tw // 2 - 6, my - 20), (mx + tw // 2 + 6, my + 6), (0, 240, 255), 1)
            cv2.putText(vis, lbl, (mx - tw // 2, my - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1, cv2.LINE_AA)

        # 2. Draw Placement Confirmation (Origin A -> Destination B)
        elif now < self.placed_flash_until and self.start_point and self.end_point:
            sx, sy = self.start_point
            ex, ey = self.end_point

            # Completed green trajectory line
            cv2.line(vis, (sx, sy), (ex, ey), (50, 255, 120), 2, cv2.LINE_AA)
            cv2.circle(vis, (sx, sy), 6, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (ex, ey), 8, (50, 255, 120), -1, cv2.LINE_AA)
            cv2.putText(vis, "A", (sx - 8, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, "B (PLACED)", (ex - 20, ey - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 120), 1, cv2.LINE_AA)

            # Big success banner at bottom
            confirm_txt = f"SUCCESS: {self.active_object_name} RELOCATED +{self.total_distance_cm} cm"
            (cw, ch), _ = cv2.getTextSize(confirm_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bx = (w - cw) // 2
            cv2.rectangle(vis, (bx - 10, h - 50), (bx + cw + 10, h - 14), (10, 30, 20), -1)
            cv2.rectangle(vis, (bx - 10, h - 50), (bx + cw + 10, h - 14), (50, 255, 120), 2)
            cv2.putText(vis, confirm_txt, (bx, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 120), 2, cv2.LINE_AA)

        # 3. Top Action Banner
        banner_w = 440
        banner_x = (w - banner_w) // 2
        if self.state == "TRANSPORTING":
            status_text = f"ACTION: MOVING {self.active_object_name} ({self.total_distance_cm} cm)"
            color = (0, 215, 255)  # Gold/Yellow
        elif self.state == "GRASPED":
            status_text = f"ACTION: GRASPED {self.active_object_name}"
            color = (34, 211, 153)  # Emerald
        elif now < self.placed_flash_until:
            status_text = f"ACTION: {self.active_object_name} PLACED AT NEW POSITION"
            color = (50, 255, 120)  # Green
        else:
            status_text = "ACTION: MONITORING CAN & BOTTLE POSITIONS"
            color = (148, 163, 184)  # Slate

        cv2.rectangle(vis, (banner_x, 10), (banner_x + banner_w, 42), (10, 15, 26), -1)
        cv2.rectangle(vis, (banner_x, 10), (banner_x + banner_w, 42), color, 1)
        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(vis, status_text, ((w - tw) // 2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        return vis
