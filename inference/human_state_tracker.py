"""
human_state_tracker.py — Human Activity & Motion State Engine (IDLE, MOVING, WALKING)
======================================================================================
Tracks detected humans across frames, computes centroid kinematics, and classifies state:
  - IDLE: Stationary posture (seated or standing still, velocity < 12 px/s).
  - MOVING: Local motion (gesturing, shifting in place, reaching, 12 <= velocity < 40 px/s).
  - WALKING: Horizontal/directional locomotion across frame (velocity >= 40 px/s with displacement).
"""

from __future__ import annotations

import collections
import math
import time
from typing import Any, Optional

import cv2
import numpy as np


class HumanTrack:
    def __init__(self, track_id: int, box: tuple[float, float, float, float], conf: float):
        self.track_id = track_id
        self.box = list(box)  # [x1, y1, x2, y2]
        self.confidence = conf
        self.history: collections.deque = collections.deque(maxlen=20)  # (cx, cy, timestamp)
        self.missing_count = 0
        self.hit_count = 1

        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.displacement = 0.0
        self.state = "IDLE"

        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        self.history.append((cx, cy, time.time()))

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0

    @property
    def w(self) -> float:
        return max(1.0, self.box[2] - self.box[0])

    @property
    def h(self) -> float:
        return max(1.0, self.box[3] - self.box[1])

    def update_box(self, new_box: tuple[float, float, float, float], conf: float, alpha: float = 0.65):
        # Coordinate EMA smoothing
        self.box[0] = alpha * new_box[0] + (1 - alpha) * self.box[0]
        self.box[1] = alpha * new_box[1] + (1 - alpha) * self.box[1]
        self.box[2] = alpha * new_box[2] + (1 - alpha) * self.box[2]
        self.box[3] = alpha * new_box[3] + (1 - alpha) * self.box[3]
        self.confidence = 0.6 * self.confidence + 0.4 * conf
        self.hit_count += 1
        self.missing_count = 0

        now = time.time()
        self.history.append((self.cx, self.cy, now))
        self._evaluate_kinematics()

    def _evaluate_kinematics(self):
        if len(self.history) < 3:
            self.state = "IDLE"
            return

        oldest_cx, oldest_cy, oldest_t = self.history[0]
        newest_cx, newest_cy, newest_t = self.history[-1]

        dt = max(0.001, newest_t - oldest_t)
        dx = newest_cx - oldest_cx
        dy = newest_cy - oldest_cy

        raw_vx = dx / dt
        raw_vy = dy / dt
        self.vx = 0.7 * self.vx + 0.3 * raw_vx
        self.vy = 0.7 * self.vy + 0.3 * raw_vy

        self.speed = math.hypot(self.vx, self.vy)
        self.displacement = math.hypot(dx, dy)

        # State Decision Tree:
        # Walking: Significant displacement & speed >= 40 px/s
        if self.speed >= 40.0 and self.displacement >= 30.0:
            self.state = "WALKING"
        elif self.speed >= 12.0 or self.displacement >= 12.0:
            self.state = "MOVING"
        else:
            self.state = "IDLE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": 3,
            "class_name": "person",
            "state": self.state,
            "speed_px_s": round(self.speed, 1),
            "displacement_px": round(self.displacement, 1),
            "velocity": (round(self.vx, 1), round(self.vy, 1)),
            "confidence": round(float(self.confidence), 2),
            "bbox": {
                "x1": round(float(self.box[0]), 1),
                "y1": round(float(self.box[1]), 1),
                "x2": round(float(self.box[2]), 1),
                "y2": round(float(self.box[3]), 1),
                "cx": round(float(self.cx), 1),
                "cy": round(float(self.cy), 1),
                "w": round(float(self.w), 1),
                "h": round(float(self.h), 1),
            },
        }


class HumanStateTracker:
    def __init__(self, max_missing_frames: int = 8, min_hits_to_show: int = 2):
        self.tracks: dict[int, HumanTrack] = {}
        self.next_id: int = 1
        self.max_missing_frames = max_missing_frames
        self.min_hits_to_show = min_hits_to_show

    def update(self, person_detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Matches fresh person bounding boxes to tracks and returns active human states."""
        # Step 0: Non-Maximum Suppression (NMS) to eliminate duplicate boxes on same person
        deduped: list[dict[str, Any]] = []
        sorted_dets = sorted(person_detections, key=lambda d: float(d["confidence"]), reverse=True)
        for det in sorted_dets:
            b1 = det["bbox"]
            box1 = (b1["x1"], b1["y1"], b1["x2"], b1["y2"])
            overlap = False
            for accepted in deduped:
                b2 = accepted["bbox"]
                box2 = (b2["x1"], b2["y1"], b2["x2"], b2["y2"])
                ix1 = max(box1[0], box2[0])
                iy1 = max(box1[1], box2[1])
                ix2 = min(box1[2], box2[2])
                iy2 = min(box1[3], box2[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter)
                iou = inter / union
                if iou > 0.35:
                    overlap = True
                    break
            if not overlap:
                deduped.append(det)

        person_detections = deduped
        matched_tracks = set()
        matched_dets = set()

        # Step 1: Match detections by IoU and centroid distance
        for tid, trk in list(self.tracks.items()):
            best_score = -1.0
            best_idx = -1
            trk_box = trk.box

            for i, det in enumerate(person_detections):
                if i in matched_dets:
                    continue
                b = det["bbox"]
                b_box = [b["x1"], b["y1"], b["x2"], b["y2"]]

                # IoU
                ix1, iy1 = max(trk_box[0], b_box[0]), max(trk_box[1], b_box[1])
                ix2, iy2 = min(trk_box[2], b_box[2]), min(trk_box[3], b_box[3])
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                union = trk.w * trk.h + b["w"] * b["h"] - inter
                iou = inter / union if union > 0 else 0.0

                dist = math.hypot(trk.cx - b["cx"], trk.cy - b["cy"])
                if iou > 0.20 or dist < 120:
                    score = iou + (1.0 - min(dist, 120) / 120.0)
                    if score > best_score:
                        best_score = score
                        best_idx = i

            if best_idx >= 0:
                matched_tracks.add(tid)
                matched_dets.add(best_idx)
                d = person_detections[best_idx]
                b = d["bbox"]
                trk.update_box((b["x1"], b["y1"], b["x2"], b["y2"]), d["confidence"])

        # Step 2: Handle unmatched tracks
        for tid, trk in list(self.tracks.items()):
            if tid not in matched_tracks:
                trk.missing_count += 1
                if trk.missing_count > self.max_missing_frames:
                    del self.tracks[tid]

        # Step 3: Register new tracks (limit to top 2 humans in scene)
        for i, det in enumerate(person_detections):
            if i not in matched_dets and len(self.tracks) < 2:
                b = det["bbox"]
                new_trk = HumanTrack(
                    track_id=self.next_id,
                    box=(b["x1"], b["y1"], b["x2"], b["y2"]),
                    conf=det["confidence"],
                )
                self.tracks[self.next_id] = new_trk
                self.next_id += 1

        # Return confirmed tracks
        output = []
        for trk in self.tracks.values():
            if trk.hit_count >= self.min_hits_to_show:
                output.append(trk.to_dict())

        return output

    def get_current_humans(self) -> list[dict[str, Any]]:
        """Fast 30 FPS getter returning active confirmed humans with smoothed kinematics."""
        output = []
        for trk in self.tracks.values():
            if trk.hit_count >= self.min_hits_to_show:
                output.append(trk.to_dict())
        return output

    def draw_human_overlays(self, vis: np.ndarray, humans: list[dict[str, Any]]) -> np.ndarray:
        """Renders bounding box with confidence score, state badge (IDLE, MOVING, WALKING), and motion vectors."""
        vh, vw = vis.shape[:2]
        for h in humans:
            bbox = h["bbox"]
            x1 = max(0, min(vw - 1, int(bbox["x1"])))
            y1 = max(0, min(vh - 1, int(bbox["y1"])))
            x2 = max(0, min(vw - 1, int(bbox["x2"])))
            y2 = max(0, min(vh - 1, int(bbox["y2"])))
            state = h.get("state", "IDLE")
            speed = h.get("speed_px_s", 0.0)
            conf = h.get("confidence", 0.0)
            conf_pct = f"{int(conf * 100)}%" if conf > 0 else ""

            if state == "WALKING":
                color = (52, 211, 153)   # Bright emerald
                badge_text = f"PERSON {conf_pct} [WALKING: {int(speed)} px/s]"
            elif state == "MOVING":
                color = (36, 191, 251)   # Bright amber
                badge_text = f"PERSON {conf_pct} [MOVING]"
            else:
                color = (200, 200, 210)   # Cool silver/slate
                badge_text = f"PERSON {conf_pct} [IDLE]"

            # Draw sleek box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw corner accents
            clen = min(20, max(2, (x2 - x1) // 4), max(2, (y2 - y1) // 4))
            cv2.line(vis, (x1, y1), (x1 + clen, y1), color, 3, cv2.LINE_AA)
            cv2.line(vis, (x1, y1), (x1, y1 + clen), color, 3, cv2.LINE_AA)
            cv2.line(vis, (x2, y2), (x2 - clen, y2), color, 3, cv2.LINE_AA)
            cv2.line(vis, (x2, y2), (x2, y2 - clen), color, 3, cv2.LINE_AA)

            # Badge pill
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            badge_y1 = max(0, y1 - 24)
            badge_y2 = y1 if y1 >= 24 else min(vh - 1, y1 + 24)
            cv2.rectangle(vis, (x1, badge_y1), (min(vw - 1, x1 + tw + 12), badge_y2), (15, 23, 42), -1)
            cv2.rectangle(vis, (x1, badge_y1), (min(vw - 1, x1 + tw + 12), badge_y2), color, 1)
            text_y = y1 - 7 if y1 >= 24 else y1 + 17
            cv2.putText(vis, badge_text, (x1 + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

            # If walking, draw velocity vector arrow
            if state == "WALKING":
                vx, vy = h.get("velocity", (0.0, 0.0))
                vcx, vcy = int(bbox["cx"]), int(bbox["cy"])
                end_x = int(vcx + np.clip(vx * 0.4, -60, 60))
                end_y = int(vcy + np.clip(vy * 0.4, -60, 60))
                cv2.arrowedLine(vis, (vcx, vcy), (end_x, end_y), (52, 211, 153), 2, tipLength=0.3)

        return vis
