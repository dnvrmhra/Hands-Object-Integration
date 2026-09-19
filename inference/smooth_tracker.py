"""
smooth_tracker.py — Deduplicated, Anti-Jitter Track Stabilizer for Cans & Bottles
================================================================================
Key Fixes:
  1. Strict Track Deduplication & NMS (never shows multiple boxes on the same bottle/can)
  2. Hit-Count Confirmation (requires 2 consecutive detections before displaying a box)
  3. Strict Centroid & IoU Clustering (merges any overlapping detections on the same object)
  4. Smooth Coordinate EMA (butter-smooth movement without jumping or jitter)
  5. Hand-Anchored Kinematics (moves with hand during transport)
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional


def compute_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """Computes Intersection-over-Union between two boxes [x1, y1, x2, y2]."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    a1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    a2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def is_box_on_hand(
    box: tuple[float, float, float, float] | list[float],
    hands_data: Optional[list[dict[str, Any]]],
    max_dist: float = 75.0,
    pad: float = 40.0,
) -> bool:
    """
    Validates whether an object bounding box is directly touching, overlapping,
    or held by any active hand. Strictly eliminates false positive background boxes.
    """
    if not hands_data:
        return False

    bx1, by1, bx2, by2 = box[0], box[1], box[2], box[3]
    padded_x1 = bx1 - pad
    padded_y1 = by1 - pad
    padded_x2 = bx2 + pad
    padded_y2 = by2 + pad

    for hand in hands_data:
        # 1. Direct landmark intersection check (fingertips, palm, knuckles)
        landmarks = hand.get("landmarks_px", [])
        for px, py in landmarks:
            if padded_x1 <= px <= padded_x2 and padded_y1 <= py <= padded_y2:
                return True

        # 2. Palm proximity to perimeter of bounding box
        palm = hand.get("palm_px")
        if palm:
            dx = max(0.0, bx1 - palm[0], palm[0] - bx2)
            dy = max(0.0, by1 - palm[1], palm[1] - by2)
            if math.hypot(dx, dy) <= max_dist:
                return True

        # 3. Hand bounding envelope overlap
        if landmarks:
            hx1 = min(p[0] for p in landmarks)
            hx2 = max(p[0] for p in landmarks)
            hy1 = min(p[1] for p in landmarks)
            hy2 = max(p[1] for p in landmarks)
            if max(bx1, hx1) < min(bx2, hx2) and max(by1, hy1) < min(by2, hy2):
                return True

    return False


class TrackedObject:
    def __init__(self, track_id: int, class_name: str, box: tuple[float, float, float, float], conf: float):
        self.track_id = track_id
        self.class_name = class_name
        self.class_history: list[tuple[str, float]] = [(class_name, float(conf))]
        self.box = list(box)  # [x1, y1, x2, y2]
        self.confidence = conf
        self.hit_count = 1
        self.missing_count = 0
        self.is_hand_locked = False
        self.hand_offset = (0.0, 0.0)

    def add_class_observation(self, cls_name: str, conf: float = 0.5) -> None:
        """
        Confidence-weighted voting across recent detections.
        Decisive instant transition when a new object arrives with strong confidence (>=0.65)
        or 2 consecutive frames agree, while ignoring sporadic 1-frame noise.
        """
        self.class_history.append((cls_name, float(conf)))
        if len(self.class_history) > 4:
            self.class_history.pop(0)

        # Immediate switch on strong confident detection or 2 consecutive frames of new class
        recent_matches = [c for c, _ in self.class_history[-2:] if c == cls_name]
        if cls_name != self.class_name and (float(conf) >= 0.45 or len(recent_matches) >= 2):
            self.class_name = cls_name
            self.class_history = [(cls_name, float(conf))]
            return

        weights: dict[str, float] = {}
        for c, w in self.class_history:
            weights[c] = weights.get(c, 0.0) + w

        current_weight = weights.get(self.class_name, 0.0)
        best_cls, best_weight = max(weights.items(), key=lambda item: item[1])

        if best_cls != self.class_name and best_weight > current_weight * 1.15:
            self.class_name = best_cls

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

    def to_dict(self) -> dict[str, Any]:
        cls_map = {"bottle": 0, "can": 1, "phone": 2, "person": 3}
        return {
            "track_id": self.track_id,
            "class_id": cls_map.get(self.class_name, 0),
            "class_name": self.class_name,
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
            "is_hand_locked": self.is_hand_locked,
        }


class SmoothBoxTracker:
    def __init__(
        self,
        alpha_coord: float = 0.65,
        max_missing_frames: int = 12,
        min_hits_to_show: int = 2,
        max_tracks: int = 6,
    ):
        self.alpha_coord = alpha_coord
        self.max_missing_frames = max_missing_frames
        self.min_hits_to_show = min_hits_to_show
        self.max_tracks = max_tracks
        self.tracks: dict[int, TrackedObject] = {}
        self.next_id: int = 1

    def update(
        self,
        raw_detections: list[dict[str, Any]],
        hands_data: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        # 1. Class-aware deduplication of raw incoming detections
        deduped_raw: list[dict[str, Any]] = []
        sorted_raw = sorted(raw_detections, key=lambda d: float(d["confidence"]), reverse=True)
        for d in sorted_raw:
            b = d["bbox"]
            box_tuple = (b["x1"], b["y1"], b["x2"], b["y2"])
            overlap = False
            for accepted in deduped_raw:
                ab = accepted["bbox"]
                ab_tuple = (ab["x1"], ab["y1"], ab["x2"], ab["y2"])
                same_cls = (d["class_name"] == accepted["class_name"])
                # Discard duplicate overlapping boxes
                # If same class, merge IoU > 0.45 or centroid distance < 35px
                # If different class, suppress lower-confidence candidate if IoU > 0.50 or centroid distance < 25px
                iou_thresh = 0.45 if same_cls else 0.50
                dist_thresh = 35 if same_cls else 25
                if compute_iou(box_tuple, ab_tuple) > iou_thresh:
                    overlap = True
                    break
                if math.hypot(b["cx"] - ab["cx"], b["cy"] - ab["cy"]) < dist_thresh:
                    overlap = True
                    break
            if not overlap:
                deduped_raw.append(d)

        # 2. Extract hand state for kinematic anchoring
        palm_pos = None
        hand_grasping = False
        if hands_data:
            for h in hands_data:
                if h.get("is_grasping", False):
                    palm_pos = h.get("palm_px")
                    hand_grasping = True
                    break
            if palm_pos is None and len(hands_data) > 0:
                palm_pos = hands_data[0].get("palm_px")

        # 3. Match deduped detections to existing tracks
        matched_tracks = set()
        matched_detections = set()

        for tid, trk in list(self.tracks.items()):
            best_iou = 0.0
            best_idx = -1
            trk_box = (trk.box[0], trk.box[1], trk.box[2], trk.box[3])

            for i, det in enumerate(deduped_raw):
                if i in matched_detections:
                    continue
                b = det["bbox"]
                b_box = (b["x1"], b["y1"], b["x2"], b["y2"])
                iou = compute_iou(trk_box, b_box)
                dist = math.hypot(trk.cx - b["cx"], trk.cy - b["cy"])

                if iou > 0.15 or dist < 80:
                    score = iou + (1.0 - min(dist, 80) / 80.0)
                    if score > best_iou:
                        best_iou = score
                        best_idx = i

            if best_idx >= 0:
                matched_tracks.add(tid)
                matched_detections.add(best_idx)
                d = deduped_raw[best_idx]
                b = d["bbox"]

                # Adaptive coordinate smoothing (eliminates micro-jitter when holding still, tracks fast when moving)
                shift = math.hypot(trk.cx - b["cx"], trk.cy - b["cy"])
                if shift < 12.0:
                    a = 0.35  # Butter-smooth when stationary in hand
                elif shift > 40.0:
                    a = 0.70  # Low-latency tracking during fast motions
                else:
                    a = 0.50

                trk.box[0] = a * b["x1"] + (1 - a) * trk.box[0]
                trk.box[1] = a * b["y1"] + (1 - a) * trk.box[1]
                trk.box[2] = a * b["x2"] + (1 - a) * trk.box[2]
                trk.box[3] = a * b["y2"] + (1 - a) * trk.box[3]

                trk.confidence = 0.6 * trk.confidence + 0.4 * d["confidence"]
                trk.add_class_observation(d["class_name"], d["confidence"])
                trk.hit_count += 1
                trk.missing_count = 0

                # Hand attachment: check proximity to box perimeter
                if palm_pos:
                    dx = max(0.0, trk.box[0] - palm_pos[0], palm_pos[0] - trk.box[2])
                    dy = max(0.0, trk.box[1] - palm_pos[1], palm_pos[1] - trk.box[3])
                    dist_to_palm = math.hypot(dx, dy)
                    if dist_to_palm < 65 and hand_grasping:
                        trk.is_hand_locked = True
                        trk.hand_offset = (trk.cx - palm_pos[0], trk.cy - palm_pos[1])
                    elif not hand_grasping or dist_to_palm > 100:
                        trk.is_hand_locked = False

        # 4. Handle unmatched tracks (occlusion or out of frame)
        for tid, trk in list(self.tracks.items()):
            if tid not in matched_tracks:
                trk.missing_count += 1
                # If locked to hand, carry position with hand and keep alive
                if trk.is_hand_locked and palm_pos and hand_grasping:
                    target_cx = palm_pos[0] + trk.hand_offset[0]
                    target_cy = palm_pos[1] + trk.hand_offset[1]
                    hw = trk.w / 2.0
                    hh = trk.h / 2.0
                    trk.box[0] = target_cx - hw
                    trk.box[1] = target_cy - hh
                    trk.box[2] = target_cx + hw
                    trk.box[3] = target_cy + hh
                    trk.missing_count = 0  # Hold active during grasp
                else:
                    if not hand_grasping:
                        trk.is_hand_locked = False

                if trk.missing_count > self.max_missing_frames:
                    del self.tracks[tid]

        # 5. Register new tracks (only if in/on active hand)
        for i, det in enumerate(deduped_raw):
            if i not in matched_detections and len(self.tracks) < self.max_tracks:
                b = det["bbox"]
                box_coords = (b["x1"], b["y1"], b["x2"], b["y2"])
                if not is_box_on_hand(box_coords, hands_data, max_dist=75.0):
                    continue
                new_trk = TrackedObject(
                    track_id=self.next_id,
                    class_name=det["class_name"],
                    box=box_coords,
                    conf=det["confidence"],
                )
                self.tracks[self.next_id] = new_trk
                self.next_id += 1

        # 6. Final deduplication between active tracks (ensure no two tracks overlap)
        active_tracks = list(self.tracks.values())
        to_delete = set()
        for i in range(len(active_tracks)):
            for j in range(i + 1, len(active_tracks)):
                t1 = active_tracks[i]
                t2 = active_tracks[j]
                b1 = (t1.box[0], t1.box[1], t1.box[2], t1.box[3])
                b2 = (t2.box[0], t2.box[1], t2.box[2], t2.box[3])
                if compute_iou(b1, b2) > 0.45 or math.hypot(t1.cx - t2.cx, t1.cy - t2.cy) < 35:
                    # Remove the one with fewer hits or lower confidence
                    if t1.hit_count >= t2.hit_count:
                        to_delete.add(t2.track_id)
                    else:
                        to_delete.add(t1.track_id)

        for tid in to_delete:
            if tid in self.tracks:
                del self.tracks[tid]

        # Return only confirmed tracks that are on/near an active hand
        output = []
        for trk in list(self.tracks.values()):
            if not is_box_on_hand(trk.box, hands_data, max_dist=75.0):
                continue
            if trk.hit_count >= self.min_hits_to_show or trk.is_hand_locked:
                output.append(trk.to_dict())

        return output

    def get_current_boxes(self, hands_data: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
        """
        Fast 30 FPS query. Strictly requires objects to be on/held by an active hand.
        Eliminates any background boxes when hands are not holding/touching the object.
        """
        if not hands_data:
            return []

        palm_pos = None
        hand_grasping = False
        for h in hands_data:
            if h.get("is_grasping", False):
                palm_pos = h.get("palm_px")
                hand_grasping = True
                break
        if palm_pos is None and len(hands_data) > 0:
            palm_pos = hands_data[0].get("palm_px")

        output = []
        for trk in list(self.tracks.values()):
            # Strictly verify object is still touching or held by an active hand
            if not is_box_on_hand(trk.box, hands_data, max_dist=75.0):
                trk.missing_count += 3
                if trk.missing_count > self.max_missing_frames:
                    if trk.track_id in self.tracks:
                        del self.tracks[trk.track_id]
                continue

            if trk.is_hand_locked and palm_pos and hand_grasping:
                target_cx = palm_pos[0] + trk.hand_offset[0]
                target_cy = palm_pos[1] + trk.hand_offset[1]
                hw = trk.w / 2.0
                hh = trk.h / 2.0
                trk.box[0] = target_cx - hw
                trk.box[1] = target_cy - hh
                trk.box[2] = target_cx + hw
                trk.box[3] = target_cy + hh

            if trk.hit_count >= self.min_hits_to_show or trk.is_hand_locked:
                output.append(trk.to_dict())

        return output

