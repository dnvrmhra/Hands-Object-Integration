"""
video_processor.py — Multi-Object Video Ingest & 3D Telemetry Extractor
========================================================================
Supports:
  1. Custom Fine-Tuned Model (box, container, glovebox, centrifuge, smp-red, smp-yel)
  2. General Task Object Detection (bottle, cup, laptop, cell phone, book, scissors, etc.)
  3. MediaPipe 3D Orientation-Agnostic Hand Tracking (6-DOF attitude, intrinsic normalization)
  4. 3D Spatial-Temporal Human-Object Interaction (HOI) Recognition
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import imageio
import numpy as np

from inference.hoi_engine import HOIFusionEngine, draw_hoi_overlays

ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_CANDIDATES = [
    ROOT_DIR / "runs" / "train" / "unified_model" / "weights" / "best.pt",
    ROOT_DIR / "runs" / "train" / "astra_pilot" / "weights" / "best.pt",
    ROOT_DIR / "best.pt",
]

# Everyday task objects to recognize from COCO if present in normal task videos
COMMON_TASK_CLASSES = {
    39: "bottle",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    73: "book",
    76: "scissors",
    24: "backpack",
}


class VideoProcessor:
    def __init__(self):
        self.custom_model = None
        self.coco_model = None
        self.class_names = {0: "box", 1: "container", 2: "glovebox", 3: "centrifuge", 4: "smp-red", 5: "smp-yel"}
        self.hoi_engine = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        from ultralytics import YOLO

        # 1. Custom fine-tuned model
        for p in WEIGHTS_CANDIDATES:
            if p.exists():
                try:
                    self.custom_model = YOLO(str(p))
                    self.class_names = getattr(self.custom_model, "names", self.class_names)
                    print(f"[PROCESSOR] Custom model loaded from {p} (classes: {len(self.class_names)})")
                    break
                except Exception as e:
                    print(f"[PROCESSOR] Failed to load custom model from {p}: {e}")

        # 2. General task objects model (yolov8n.pt for everyday items like bottle, cup, laptop)
        try:
            base_p = ROOT_DIR / "yolov8n.pt"
            self.coco_model = YOLO(str(base_p) if base_p.exists() else "yolov8n.pt")
            print("[PROCESSOR] General task objects detector loaded.")
        except Exception as e:
            print(f"[PROCESSOR] COCO detector note: {e}")

        # 3. 3D HOI Engine
        try:
            self.hoi_engine = HOIFusionEngine()
            print("[PROCESSOR] 3D Orientation-Agnostic HOI Engine initialized.")
        except Exception as e:
            print(f"[PROCESSOR] Failed to initialize 3D HOI Engine: {e}")

    def _run_detection(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run dual detection: custom mission objects + everyday human task objects."""
        all_dets: list[dict[str, Any]] = []

        # 1. Run custom model
        if self.custom_model is not None:
            try:
                res = self.custom_model.predict(frame, conf=0.30, verbose=False)
                for r in res:
                    if r.boxes is not None:
                        for b in r.boxes:
                            cls_id = int(b.cls[0])
                            conf = float(b.conf[0])
                            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                            all_dets.append({
                                "class_id": cls_id,
                                "class_name": str(self.class_names.get(cls_id, f"item_{cls_id}")),
                                "confidence": round(conf, 4),
                                "bbox": {
                                    "x1": round(x1, 1), "y1": round(y1, 1),
                                    "x2": round(x2, 1), "y2": round(y2, 1),
                                    "cx": round((x1 + x2) / 2, 1),
                                    "cy": round((y1 + y2) / 2, 1),
                                    "w": round(x2 - x1, 1),
                                    "h": round(y2 - y1, 1),
                                },
                            })
            except Exception:
                pass

        # 2. Run everyday task objects detector (COCO)
        if self.coco_model is not None:
            try:
                res_coco = self.coco_model.predict(frame, conf=0.35, verbose=False)
                for r in res_coco:
                    if r.boxes is not None:
                        for b in r.boxes:
                            cls_id = int(b.cls[0])
                            if cls_id in COMMON_TASK_CLASSES:
                                conf = float(b.conf[0])
                                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                                # Check IoU overlap with existing custom detections to avoid duplicate boxes
                                is_dup = False
                                for existing in all_dets:
                                    eb = existing["bbox"]
                                    if abs(eb["cx"] - cx) < 30 and abs(eb["cy"] - cy) < 30:
                                        is_dup = True
                                        break

                                if not is_dup:
                                    all_dets.append({
                                        "class_id": 100 + cls_id,
                                        "class_name": COMMON_TASK_CLASSES[cls_id],
                                        "confidence": round(conf, 4),
                                        "bbox": {
                                            "x1": round(x1, 1), "y1": round(y1, 1),
                                            "x2": round(x2, 1), "y2": round(y2, 1),
                                            "cx": round(cx, 1), "cy": round(cy, 1),
                                            "w": round(x2 - x1, 1), "h": round(y2 - y1, 1),
                                        },
                                    })
            except Exception:
                pass

        return all_dets

    def process(
        self,
        input_video_path: str,
        output_video_path: str,
        target_fps: int = 15,
        max_duration_sec: float = 60.0,
    ) -> dict[str, Any]:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {input_video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(orig_fps / target_fps)))

        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

        frames_telemetry: list[dict[str, Any]] = []
        action_counts: dict[str, int] = {}

        writer = imageio.get_writer(
            output_video_path,
            fps=target_fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )

        in_frame_idx = 0
        out_frame_idx = 0
        t0 = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                current_sec = in_frame_idx / orig_fps
                if current_sec > max_duration_sec:
                    break

                if in_frame_idx % frame_interval == 0:
                    h, w, _ = frame.shape
                    if w != 640 or h != 480:
                        frame = cv2.resize(frame, (640, 480))

                    # Multi-object detection
                    detections = self._run_detection(frame)

                    # 3D MediaPipe & Orientation-Agnostic HOI
                    hands_data = []
                    hoi_summary = {
                        "state": "IDLE",
                        "target_object": None,
                        "confidence": 0.95,
                        "proximity_cm": 0.0,
                        "action_label": "IDLE",
                        "attitude_3d": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "is_invariant": True},
                    }

                    if self.hoi_engine is not None:
                        try:
                            hands_data, tracked_objs, hoi_summary = self.hoi_engine.process(frame, detections)
                        except Exception:
                            pass

                    # Annotate frame
                    annotated_bgr = draw_hoi_overlays(frame, detections, hands_data, hoi_summary)

                    # Append to video
                    rgb_annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    writer.append_data(rgb_annotated)

                    # Record telemetry
                    primary_att = hoi_summary.get("attitude_3d", {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "is_invariant": True})
                    action = hoi_summary.get("action_label", "IDLE")
                    action_counts[action] = action_counts.get(action, 0) + 1

                    frames_telemetry.append({
                        "frame_id": out_frame_idx,
                        "timestamp": round(out_frame_idx / target_fps, 2),
                        "fps": target_fps,
                        "detections": detections,
                        "hands_count": len(hands_data),
                        "hoi": hoi_summary,
                        "attitude_3d": primary_att,
                        "proximity_cm": hoi_summary.get("proximity_cm", 0.0),
                        "is_grasping": hoi_summary.get("is_grasping", False),
                    })

                    out_frame_idx += 1

                in_frame_idx += 1

        finally:
            cap.release()
            writer.close()

        elapsed = round(time.time() - t0, 2)
        print(f"[PROCESSOR] Video processing complete: {out_frame_idx} frames in {elapsed}s")

        return {
            "output_video": output_video_path,
            "total_frames": out_frame_idx,
            "duration_sec": round(out_frame_idx / target_fps, 2),
            "processing_time_sec": elapsed,
            "action_counts": action_counts,
            "frames": frames_telemetry,
        }
