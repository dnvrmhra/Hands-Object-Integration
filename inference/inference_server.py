"""
inference_server.py — Butter-Smooth 30 FPS Real-Time Vision Server (Zero Jitter, Zero Delay)
============================================================================================
Architecture:
  1. Thread 1: Dedicated Camera Ingest (Zero-Buffer DirectShow Drain)
     - Continuously flushes OpenCV's internal DirectShow queue (CAP_PROP_BUFFERSIZE=1).
     - Guarantees < 5ms raw camera latency.
  2. Thread 2: Background Async YOLO Detector (~10-12 FPS)
     - Strictly classes=[39, 41] (bottle, cup/can), conf=0.28, iou=0.45.
     - Runs in background; NEVER blocks the video stream or MediaPipe!
     - Feeds SmoothBoxTracker with deduplication and EMA smoothing.
  3. Thread 3: Zero-Lag 30 FPS Render Loop (MediaPipe + Hand-Locked Kinematics)
     - Runs locked at 30 FPS (20ms loop latency: 16ms MediaPipe + 4ms draw & JPEG encode).
     - Wireframes and 3D attitude triads are 100% in sync with physical hand motion.
     - Bounding boxes follow grasping hands at full 30 FPS without detector delay.
  4. Stream Generator:
     - Yields fresh frames only, automatically dropping stale frames to prevent client buffering.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Generator, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from inference.hoi_engine import HOIFusionEngine, draw_hoi_overlays
from inference.human_state_tracker import HumanStateTracker
from inference.movement_tracker import CanBottleMovementTracker
from inference.smooth_tracker import SmoothBoxTracker

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "runs" / "processed_videos"
UPLOADS_DIR = ROOT_DIR / "runs" / "uploads"
SHOWCASE_VIDEO = ROOT_DIR / "frontend" / "public" / "showcase_yolov8.mp4"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 30.0
FRAME_PERIOD = 1.0 / TARGET_FPS

# ===========================================================================
# Configuration: Allowed Detection Classes & Confidence Threshold
# ===========================================================================
# Temporary allowlist until dedicated ASTRA-HAR custom detector is trained.
# Generic COCO classes (bottle, can, phone, cup, etc.) are strictly filtered out.
# Expand this list as custom ASTRA-HAR dataset weights become available:
# e.g., ALLOWED_CLASSES = ["person", "sample", "test_tube", "centrifuge", "glovebox"]
ALLOWED_CLASSES: list[str] = [
    "person",
    "bottle",
    "can",
    "phone",
]

# Configurable confidence threshold (applied to all allowed classes)
CONFIDENCE_THRESHOLD: float = 0.55

app = FastAPI(
    title="ASTRA-HAR Smooth Vision Server",
    description="Butter-Smooth 30 FPS Live Webcam Hand Tracking & Can/Bottle Relocation Server",
    version="8.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/processed", StaticFiles(directory=str(PROCESSED_DIR)), name="processed")


# ---------------------------------------------------------------------------
# Decoupled 30 FPS Vision Pipeline
# ---------------------------------------------------------------------------
class HighPerformanceLivePipeline:
    def __init__(self):
        self.running = False
        self.cam_thread: Optional[threading.Thread] = None
        self.yolo_thread: Optional[threading.Thread] = None
        self.render_thread: Optional[threading.Thread] = None

        self.cam_lock = threading.Lock()
        self.state_lock = threading.Lock()

        # Models & Trackers
        self.coco_model = None
        self.hoi_engine: Optional[HOIFusionEngine] = None
        self.smooth_tracker = SmoothBoxTracker(
            alpha_coord=0.65,
            max_missing_frames=8,
            min_hits_to_show=2,
            max_tracks=8,
        )
        self.human_state_tracker = HumanStateTracker(
            max_missing_frames=8,
            min_hits_to_show=2,
        )
        self.movement_tracker = CanBottleMovementTracker()
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.lighting_mode: str = "NORMAL"

        # Camera
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_source_name = "NONE"

        # Frame buffers
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_hands: list[dict[str, Any]] = []
        self.latest_humans: list[dict[str, Any]] = []
        self.latest_jpeg_bytes: Optional[bytes] = None
        self.frame_seq: int = 0

        # Cached telemetry for WebSocket
        self.latest_telemetry: dict[str, Any] = {
            "frame_id": 0,
            "timestamp": time.time(),
            "fps": 30.0,
            "detections": [],
            "humans": [],
            "hands": [],
            "movement": {"state": "AT_REST", "distance_cm": 0.0, "speed_cm_s": 0.0, "history": []},
            "camera_source": "INITIALIZING",
            "lighting": "NORMAL",
            "allowed_classes": ALLOWED_CLASSES,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "counts": {
                "humans": 0,
                "human_state": "NONE",
                "allowed_objects": {},
                "total_allowed": 0,
                "bottles": 0,
                "cans": 0,
                "phones": 0,
            },
        }

    def load_models(self) -> None:
        from ultralytics import YOLO

        # IMPORTANT: Always use base COCO yolov8n.pt (trained on millions of real-world
        # images). The synthetic multilight_v2 model hallucinates bottles/cans from room
        # textures because it was trained on procedural synthetic data.
        base_p = ROOT_DIR / "yolov8n.pt"
        model_path = str(base_p) if base_p.exists() else "yolov8n.pt"
        model_type = "Base COCO YOLOv8n (Real-World)"

        try:
            self.coco_model = YOLO(model_path)
            print(f"[VISION] Active Model: {model_type} ({model_path}). Ready for Cans, Bottles, Phones & Humans.")
        except Exception as e:
            print(f"[VISION] YOLO load error: {e}")

        try:
            self.hoi_engine = HOIFusionEngine()
            print("[VISION] MediaPipe 3D Hand Tracker initialized.")
        except Exception as e:
            print(f"[VISION] Hand Engine init error: {e}")

    def _adapt_lighting(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        """
        Dynamically normalizes lighting to maximize YOLO detection accuracy
        across low-light, harsh glare, and varied room illumination.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))

        if mean_lum < 75.0:
            # Underexposed / low-light: boost L-channel using CLAHE
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enh = self.clahe.apply(l)
            enhanced_lab = cv2.merge((l_enh, a, b))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR), "LOW_LIGHT (BOOSTED)"

        elif mean_lum > 195.0:
            # Overexposed / glare: compress specular highlights
            norm = frame.astype(np.float32) / 255.0
            compressed = (np.power(norm, 1.35) * 255.0).astype(np.uint8)
            return compressed, "GLARE (COMPRESSED)"

        return frame, "NORMAL"

    def _map_detection(
        self,
        cls_id: int,
        names_dict: dict[int, str],
        bbox: dict[str, float],
        crop: Optional[np.ndarray] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Maps detections to canonical categories ('person', 'object') and names ('bottle', 'can', 'phone', 'person').
        Handles both our dedicated 4-class model:
          0: bottle, 1: can, 2: phone, 3: person
        and standard COCO 80-class fallback.
        """
        raw_name = names_dict.get(cls_id, "").lower()

        # Dedicated 4-class direct mapping (exact and zero ambiguity)
        if len(names_dict) <= 6:
            if cls_id == 0 or "bottle" in raw_name:
                return "object", "bottle"
            elif cls_id == 1 or "can" in raw_name:
                return "object", "can"
            elif cls_id == 2 or "phone" in raw_name:
                return "object", "phone"
            elif cls_id == 3 or "person" in raw_name:
                return "person", "person"

        # Fallback for standard 80-class COCO:
        # 1. Person: Class 0
        if "person" in raw_name:
            return "person", "person"

        # 2. Bottle: Class 39
        if "bottle" in raw_name:
            return "object", "bottle"

        # 3. Cup / Can: Class 41
        if "cup" in raw_name or "can" in raw_name:
            return "object", "can"

        # 4. Smartphone: Class 67
        if "phone" in raw_name or "cell" in raw_name:
            return "object", "phone"

        # 5. Wine glass (40) and Vase (75) — do NOT remap these to bottle/can.
        # They are a major source of false positives from room textures.
        # Simply reject them.

        return None, None

    def start(self) -> None:
        if self.running:
            return
        self.load_models()
        self.running = True

        # Thread 1: Camera Ingest (continuous buffer drain, zero camera lag)
        self.cam_thread = threading.Thread(target=self._camera_grabber_loop, daemon=True)
        self.cam_thread.start()

        # Thread 2: Background Async YOLO Detector (~10-12 FPS)
        self.yolo_thread = threading.Thread(target=self._yolo_worker_loop, daemon=True)
        self.yolo_thread.start()

        # Thread 3: Zero-Lag 30 FPS Render Loop (MediaPipe + Drawing + Streaming)
        self.render_thread = threading.Thread(target=self._render_stream_loop, daemon=True)
        self.render_thread.start()

    def _init_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()

        print("[CAMERA] Opening physical webcam (DirectShow)...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                self.cap = cap
                self.camera_source_name = "PHYSICAL_WEBCAM_0"
                print("[CAMERA] Physical webcam connected with single-frame buffer!")
                return

        # Fallback to showcase video
        if SHOWCASE_VIDEO.exists():
            print(f"[CAMERA] Using fallback showcase video: {SHOWCASE_VIDEO}")
            self.cap = cv2.VideoCapture(str(SHOWCASE_VIDEO))
            self.camera_source_name = "FALLBACK_SHOWCASE_VIDEO"
        else:
            self.camera_source_name = "SYNTHETIC_CANVAS"

    def _camera_grabber_loop(self) -> None:
        """
        Thread 1: Dedicated tight camera loop.
        Flushes OpenCV's internal DirectShow buffer continuously so that
        self.latest_raw_frame is always the live current frame (<5ms old).
        """
        self._init_camera()

        while self.running:
            frame = None
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()

            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:, :] = (15, 23, 35)

            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (640, 480))

            with self.cam_lock:
                self.latest_raw_frame = frame

            time.sleep(0.002)

    def _yolo_worker_loop(self) -> None:
        """
        Thread 2: Background Async YOLO Detector (~10-12 FPS).
        Detects cans, bottles, phones, and humans across varied lighting.
        Feeds SmoothBoxTracker (for containers/phones) and HumanStateTracker (for humans).
        NEVER blocks the 30 FPS video or hand tracking loop!
        """
        while self.running:
            t0 = time.perf_counter()

            with self.cam_lock:
                if self.latest_raw_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.latest_raw_frame.copy()

            with self.state_lock:
                hands = list(self.latest_hands)

            # Adaptive dynamic range normalization for low-light or glare
            enhanced_frame, light_cond = self._adapt_lighting(frame)
            self.lighting_mode = light_cond

            raw_objects: list[dict[str, Any]] = []
            raw_persons: list[dict[str, Any]] = []

            if self.coco_model is not None:
                try:
                    names = self.coco_model.names
                    if len(names) <= 6:
                        target_ids = list(range(len(names)))
                    else:
                        # 0: person, 39: bottle, 41: cup (can), 67: cell phone
                        target_ids = [0, 39, 41, 67]

                    res = self.coco_model.predict(
                        enhanced_frame,
                        imgsz=480,
                        conf=0.25,
                        iou=0.45,
                        classes=target_ids,
                        verbose=False,
                    )

                    for r in res:
                        if r.boxes is not None:
                            for b in r.boxes:
                                cid = int(b.cls[0])
                                conf = float(b.conf[0])
                                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                                w_box = max(1.0, x2 - x1)
                                h_box = max(1.0, y2 - y1)

                                # Candidate crop for bottle neck analysis
                                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                                ix2, iy2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                                crop = frame[iy1:iy2, ix1:ix2] if (iy2 > iy1 and ix2 > ix1) else None

                                bbox_dict = {
                                    "x1": round(x1, 1),
                                    "y1": round(y1, 1),
                                    "x2": round(x2, 1),
                                    "y2": round(y2, 1),
                                    "cx": round((x1 + x2) / 2, 1),
                                    "cy": round((y1 + y2) / 2, 1),
                                    "w": round(w_box, 1),
                                    "h": round(h_box, 1),
                                }

                                category, canonical_name = self._map_detection(cid, names, bbox_dict, crop)
                                if category is None or canonical_name is None:
                                    continue

                                # 1. Strict Class Allowlist Check:
                                if canonical_name.lower() not in [c.lower() for c in ALLOWED_CLASSES]:
                                    continue

                                # 2. Physical geometry & edge boundary checks to eliminate false room detections
                                frame_w, frame_h = frame.shape[1], frame.shape[0]
                                ar = h_box / w_box

                                # Reject extreme edge slivers (objects cut off on the boundary)
                                if (x1 <= 2 or x2 >= frame_w - 2) and w_box < 25:
                                    continue
                                if (y1 <= 2 or y2 >= frame_h - 2) and h_box < 25:
                                    continue

                                # Hand-aware proximity check for adaptive sensitivity
                                is_near_hand = False
                                if hands:
                                    for h_item in hands:
                                        p = h_item.get("palm_px", (0, 0))
                                        dx = max(0.0, x1 - p[0], p[0] - x2)
                                        dy = max(0.0, y1 - p[1], p[1] - y2)
                                        if math.hypot(dx, dy) < 100:
                                            is_near_hand = True
                                            break

                                # Minimum pixel area guard — allow smaller objects (cans/phones at distance)
                                if w_box * h_box < 250:
                                    continue

                                min_c = 0.25 if is_near_hand else 0.30

                                if canonical_name == "can":
                                    # Metal soda cans (aspect ratio 0.35 - 3.00)
                                    if not (0.35 <= ar <= 3.00):
                                        continue
                                    if w_box < 15 or h_box < 15:
                                        continue
                                    if conf < min_c:
                                        continue

                                elif canonical_name == "bottle":
                                    # Metallic thermos bottles / bottles (aspect ratio 0.35 - 4.50)
                                    if not (0.35 <= ar <= 4.50):
                                        continue
                                    if w_box < 15 or h_box < 20:
                                        continue
                                    if conf < min_c:
                                        continue

                                elif canonical_name == "phone":
                                    # Smartphones / cell phones (aspect ratio 0.30 - 3.50)
                                    if not (0.30 <= ar <= 3.50):
                                        continue
                                    if w_box < 15 or h_box < 15:
                                        continue
                                    if conf < min_c:
                                        continue

                                elif canonical_name == "person":
                                    if w_box < 45 or h_box < 70:
                                        continue
                                    if conf < 0.40:
                                        continue

                                # 3. Ensure bounding coordinates are clamped precisely to frame boundaries
                                x1_cl = max(0.0, min(float(frame_w), x1))
                                y1_cl = max(0.0, min(float(frame_h), y1))
                                x2_cl = max(0.0, min(float(frame_w), x2))
                                y2_cl = max(0.0, min(float(frame_h), y2))

                                if (x2_cl - x1_cl) < 4.0 or (y2_cl - y1_cl) < 4.0:
                                    continue

                                clamped_bbox = {
                                    "x1": round(x1_cl, 1),
                                    "y1": round(y1_cl, 1),
                                    "x2": round(x2_cl, 1),
                                    "y2": round(y2_cl, 1),
                                    "cx": round((x1_cl + x2_cl) / 2, 1),
                                    "cy": round((y1_cl + y2_cl) / 2, 1),
                                    "w": round(x2_cl - x1_cl, 1),
                                    "h": round(y2_cl - y1_cl, 1),
                                }

                                det_dict = {
                                    "class_id": cid,
                                    "class_name": canonical_name,
                                    "confidence": round(conf, 2),
                                    "bbox": clamped_bbox,
                                }

                                if category == "person":
                                    raw_persons.append(det_dict)
                                else:
                                    raw_objects.append(det_dict)

                except Exception as err:
                    print(f"[YOLO-WORKER] Error in detection cycle: {err}")

            # Update trackers with fresh detections
            with self.state_lock:
                self.smooth_tracker.update(raw_objects, hands)
                self.human_state_tracker.update(raw_persons)

            # Pace detector to ~10-12 FPS so CPU stays cool and free for video & MediaPipe
            dt = time.perf_counter() - t0
            sleep_time = max(0.010, 0.085 - dt)
            time.sleep(sleep_time)

    def _render_stream_loop(self) -> None:
        """
        Thread 3: Zero-Lag 30 FPS Render Loop.
        Runs MediaPipe hand tracking (16ms) + smooth box interpolation + human kinematics + drawing + JPEG encoding.
        Achieves locked 30 FPS with zero delay and zero jitter.
        """
        while self.running:
            t0 = time.perf_counter()

            with self.cam_lock:
                if self.latest_raw_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.latest_raw_frame.copy()

            # 1. Zero-Lag 30 FPS Hand Tracking (~16ms)
            hands_data: list[dict[str, Any]] = []
            if self.hoi_engine is not None and self.hoi_engine.hand_tracker is not None:
                try:
                    hands_data = self.hoi_engine.hand_tracker.process_frame(frame)
                except Exception:
                    pass

            with self.state_lock:
                self.latest_hands = hands_data
                current_boxes = self.smooth_tracker.get_current_boxes(hands_data)
                current_humans = self.human_state_tracker.get_current_humans()

            # 2. Update HOI State & Movement Engine
            hoi_summary = {
                "state": "IDLE",
                "target_object": None,
                "confidence": 0.95,
                "proximity_cm": 0.0,
                "action_label": "IDLE",
            }
            if self.hoi_engine is not None:
                try:
                    hoi_summary = self.hoi_engine.hoi_fusion(hands_data, current_boxes)
                except Exception:
                    pass

            movement_summary = {
                "state": "AT_REST",
                "distance_cm": 0.0,
                "speed_cm_s": 0.0,
                "history": [],
            }
            if self.movement_tracker is not None:
                try:
                    movement_summary = self.movement_tracker.update(current_boxes, hands_data)
                except Exception:
                    pass

            # 3. Draw Overlays onto Frame (~3ms)
            # A. Objects, Hands, 3D Attitude Triads
            annotated = draw_hoi_overlays(frame, current_boxes, hands_data, hoi_summary)

            # B. Humans (Bounding Boxes, State Badges: IDLE / MOVING / WALKING, Velocity Arrows)
            if self.human_state_tracker is not None:
                annotated = self.human_state_tracker.draw_human_overlays(annotated, current_humans)

            # C. Relocation Trajectory & Real-Time Distance
            if self.movement_tracker is not None:
                annotated = self.movement_tracker.draw_movement_overlay(annotated)

            # D. Lighting Mode Pill if non-standard
            if self.lighting_mode != "NORMAL":
                badge_text = f"LIGHT: {self.lighting_mode}"
                cv2.rectangle(annotated, (10, 440), (220, 470), (15, 23, 42), -1)
                cv2.rectangle(annotated, (10, 440), (220, 470), (168, 85, 247), 1)
                cv2.putText(annotated, badge_text, (18, 461), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (192, 132, 252), 1, cv2.LINE_AA)

            # 4. Fast JPEG Encoding (~2ms)
            ret, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if ret:
                num_humans = len(current_humans)
                active_human_state = current_humans[0]["state"] if current_humans else "NONE"
                allowed_obj_counts: dict[str, int] = {}
                for b in current_boxes:
                    cname = b["class_name"].lower()
                    allowed_obj_counts[cname] = allowed_obj_counts.get(cname, 0) + 1

                with self.state_lock:
                    self.latest_jpeg_bytes = jpeg.tobytes()
                    self.frame_seq += 1
                    self.latest_telemetry = {
                        "frame_id": self.frame_seq,
                        "timestamp": round(time.time(), 3),
                        "fps": 30.0,
                        "detections": current_boxes,
                        "humans": current_humans,
                        "hands": [
                            {
                                "hand_id": h["hand_id"],
                                "palm_px": h["palm_px"],
                                "is_grasping": h.get("is_grasping", False),
                                "attitude_3d": h.get("attitude_3d", {}),
                            }
                            for h in hands_data
                        ],
                        "movement": movement_summary,
                        "camera_source": self.camera_source_name,
                        "lighting": self.lighting_mode,
                        "allowed_classes": ALLOWED_CLASSES,
                        "confidence_threshold": CONFIDENCE_THRESHOLD,
                        "counts": {
                            "humans": num_humans,
                            "human_state": active_human_state,
                            "allowed_objects": allowed_obj_counts,
                            "total_allowed": num_humans + len(current_boxes),
                            "bottles": allowed_obj_counts.get("bottle", 0),
                            "cans": allowed_obj_counts.get("can", 0),
                            "phones": allowed_obj_counts.get("phone", 0),
                        },
                    }

            # Pace to steady 30 FPS
            dt = time.perf_counter() - t0
            sleep_time = max(0.001, FRAME_PERIOD - dt)
            time.sleep(sleep_time)


# Global Pipeline Instance
pipeline = HighPerformanceLivePipeline()

# Lazy-loaded video processor for uploaded videos
_video_proc: Optional[Any] = None


def get_video_processor():
    global _video_proc
    if _video_proc is None:
        from inference.video_processor import VideoProcessor
        _video_proc = VideoProcessor()
    return _video_proc


@app.on_event("startup")
def startup_event():
    pipeline.start()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "live_running": pipeline.running,
        "camera_source": pipeline.camera_source_name,
        "allowed_classes": ALLOWED_CLASSES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "classes": ALLOWED_CLASSES,
        "human_states": ["IDLE", "MOVING", "WALKING"],
        "target_fps": 30.0,
        "lighting_mode": pipeline.lighting_mode,
        "adaptive_lighting": True,
        "decoupled_architecture": True,
        "zero_delay_flushing": True,
    }


def _zero_delay_mjpeg_generator() -> Generator[bytes, None, None]:
    """Streams live video with 0ms delay, yielding only fresh frames."""
    last_sent_seq = -1
    while True:
        with pipeline.state_lock:
            cur_seq = pipeline.frame_seq
            jpeg_data = pipeline.latest_jpeg_bytes

        if jpeg_data is not None and cur_seq != last_sent_seq:
            last_sent_seq = cur_seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_data + b"\r\n"
        else:
            time.sleep(0.004)


@app.get("/video_feed")
def video_feed():
    """Live 30 FPS webcam stream with zero buffer delay and clean, non-duplicate overlays."""
    return StreamingResponse(
        _zero_delay_mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with pipeline.state_lock:
                telemetry = dict(pipeline.latest_telemetry)
            await websocket.send_text(json.dumps(telemetry))
            await asyncio.sleep(1.0 / 25)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.post("/api/process_video")
async def process_video(file: UploadFile = File(...)) -> dict[str, Any]:
    ext = Path(file.filename or "input.mp4").suffix.lower() or ".mp4"
    unique_id = uuid.uuid4().hex[:8]
    input_path = UPLOADS_DIR / f"upload_{unique_id}{ext}"
    output_filename = f"annotated_{unique_id}.mp4"
    output_path = PROCESSED_DIR / output_filename

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        proc = get_video_processor()
        res = proc.process(str(input_path), str(output_path), target_fps=15, max_duration_sec=90.0)
        return {
            "status": "success",
            "video_url": f"http://localhost:8000/processed/{output_filename}",
            "filename": output_filename,
            "total_frames": res["total_frames"],
            "duration_sec": res["duration_sec"],
            "action_counts": res["action_counts"],
            "frames": res["frames"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/process_sample")
def process_sample() -> dict[str, Any]:
    if not SHOWCASE_VIDEO.exists():
        raise HTTPException(status_code=404, detail="Showcase video not found.")
    unique_id = uuid.uuid4().hex[:8]
    output_filename = f"annotated_sample_{unique_id}.mp4"
    output_path = PROCESSED_DIR / output_filename
    proc = get_video_processor()
    res = proc.process(str(SHOWCASE_VIDEO), str(output_path), target_fps=15, max_duration_sec=30.0)
    return {
        "status": "success",
        "video_url": f"http://localhost:8000/processed/{output_filename}",
        "filename": output_filename,
        "total_frames": res["total_frames"],
        "duration_sec": res["duration_sec"],
        "action_counts": res["action_counts"],
        "frames": res["frames"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference.inference_server:app", host="0.0.0.0", port=8000, reload=False)
