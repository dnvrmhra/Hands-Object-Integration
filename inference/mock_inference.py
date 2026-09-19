"""
mock_inference.py — ASTRA-HAR Standalone Mock Inference
========================================================
Simulates YOLO detection output for all four ASTRA-HAR classes with
realistic confidence scores and bounding boxes. Prints JSON frames to
stdout at 10 fps so the WebSocket pipeline can be tested without a
trained model.

No ML libraries required — only Python standard library + (optional) rich
for prettier output.

Usage:
    python inference/mock_inference.py
    python inference/mock_inference.py --fps 5 --frames 100
    python inference/mock_inference.py --json-only   # pure JSON, no colour

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "GLOVEBOX-01",
    1: "CENTRIFUGE-02",
    2: "SMP-RED",
    3: "SMP-YEL",
}

IMG_SIZE = 640

# Canonical positions for each class (pixel space, 640×640)
_CANONICAL: dict[int, dict[str, float]] = {
    0: {"cx": 207, "cy": 307, "w": 295, "h": 245},  # GLOVEBOX-01  (large, left-centre)
    1: {"cx": 205, "cy": 315, "w": 130, "h": 130},  # CENTRIFUGE-02 (inside glovebox)
    2: {"cx": 419, "cy": 236, "w":  18, "h":  52},  # SMP-RED (small tube, right)
    3: {"cx": 498, "cy": 246, "w":  17, "h":  52},  # SMP-YEL (small tube, right)
}

# Base confidences — will oscillate slightly each frame
_BASE_CONF: dict[int, float] = {
    0: 0.92,   # GLOVEBOX usually easiest to detect (large)
    1: 0.86,   # CENTRIFUGE medium confidence
    2: 0.80,   # SMP-RED small, slightly lower
    3: 0.77,   # SMP-YEL similar size, slightly lower
}

# Protocol step definition: which classes are present at each step
# Steps cycle every ~5 s (50 frames @ 10 fps)
_STEPS: list[list[int]] = [
    [0],             # Step 0: only glovebox
    [0, 1],          # Step 1: glovebox + centrifuge
    [0, 1, 2, 2],    # Step 2: + 2× SMP-RED
    [0, 1, 2, 2, 3, 3],  # Step 3: full scene
]

# Frames per protocol step
FRAMES_PER_STEP = 50


# ---------------------------------------------------------------------------
# Detection generation
# ---------------------------------------------------------------------------

def _make_bbox(cls_id: int, instance_idx: int = 0) -> dict[str, float]:
    """
    Return a bounding box dict for the given class, with small random jitter.
    Multiple instances of the same class are offset slightly.
    """
    canon = _CANONICAL[cls_id]
    offset_x = instance_idx * 38   # horizontal offset for multiple instances

    cx = canon["cx"] + offset_x + random.uniform(-3, 3)
    cy = canon["cy"] + random.uniform(-3, 3)
    w  = canon["w"]  + random.uniform(-2, 2)
    h  = canon["h"]  + random.uniform(-2, 2)

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return {
        "x1": round(x1, 1), "y1": round(y1, 1),
        "x2": round(x2, 1), "y2": round(y2, 1),
        "cx": round(cx,  1), "cy": round(cy, 1),
        "w":  round(w,   1), "h":  round(h,  1),
    }


def _make_confidence(cls_id: int, frame_id: int) -> float:
    """Return a realistically oscillating confidence value."""
    base  = _BASE_CONF[cls_id]
    osc   = 0.04 * math.sin(frame_id * 0.3 + cls_id)    # slow oscillation
    noise = random.uniform(-0.02, 0.02)
    return round(max(0.50, min(0.99, base + osc + noise)), 4)


def generate_frame(frame_id: int) -> dict[str, Any]:
    """
    Generate one mock detection frame payload.
    Returns a dict matching the WebSocket message format.
    """
    step_idx   = (frame_id // FRAMES_PER_STEP) % len(_STEPS)
    step_classes = _STEPS[step_idx]

    detections: list[dict[str, Any]] = []
    class_instance_count: dict[int, int] = {}

    for cls_id in step_classes:
        inst_idx = class_instance_count.get(cls_id, 0)
        class_instance_count[cls_id] = inst_idx + 1

        det = {
            "class_id":   cls_id,
            "class_name": CLASS_NAMES[cls_id],
            "confidence": _make_confidence(cls_id, frame_id),
            "bbox":       _make_bbox(cls_id, inst_idx),
        }
        detections.append(det)

    # Estimate bandwidth saved vs raw frame
    det_bytes    = len(json.dumps(detections).encode())
    raw_bytes    = IMG_SIZE * IMG_SIZE * 3
    bw_saved_pct = round(max(0.0, (1 - det_bytes / raw_bytes) * 100), 2)

    return {
        "frame_id":            frame_id,
        "timestamp":           round(time.time(), 4),
        "fps":                 10.0,
        "detections":          detections,
        "bandwidth_saved_pct": bw_saved_pct,
        "active_step":         step_idx,
    }


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty()

_ANSI = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "cyan":   "\033[36m",
    "yellow": "\033[33m",
    "green":  "\033[32m",
    "grey":   "\033[90m",
    "red":    "\033[31m",
}

_CLASS_COLOUR = {
    0: "\033[34m",   # blue
    1: "\033[35m",   # magenta
    2: "\033[31m",   # red
    3: "\033[33m",   # yellow
}


def _col(s: str, code: str) -> str:
    return f"{code}{s}{_ANSI['reset']}" if _USE_COLOUR else s


def pretty_print_frame(payload: dict[str, Any]) -> None:
    """Print a human-readable summary of a detection frame."""
    n_det  = len(payload["detections"])
    step   = payload["active_step"]
    fps    = payload["fps"]
    bw     = payload["bandwidth_saved_pct"]
    fid    = payload["frame_id"]

    header = (
        f"{_col('Frame', _ANSI['bold'])} {_col(str(fid), _ANSI['cyan']):<8}"
        f"  step={_col(str(step), _ANSI['yellow'])}  "
        f"det={_col(str(n_det), _ANSI['green'])}  "
        f"bw_saved={_col(f'{bw:.1f}%', _ANSI['green'])}  "
        f"fps={fps}"
    )
    print(header)

    for det in payload["detections"]:
        cls_id = det["class_id"]
        name   = det["class_name"]
        conf   = det["confidence"]
        b      = det["bbox"]
        col    = _CLASS_COLOUR.get(cls_id, "")
        print(
            f"  {_col(f'[{cls_id}]', col)} {_col(name, col):<18} "
            f"conf={conf:.4f}  "
            f"bbox=({b['x1']:.0f},{b['y1']:.0f})-({b['x2']:.0f},{b['y2']:.0f})  "
            f"wh=({b['w']:.0f}×{b['h']:.0f})"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASTRA-HAR Mock Inference — streams detection JSON")
    p.add_argument("--fps",       type=float, default=10.0,
                   help="Target frames per second (default: 10)")
    p.add_argument("--frames",    type=int,   default=0,
                   help="Number of frames to generate (0 = infinite)")
    p.add_argument("--json-only", action="store_true",
                   help="Print raw JSON only (no colour formatting)")
    p.add_argument("--seed",      type=int,   default=None,
                   help="Random seed for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    period    = 1.0 / max(0.1, args.fps)
    json_only = args.json_only or not sys.stdout.isatty()

    if not json_only:
        print(_col("ASTRA-HAR Mock Inference", _ANSI["bold"]))
        print(_col(f"  fps={args.fps}  frames={'∞' if args.frames == 0 else args.frames}", _ANSI["grey"]))
        print(_col("  Press Ctrl+C to stop.", _ANSI["grey"]))
        print()

    frame_id = 0
    try:
        while True:
            if args.frames > 0 and frame_id >= args.frames:
                break

            t0      = time.perf_counter()
            payload = generate_frame(frame_id)

            if json_only:
                print(json.dumps(payload))
            else:
                pretty_print_frame(payload)

            frame_id += 1
            elapsed  = time.perf_counter() - t0
            sleep_s  = max(0.0, period - elapsed)
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        if not json_only:
            print(_col(f"\nStopped after {frame_id} frames.", _ANSI["grey"]))


if __name__ == "__main__":
    main()
