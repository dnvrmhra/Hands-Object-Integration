"""
train.py — ASTRA-HAR YOLOv8 Training Script
=============================================
Trains a YOLOv8n model on the synthetic ASTRA-HAR dataset.

Usage (from YOLO/ root):
    python training/train.py

Prerequisites:
    pip install ultralytics torch torchvision
    python synthetic_data/generate_dataset.py   # must be done first
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAINING_DIR  = Path(__file__).resolve().parent
YOLO_ROOT     = TRAINING_DIR.parent
DATASET_YAML  = YOLO_ROOT / "synthetic_data" / "dataset" / "data.yaml"
RUNS_DIR      = YOLO_ROOT / "runs" / "train"


def _detect_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    """Train YOLOv8n on the ASTRA-HAR synthetic dataset."""
    # --- Validate dataset YAML exists ---
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {DATASET_YAML}\n"
            "Please run:  python synthetic_data/generate_dataset.py"
        )

    from ultralytics import YOLO

    device = _detect_device()
    print("=" * 60)
    print("  ASTRA-HAR YOLOv8 Training")
    print("=" * 60)
    print(f"  Dataset YAML : {DATASET_YAML}")
    print(f"  Device       : {device}")
    print(f"  Output dir   : {RUNS_DIR / 'astra_har'}")
    print("=" * 60)
    print()

    # --- Load pretrained YOLOv8n model ---
    model = YOLO("yolov8n.pt")   # downloads automatically on first run

    # --- Train ---
    results = model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(RUNS_DIR),
        name="astra_har",
        optimizer="Adam",
        lr0=0.001,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        device=device,
        exist_ok=True,          # overwrite previous run if re-training
        verbose=True,
    )

    # --- Report ---
    best_weights = RUNS_DIR / "astra_har" / "weights" / "best.pt"
    print()
    print("=" * 60)
    print("  Training Complete!")
    print(f"  Best weights → {best_weights}")
    print("=" * 60)

    # Save a brief training summary
    summary = {
        "model": "yolov8n",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "device": device,
        "best_weights": str(best_weights),
    }
    summary_path = RUNS_DIR / "astra_har" / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Training summary saved → {summary_path}")


if __name__ == "__main__":
    main()
