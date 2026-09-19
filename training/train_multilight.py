"""
train_multilight.py — YOLOv8 Transfer Learning for Cans, Bottles, Phones & Humans
==================================================================================
Trains YOLOv8n on the multi-lighting dataset (1,000 scenes, 4 classes).
Saves trained weights to runs/train/multilight/weights/best.pt.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = ROOT_DIR / "synthetic_data" / "dataset_multilight" / "data.yaml"
OUTPUT_DIR = ROOT_DIR / "runs" / "train"


def main():
    print("=" * 65)
    print("  Starting Multi-Lighting YOLOv8n Training")
    print(f"  Dataset: {DATASET_YAML}")
    print("  Classes: 0: bottle, 1: can, 2: phone, 3: person")
    print("=" * 65)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found at {DATASET_YAML}")

    base_model_path = ROOT_DIR / "yolov8n.pt"
    model = YOLO(str(base_model_path) if base_model_path.exists() else "yolov8n.pt")

    # Fast CPU training with transfer learning
    results = model.train(
        data=str(DATASET_YAML),
        epochs=15,
        imgsz=416,
        batch=16,
        device="cpu",
        project=str(OUTPUT_DIR),
        name="multilight",
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.6,
        fliplr=0.5,
        mosaic=0.5,
        exist_ok=True,
        verbose=True,
    )

    best_weights = OUTPUT_DIR / "multilight" / "weights" / "best.pt"
    if best_weights.exists():
        target_copy = ROOT_DIR / "multilight_best.pt"
        shutil.copy(best_weights, target_copy)
        print("=" * 65)
        print("TRAINING COMPLETE!")
        print(f"  Best Weights: {best_weights}")
        print(f"  Exported Copy: {target_copy}")
        print("=" * 65)
    else:
        print(f"[WARNING] Best weights not found at {best_weights}")


if __name__ == "__main__":
    main()
