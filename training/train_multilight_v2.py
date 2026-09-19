"""
train_multilight_v2.py — Fine-tune YOLOv8n on Photorealistic Multi-Lighting Dataset
===================================================================================
Trains dedicated 4-class detector:
  0: bottle
  1: can
  2: phone
  3: person

Under daylight, low-light, harsh glare, warm tungsten, and cool fluorescent conditions.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT_DIR / "synthetic_data" / "dataset_multilight_v2" / "data.yaml"
OUTPUT_DIR = ROOT_DIR / "runs" / "train" / "multilight_v2"


def main():
    print("=" * 70)
    print("ASTRA-HAR: Training Photorealistic Multi-Lighting YOLOv8n")
    print(f"Dataset config: {DATA_YAML}")
    print("Target classes: 0=bottle, 1=can, 2=phone, 3=person")
    print("=" * 70)

    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml not found at {DATA_YAML}. Run generate_real_multilight_dataset.py first!")
        sys.exit(1)

    # Load base pretrained weights for rapid transfer learning
    base_weights = ROOT_DIR / "yolov8n.pt"
    model = YOLO(str(base_weights) if base_weights.exists() else "yolov8n.pt")

    t0 = time.time()

    # Fine-tune model
    results = model.train(
        data=str(DATA_YAML),
        epochs=25,
        imgsz=480,
        batch=16,
        device="cpu",
        workers=4,
        project="runs/train",
        name="multilight_v2",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.003,
        lrf=0.01,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.4,
        shear=2.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        verbose=True,
    )

    elapsed_min = (time.time() - t0) / 60.0
    print("=" * 70)
    print(f"Training completed in {elapsed_min:.2f} minutes!")

    best_weights = OUTPUT_DIR / "weights" / "best.pt"
    if best_weights.exists():
        print(f"[SUCCESS] Trained weights saved to: {best_weights}")
    else:
        print(f"[WARNING] Best weights not found at expected path: {best_weights}")
    print("=" * 70)


if __name__ == "__main__":
    main()
