"""
train_astra_pilot.py
====================
Trains Ultralytics YOLOv8 on the provided ASTRA_YOLO_PILOT dataset.
Classes:
  0: box
  1: container

Outputs saved to:
  runs/train/astra_pilot/
"""

from __future__ import annotations
import sys
import io
from pathlib import Path

# Force UTF-8 stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
from ultralytics import YOLO

def main():
    # Root is YOLO directory (parent of training/)
    root = Path(__file__).resolve().parent.parent
    data_yaml = root / "ASTRA_YOLO_PILOT" / "ASTRA_YOLO" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Cannot find dataset configuration at {data_yaml}")

    device = "0" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  ASTRA-HAR YOLOv8 Pilot Dataset Training")
    print("=" * 60)
    print(f"  Dataset YAML : {data_yaml}")
    print(f"  Compute Device: {device} (CUDA: {torch.cuda.is_available()})")
    print("=" * 60)

    # Load pretrained YOLOv8n (nano architecture, optimal for edge microgravity inference)
    model = YOLO("yolov8n.pt")

    # Train model
    # Using 25 epochs with batch=16 (or 8 on CPU) for rapid convergence on 246 training images
    batch_size = 16 if torch.cuda.is_available() else 8
    epochs = 25

    print(f"Starting training for {epochs} epochs (batch={batch_size})...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        project=str(root / "runs" / "train"),
        name="astra_pilot",
        exist_ok=True,
        workers=2,
        plots=True,
        save=True,
        val=True,
    )

    print("\n" + "=" * 60)
    print("  Training Completed Successfully!")
    weights_path = root / "runs" / "train" / "astra_pilot" / "weights" / "best.pt"
    print(f"  Best model weights: {weights_path}")
    print("=" * 60)

    # Export best model to ONNX for edge runtime
    if weights_path.exists():
        print("Exporting trained model to ONNX...")
        best_model = YOLO(str(weights_path))
        best_model.export(format="onnx", imgsz=640, dynamic=True)
        print(f"Exported to ONNX: {weights_path.with_suffix('.onnx')}")

if __name__ == "__main__":
    main()
