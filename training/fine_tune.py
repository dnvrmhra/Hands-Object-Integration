"""
fine_tune.py — Fine-tune YOLOv8 on Unified Multi-Object Dataset
==============================================================
Trains on unified_dataset/data.yaml containing:
  0: box
  1: container
  2: glovebox
  3: centrifuge
  4: smp-red
  5: smp-yel

Saves best weights to runs/train/unified_model/weights/best.pt and copies to root best.pt.
"""

from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path

# Force UTF-8 stdout
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent.parent
    data_yaml = root / "unified_dataset" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Cannot find dataset configuration at {data_yaml}")

    device = "0" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  ASTRA-HAR Fine-Tuning Multi-Object YOLOv8")
    print("=" * 60)
    print(f"  Dataset YAML : {data_yaml}")
    print(f"  Compute Device: {device} (CUDA: {torch.cuda.is_available()})")
    print("=" * 60)

    # Base model: start from pretrained yolov8n.pt for fresh 6-class detection head
    base_model = root / "yolov8n.pt"
    model = YOLO(str(base_model) if base_model.exists() else "yolov8n.pt")

    batch_size = 16 if torch.cuda.is_available() else 8
    epochs = 15

    print(f"Starting fine-tuning for {epochs} epochs (batch={batch_size})...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        project=str(root / "runs" / "train"),
        name="unified_model",
        exist_ok=True,
        workers=2,
        plots=True,
        save=True,
        val=True,
        mosaic=1.0,
        mixup=0.1,
    )

    print("\n" + "=" * 60)
    print("  Fine-Tuning Completed Successfully!")
    weights_path = root / "runs" / "train" / "unified_model" / "weights" / "best.pt"
    print(f"  Best model weights: {weights_path}")
    print("=" * 60)

    # Copy to root best.pt for immediate deployment
    if weights_path.exists():
        root_best = root / "best.pt"
        shutil.copy2(weights_path, root_best)
        print(f"[DEPLOY] Copied best weights to root: {root_best}")

        # Export to ONNX
        print("[EXPORT] Exporting to ONNX format...")
        try:
            m = YOLO(str(weights_path))
            m.export(format="onnx", imgsz=640, dynamic=True)
            print(f"[EXPORT] Saved ONNX to {weights_path.with_suffix('.onnx')}")
        except Exception as e:
            print(f"[EXPORT] ONNX export note: {e}")


if __name__ == "__main__":
    main()
