"""
export_onnx.py — ASTRA-HAR Model Export Script
===============================================
Exports the trained YOLOv8 model to:
  1. ONNX format (opset 12, dynamic batch size)
  2. TorchScript format (for edge deployment)

Usage (from YOLO/ root):
    python training/export_onnx.py

Prerequisites:
    python training/train.py   # must be done first
    pip install onnx onnxruntime
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAINING_DIR  = Path(__file__).resolve().parent
YOLO_ROOT     = TRAINING_DIR.parent
WEIGHTS_PATH  = YOLO_ROOT / "runs" / "train" / "astra_har" / "weights" / "best.pt"
EXPORT_DIR    = YOLO_ROOT / "runs" / "train" / "astra_har" / "exports"


def export_onnx(model) -> Path:
    """
    Export the model to ONNX format with:
      - opset version 12
      - dynamic batch axis
      - simplification if onnxsim is available
    Returns path to the exported .onnx file.
    """
    print("[1/2] Exporting to ONNX (opset=12, dynamic batch)...")
    # ultralytics export returns the path to the exported file
    onnx_path = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        dynamic=True,        # dynamic batch dimension
        simplify=True,       # simplify graph via onnx-simplifier (if installed)
        half=False,          # FP32 for maximum compatibility
    )
    print(f"  → ONNX model saved: {onnx_path}")
    return Path(onnx_path)


def export_torchscript(model) -> Path:
    """
    Export the model to TorchScript format for edge / mobile deployment.
    Returns path to the exported .torchscript file.
    """
    print("[2/2] Exporting to TorchScript...")
    ts_path = model.export(
        format="torchscript",
        imgsz=640,
        optimize=False,      # set True for ARM / mobile optimisation
    )
    print(f"  → TorchScript model saved: {ts_path}")
    return Path(ts_path)


def verify_onnx(onnx_path: Path) -> None:
    """
    Run a quick ONNX runtime inference pass to verify the exported model.
    Requires onnxruntime.
    """
    try:
        import numpy as np
        import onnxruntime as ort

        print()
        print("  Verifying ONNX model with onnxruntime...")
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name  = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
        outputs = sess.run([output_name], {input_name: dummy_input})
        print(f"  ✓ ONNX inference OK — output shape: {outputs[0].shape}")
    except ImportError:
        print("  [SKIP] onnxruntime not installed; skipping verification.")
    except Exception as exc:
        print(f"  [WARN] ONNX verification failed: {exc}")


def main() -> None:
    """Export best.pt to ONNX and TorchScript."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Weights not found at {WEIGHTS_PATH}\n"
            "Please run:  python training/train.py"
        )

    from ultralytics import YOLO

    print("=" * 60)
    print("  ASTRA-HAR Model Export")
    print("=" * 60)
    print(f"  Source weights : {WEIGHTS_PATH}")
    print()

    model = YOLO(str(WEIGHTS_PATH))

    onnx_path = export_onnx(model)
    ts_path   = export_torchscript(model)

    verify_onnx(onnx_path)

    print()
    print("=" * 60)
    print("  Export Complete!")
    print(f"  ONNX         → {onnx_path}")
    print(f"  TorchScript  → {ts_path}")
    print()
    print("  To run ONNX inference:")
    print("    import onnxruntime as ort")
    print("    sess = ort.InferenceSession('<path>/best.onnx')")
    print("=" * 60)


if __name__ == "__main__":
    main()
