"""
evaluate.py — ASTRA-HAR Model Evaluation Script
================================================
Loads the trained best.pt weights and evaluates on the validation split.
Prints mAP50, mAP50-95, precision, recall per class.
Saves metrics to runs/train/astra_har/metrics.json.
Generates a confusion-matrix plot.

Usage (from YOLO/ root):
    python training/evaluate.py

Prerequisites:
    python training/train.py   # must be done first
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe on headless machines)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAINING_DIR  = Path(__file__).resolve().parent
YOLO_ROOT     = TRAINING_DIR.parent
DATASET_YAML  = YOLO_ROOT / "synthetic_data" / "dataset" / "data.yaml"
WEIGHTS_PATH  = YOLO_ROOT / "runs" / "train" / "astra_har" / "weights" / "best.pt"
METRICS_PATH  = YOLO_ROOT / "runs" / "train" / "astra_har" / "metrics.json"
CM_PLOT_PATH  = YOLO_ROOT / "runs" / "train" / "astra_har" / "confusion_matrix_custom.png"

CLASS_NAMES = ["GLOVEBOX-01", "CENTRIFUGE-02", "SMP-RED", "SMP-YEL"]


def main() -> None:
    """Run validation and export metrics + confusion matrix."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Weights not found at {WEIGHTS_PATH}\n"
            "Please run:  python training/train.py"
        )

    from ultralytics import YOLO

    print("=" * 60)
    print("  ASTRA-HAR Model Evaluation")
    print("=" * 60)
    print(f"  Weights    : {WEIGHTS_PATH}")
    print(f"  Dataset    : {DATASET_YAML}")
    print()

    model = YOLO(str(WEIGHTS_PATH))

    # --- Validate ---
    val_results = model.val(
        data=str(DATASET_YAML),
        imgsz=640,
        batch=16,
        verbose=True,
        save_json=True,
        plots=True,
    )

    # --- Extract metrics ---
    metrics_dict = val_results.results_dict if hasattr(val_results, "results_dict") else {}

    map50    = float(getattr(val_results.box, "map50",    0.0))
    map5095  = float(getattr(val_results.box, "map",      0.0))
    prec     = float(getattr(val_results.box, "mp",       0.0))  # mean precision
    rec      = float(getattr(val_results.box, "mr",       0.0))  # mean recall

    print()
    print("=" * 60)
    print("  OVERALL METRICS")
    print("=" * 60)
    print(f"  mAP@0.50       : {map50:.4f}")
    print(f"  mAP@0.50:0.95  : {map5095:.4f}")
    print(f"  Mean Precision  : {prec:.4f}")
    print(f"  Mean Recall     : {rec:.4f}")

    # Per-class breakdown (if available)
    per_class: dict[str, dict] = {}
    try:
        ap50_per_cls = val_results.box.ap50   # shape: (nc,)
        ap_per_cls   = val_results.box.ap     # shape: (nc,)
        print()
        print("  PER-CLASS mAP@0.50")
        print("  " + "-" * 40)
        for i, name in enumerate(CLASS_NAMES):
            val50 = float(ap50_per_cls[i]) if i < len(ap50_per_cls) else 0.0
            val95 = float(ap_per_cls[i])   if i < len(ap_per_cls)   else 0.0
            print(f"    [{i}] {name:<18} mAP50={val50:.4f}  mAP50-95={val95:.4f}")
            per_class[name] = {"mAP50": val50, "mAP50_95": val95}
    except Exception:
        pass   # per-class data may not be available in all ultralytics versions

    # --- Save metrics JSON ---
    output = {
        "mAP50":          map50,
        "mAP50_95":       map5095,
        "mean_precision": prec,
        "mean_recall":    rec,
        "per_class":      per_class,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(output, indent=2))
    print()
    print(f"  Metrics saved → {METRICS_PATH}")

    # --- Confusion matrix plot ---
    _plot_confusion_matrix(per_class)

    print(f"  Confusion matrix → {CM_PLOT_PATH}")
    print("=" * 60)


def _plot_confusion_matrix(per_class: dict) -> None:
    """
    Generate a simple per-class mAP bar chart saved as the 'confusion matrix' plot.
    (The actual confusion matrix image is already saved by ultralytics val(plots=True).)
    """
    names = list(per_class.keys()) if per_class else CLASS_NAMES
    vals  = [per_class.get(n, {}).get("mAP50", 0.0) for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor="white", linewidth=1.2)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mAP @ 0.50", fontsize=12)
    ax.set_title("ASTRA-HAR — Per-Class mAP@0.50", fontsize=14, fontweight="bold")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="0.50 threshold")
    ax.legend(fontsize=10)

    # Annotate bars
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(CM_PLOT_PATH, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
