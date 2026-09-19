"""
build_unified_dataset.py — Merge Pilot and Microgravity Datasets
================================================================
Merges:
  1. ASTRA_YOLO_PILOT (246 train, 62 val):
     - Class 0: box
     - Class 1: container
  2. synthetic_data/dataset (800 train, 200 val):
     - Class 0 (GLOVEBOX-01)  -> Remapped to Class 2: glovebox
     - Class 1 (CENTRIFUGE-02) -> Remapped to Class 3: centrifuge
     - Class 2 (SMP-RED)       -> Remapped to Class 4: smp-red
     - Class 3 (SMP-YEL)       -> Remapped to Class 5: smp-yel

Generates `unified_dataset/` with:
  - images/train, images/val
  - labels/train, labels/val
  - data.yaml (6 classes)
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PILOT_DIR = ROOT_DIR / "ASTRA_YOLO_PILOT" / "ASTRA_YOLO"
SYNTH_DIR = ROOT_DIR / "synthetic_data" / "dataset"
OUT_DIR = ROOT_DIR / "unified_dataset"


def build_unified_dataset():
    print("[DATASET] Building unified multi-object dataset...")

    # Create destination directories
    for split in ["train", "val"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {i: 0 for i in range(6)}

    # -------------------------------------------------------------
    # 1. Ingest ASTRA_YOLO_PILOT (Classes 0: box, 1: container)
    # -------------------------------------------------------------
    pilot_splits = {
        "train": (PILOT_DIR / "images" / "train", PILOT_DIR / "labels" / "train"),
        "val": (PILOT_DIR / "images" / "val", PILOT_DIR / "labels" / "val"),
    }

    for split, (img_dir, lbl_dir) in pilot_splits.items():
        if not img_dir.exists():
            continue
        for img_file in img_dir.glob("*.jpg"):
            dest_img = OUT_DIR / "images" / split / f"pilot_{img_file.name}"
            shutil.copy2(img_file, dest_img)

            # Label file
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            dest_lbl = OUT_DIR / "labels" / split / f"pilot_{img_file.stem}.txt"
            if lbl_file.exists():
                with open(lbl_file, "r") as f_in, open(dest_lbl, "w") as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            counts[cls_id] += 1
                            f_out.write(line)

    print(f"[DATASET] Ingested ASTRA_YOLO_PILOT images.")

    # -------------------------------------------------------------
    # 2. Ingest synthetic_data/dataset (Remapped classes 2, 3, 4, 5)
    # -------------------------------------------------------------
    synth_splits = {
        "train": (SYNTH_DIR / "images" / "train", SYNTH_DIR / "labels" / "train"),
        "val": (SYNTH_DIR / "images" / "val", SYNTH_DIR / "labels" / "val"),
    }

    # Offset mapping: 0->2, 1->3, 2->4, 3->5
    SYNTH_OFFSET = 2

    for split, (img_dir, lbl_dir) in synth_splits.items():
        if not img_dir.exists():
            continue
        # Limit to 300 train and 80 val for class balance with pilot data
        limit = 300 if split == "train" else 80
        added = 0

        for img_file in sorted(img_dir.glob("*.png")):
            if added >= limit:
                break

            dest_img = OUT_DIR / "images" / split / f"synth_{img_file.name}"
            shutil.copy2(img_file, dest_img)

            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            dest_lbl = OUT_DIR / "labels" / split / f"synth_{img_file.stem}.txt"
            if lbl_file.exists():
                with open(lbl_file, "r") as f_in, open(dest_lbl, "w") as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if parts:
                            old_cls = int(parts[0])
                            new_cls = old_cls + SYNTH_OFFSET
                            counts[new_cls] += 1
                            new_line = f"{new_cls} " + " ".join(parts[1:]) + "\n"
                            f_out.write(new_line)
            added += 1

    print(f"[DATASET] Ingested synthetic mission objects with remapped indices.")

    # -------------------------------------------------------------
    # 3. Create unified data.yaml
    # -------------------------------------------------------------
    yaml_content = f"""path: {OUT_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: box
  1: container
  2: glovebox
  3: centrifuge
  4: smp-red
  5: smp-yel
"""
    yaml_path = OUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[DATASET] Created {yaml_path}")
    print(f"[DATASET] Label distribution: {counts}")
    return yaml_path


if __name__ == "__main__":
    build_unified_dataset()
