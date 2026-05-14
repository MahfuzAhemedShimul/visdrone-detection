"""
Task 1: VisDrone Dataset Preprocessing
Converts VisDrone annotations to YOLO format and applies augmentations.

VisDrone annotation format per line:
  x, y, w, h, score, category, truncation, occlusion

Categories we keep:
  1 = pedestrian  → 0 (person)
  2 = people      → 0 (person)
  4 = car         → 1 (car)
  5 = van         → 1 (car)
"""

import os
import cv2
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
VISDRONE_ROOT = "./VisDrone2019-DET-train"   # change to your dataset path
OUTPUT_ROOT   = "./visdrone_yolo"
VAL_SPLIT     = 0.15
SEED          = 42

# Only these VisDrone class IDs will be kept
VIS2YOLO = {1: 0, 2: 0, 4: 1, 5: 1}
CLASS_NAMES = ["person", "car"]

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def parse_visdrone_annotation(ann_path, img_w, img_h):
    """Return list of (yolo_class, cx, cy, w, h) for kept classes."""
    boxes = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score = int(parts[4])
            cls   = int(parts[5])

            if score == 0 or cls not in VIS2YOLO:
                continue
            if w <= 0 or h <= 0:
                continue

            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            boxes.append((VIS2YOLO[cls], cx, cy, nw, nh))
    return boxes


def convert_split(img_dir, ann_dir, out_img_dir, out_lbl_dir):
    """Convert all images + annotations in one split."""
    Path(out_img_dir).mkdir(parents=True, exist_ok=True)
    Path(out_lbl_dir).mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped   = 0

    for ann_file in sorted(Path(ann_dir).glob("*.txt")):
        stem = ann_file.stem

        # find corresponding image
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = Path(img_dir) / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        H, W = img.shape[:2]

        boxes = parse_visdrone_annotation(ann_file, W, H)
        if not boxes:          # skip images with no relevant objects
            skipped += 1
            continue

        # copy image
        shutil.copy(img_path, Path(out_img_dir) / img_path.name)

        # write YOLO label
        label_path = Path(out_lbl_dir) / (stem + ".txt")
        with open(label_path, "w") as f:
            for cls, cx, cy, w, h in boxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        converted += 1

    print(f"  Converted: {converted}  |  Skipped: {skipped}")
    return converted


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 55)
    print("VisDrone → YOLO Conversion")
    print("=" * 55)

    # ---- locate source splits ----
    # VisDrone typically has train/val folders; we merge & re-split
    source_pairs = []
    for split_name in ["VisDrone2019-DET-train", "VisDrone2019-DET-val"]:
        img_d = Path(f"./{split_name}/images")
        ann_d = Path(f"./{split_name}/annotations")
        if img_d.exists() and ann_d.exists():
            source_pairs.append((img_d, ann_d))

    if not source_pairs:
        # fallback: assume flat structure
        img_d = Path(VISDRONE_ROOT) / "images"
        ann_d = Path(VISDRONE_ROOT) / "annotations"
        source_pairs = [(img_d, ann_d)]

    # ---- collect all valid (image, annotation) pairs ----
    all_pairs = []
    for img_d, ann_d in source_pairs:
        for ann_file in sorted(ann_d.glob("*.txt")):
            stem = ann_file.stem
            for ext in (".jpg", ".jpeg", ".png"):
                img_file = img_d / (stem + ext)
                if img_file.exists():
                    all_pairs.append((img_file, ann_file))
                    break

    random.shuffle(all_pairs)
    n_val   = int(len(all_pairs) * VAL_SPLIT)
    val_pairs   = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    print(f"\nTotal images found : {len(all_pairs)}")
    print(f"Train              : {len(train_pairs)}")
    print(f"Val                : {len(val_pairs)}")

    def process_pairs(pairs, split):
        out_img = Path(OUTPUT_ROOT) / "images" / split
        out_lbl = Path(OUTPUT_ROOT) / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        ok = 0
        for img_path, ann_path in pairs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            boxes = parse_visdrone_annotation(ann_path, W, H)
            if not boxes:
                continue
            shutil.copy(img_path, out_img / img_path.name)
            with open(out_lbl / (ann_path.stem + ".txt"), "w") as f:
                for cls, cx, cy, w, h in boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            ok += 1
        print(f"  [{split}] written: {ok}")

    print("\n[train]")
    process_pairs(train_pairs, "train")
    print("[val]")
    process_pairs(val_pairs,   "val")

    # ---- write data.yaml ----
    yaml_path = Path(OUTPUT_ROOT) / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {Path(OUTPUT_ROOT).resolve()}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")
    print(f"\ndata.yaml written → {yaml_path}")

    # ---- sample visualizations ----
    visualize_samples(Path(OUTPUT_ROOT) / "images" / "train",
                      Path(OUTPUT_ROOT) / "labels" / "train")


def visualize_samples(img_dir, lbl_dir, n=6):
    """Plot sample images with YOLO bounding boxes overlaid."""
    img_files = list(img_dir.glob("*.jpg"))[:n]
    if not img_files:
        return

    cols = 3
    rows = (len(img_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = np.array(axes).flatten()

    colors = {"person": "lime", "car": "dodgerblue"}

    for ax, img_path in zip(axes, img_files):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        ax.imshow(img)

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in open(lbl_path):
                cls, cx, cy, w, h = map(float, line.split())
                cls = int(cls)
                x1 = (cx - w / 2) * W
                y1 = (cy - h / 2) * H
                name  = CLASS_NAMES[cls]
                color = colors[name]
                ax.add_patch(patches.Rectangle(
                    (x1, y1), w * W, h * H,
                    linewidth=1.5, edgecolor=color, facecolor="none"
                ))
                ax.text(x1, y1 - 3, name, color=color,
                        fontsize=7, fontweight="bold")

        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")

    for ax in axes[len(img_files):]:
        ax.axis("off")

    plt.tight_layout()
    out = Path("outputs/sample_visualizations.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSample visualizations saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
