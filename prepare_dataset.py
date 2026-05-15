"""
Task 1: VisDrone Dataset Understanding & Preprocessing

VisDrone annotation format per line:
  x, y, w, h, score, category_id, truncation, occlusion

All 10 classes:
  0=pedestrian, 1=people, 2=bicycle, 3=car, 4=van,
  5=truck, 6=tricycle, 7=awning-tricycle, 8=bus, 9=motor

Usage:
  python prepare_dataset.py
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
# Config
# ------------------------------------------------------------------ #
# Update BASE to your dataset location
BASE        = '/kaggle/input/datasets/banuprasadb/visdrone-dataset/VisDrone_Dataset'
OUTPUT_ROOT = './visdrone_yolo'
SEED        = 42

CLASS_NAMES = ['pedestrian','people','bicycle','car','van',
               'truck','tricycle','awning-tricycle','bus','motor']

CLASS_COLORS = [
    'lime','cyan','yellow','dodgerblue','orange',
    'red','magenta','white','pink','lightgreen'
]

# ------------------------------------------------------------------ #
# Parse annotation
# ------------------------------------------------------------------ #
def parse_visdrone_annotation(ann_path, img_w, img_h):
    """Parse VisDrone .txt → list of (class_id, cx, cy, w, h) normalized."""
    boxes = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score = int(parts[4])
            cls   = int(parts[5])

            if score == 0:          # ignore region
                continue
            if cls < 0 or cls > 9:  # unknown class
                continue
            if w <= 0 or h <= 0:
                continue

            cx = max(0.0, min(1.0, (x + w / 2) / img_w))
            cy = max(0.0, min(1.0, (y + h / 2) / img_h))
            nw = max(0.0, min(1.0, w / img_w))
            nh = max(0.0, min(1.0, h / img_h))
            boxes.append((cls, cx, cy, nw, nh))
    return boxes


# ------------------------------------------------------------------ #
# Dataset statistics
# ------------------------------------------------------------------ #
def print_dataset_stats(base_path):
    """Print class distribution and image counts."""
    print("=" * 55)
    print("VisDrone Dataset Statistics")
    print("=" * 55)

    for split in ['VisDrone2019-DET-train', 'VisDrone2019-DET-val']:
        ann_dir = Path(base_path) / split / 'labels'
        img_dir = Path(base_path) / split / 'images'

        if not ann_dir.exists():
            ann_dir = Path(base_path) / split / 'annotations'

        if not img_dir.exists():
            continue

        n_images = len(list(img_dir.glob('*.jpg')))
        class_counts = {i: 0 for i in range(10)}

        for ann_file in ann_dir.glob('*.txt'):
            img_path = img_dir / (ann_file.stem + '.jpg')
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            for cls, *_ in parse_visdrone_annotation(ann_file, W, H):
                class_counts[cls] += 1

        print(f"\n[{split}]")
        print(f"  Images: {n_images}")
        print(f"  {'Class':<20} {'Count':>8}")
        print(f"  {'-'*30}")
        for cls_id, count in class_counts.items():
            print(f"  {CLASS_NAMES[cls_id]:<20} {count:>8,}")


# ------------------------------------------------------------------ #
# Visualization
# ------------------------------------------------------------------ #
def visualize_samples(img_dir, lbl_dir, n=6, save_path='outputs/sample_visualizations.png'):
    """Plot sample images with bounding boxes."""
    img_files = [p for p in sorted(Path(img_dir).glob('*.jpg'))][:n]
    if not img_files:
        print(f"[WARN] No images found in {img_dir}")
        return

    cols = 3
    rows = (len(img_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = np.array(axes).flatten()
    fig.suptitle('VisDrone Sample Images with Annotations', fontsize=14)

    for ax, img_path in zip(axes, img_files):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        ax.imshow(img)

        lbl_path = Path(lbl_dir) / (img_path.stem + '.txt')
        if lbl_path.exists():
            for line in open(lbl_path):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                color = CLASS_COLORS[cls % len(CLASS_COLORS)]
                ax.add_patch(patches.Rectangle(
                    (x1, y1), bw * W, bh * H,
                    linewidth=1.5, edgecolor=color, facecolor='none'
                ))
                ax.text(x1, y1 - 2, CLASS_NAMES[cls],
                        color=color, fontsize=6, fontweight='bold')

        ax.set_title(img_path.name, fontsize=7)
        ax.axis('off')

    for ax in axes[len(img_files):]:
        ax.axis('off')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample visualizations saved → {save_path}")
    plt.show()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print_dataset_stats(BASE)

    train_img = Path(BASE) / 'VisDrone2019-DET-train' / 'images'
    train_lbl = Path(BASE) / 'VisDrone2019-DET-train' / 'labels'

    print("\nGenerating sample visualizations...")
    visualize_samples(train_img, train_lbl, n=6)


if __name__ == '__main__':
    main()
