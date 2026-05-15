"""
Task 5: Evaluation & Visualization

Runs validation, prints metrics, and saves visual reports.

Usage:
  python evaluate.py --weights runs/visdrone/yolov8m_exp/weights/best.pt
                     --data    /kaggle/working/visdrone_yolo/data.yaml
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

CLASS_NAMES = ['pedestrian','people','bicycle','car','van',
               'truck','tricycle','awning-tricycle','bus','motor']

COLORS_MPL  = ['lime','cyan','yellow','dodgerblue','orange',
               'red','magenta','white','pink','lightgreen']

OUT = Path('outputs/evaluation')


# ------------------------------------------------------------------ #
# Validation metrics
# ------------------------------------------------------------------ #
def run_val(model, data_yaml, split='val'):
    print("=" * 55)
    print(f"Validation — {split} split")
    print("=" * 55)

    metrics = model.val(data=data_yaml, split=split, verbose=True)

    res = {
        'mAP50'    : float(metrics.box.map50),
        'mAP50_95' : float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall'   : float(metrics.box.mr),
    }
    for i, name in enumerate(CLASS_NAMES):
        try:
            res[f'ap50_{name}'] = float(metrics.box.maps[i])
        except Exception:
            pass

    print(f"\n  mAP@0.5       : {res['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95  : {res['mAP50_95']:.4f}")
    print(f"  Precision     : {res['precision']:.4f}")
    print(f"  Recall        : {res['recall']:.4f}")
    print("\n  Per-class mAP@0.5:")
    for name in CLASS_NAMES:
        k = f'ap50_{name}'
        if k in res:
            print(f"    {name:<20}: {res[k]:.4f}")

    return res, metrics


# ------------------------------------------------------------------ #
# FPS benchmark
# ------------------------------------------------------------------ #
def benchmark_fps(model, img_dir, n=100):
    files = list(Path(img_dir).glob('*.jpg'))[:n]
    if not files:
        return None
    times = []
    for f in files:
        img = cv2.imread(str(f))
        t0  = time.perf_counter()
        model(img, conf=0.25, verbose=False)
        times.append(time.perf_counter() - t0)
    fps = 1.0 / np.mean(times)
    print(f"\n  FPS benchmark ({len(files)} images): {fps:.1f} FPS")
    return fps


# ------------------------------------------------------------------ #
# Plot metrics bar chart
# ------------------------------------------------------------------ #
def plot_metrics(res, out_dir):
    labels = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
    values = [res['precision'], res['recall'], res['mAP50'], res['mAP50_95']]
    colors = ['#5a8dee', '#51d0a0', '#f4a44f', '#e8605a']

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Detection Metrics — VisDrone Val Split', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = out_dir / 'metrics_chart.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Metrics chart → {out}")
    plt.show()


# ------------------------------------------------------------------ #
# Sample prediction grid
# ------------------------------------------------------------------ #
def plot_predictions(model, img_dir, out_dir, n=6):
    files = [p for p in sorted(Path(img_dir).glob('*.jpg'))][:n]
    if not files:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = np.array(axes).flatten()
    fig.suptitle('Sample Predictions — Val Images', fontsize=14)

    for ax, img_path in zip(axes, files):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        ax.imshow(img)

        results = model(img, conf=0.25, verbose=False)[0]
        persons = 0
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS_MPL[cls % len(COLORS_MPL)]
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=1.2, edgecolor=color, facecolor='none'))
            ax.text(x1, y1-2, f'{CLASS_NAMES[cls]} {conf:.2f}',
                    color=color, fontsize=5.5, fontweight='bold')
            if cls in {0, 1}:
                persons += 1

        ax.set_title(f'{img_path.name} | humans: {persons}', fontsize=7)
        ax.axis('off')

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    out = out_dir / 'sample_predictions.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Predictions grid → {out}")
    plt.show()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='runs/visdrone/yolov8m_exp/weights/best.pt')
    p.add_argument('--data',    default='/kaggle/working/visdrone_yolo/data.yaml')
    p.add_argument('--imgdir',  default=None, help='val images dir for FPS benchmark')
    return p.parse_args()


if __name__ == '__main__':
    args    = parse_args()
    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    # Metrics
    res, metrics = run_val(model, args.data)

    # FPS
    if args.imgdir:
        fps = benchmark_fps(model, args.imgdir)
        if fps:
            res['fps'] = fps

    # Save metrics to text
    with open(out_dir / 'metrics.txt', 'w') as f:
        for k, v in res.items():
            f.write(f'{k}: {v:.4f}\n')
    print(f"\n  Metrics saved → {out_dir}/metrics.txt")

    # Plots
    plot_metrics(res, out_dir)

    if args.imgdir:
        plot_predictions(model, args.imgdir, out_dir)

    print(f'\n✅ All outputs saved → {out_dir}')
