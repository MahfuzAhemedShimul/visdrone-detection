"""
Task 2: YOLOv8m Fine-tuning on VisDrone 2019

Trains YOLOv8m on all 10 VisDrone classes with drone-optimized settings.

Usage:
  python train.py
  python train.py --epochs 30 --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
# Default config (matches what was actually trained)
# ------------------------------------------------------------------ #
CFG = {
    'model'   : 'yolov8m.pt',
    'data'    : '/kaggle/working/visdrone_yolo/data.yaml',
    'epochs'  : 50,
    'imgsz'   : 640,
    'batch'   : 16,
    'patience': 15,
    'mosaic'  : 1.0,
    'mixup'   : 0.0,
    'hsv_h'   : 0.015,
    'hsv_s'   : 0.7,
    'hsv_v'   : 0.4,
    'fliplr'  : 0.5,
    'scale'   : 0.5,
    'translate': 0.1,
    'degrees' : 0.0,
    'project' : '/kaggle/working/runs/visdrone',
    'name'    : 'yolov8m_exp',
    'device'  : 0,
    'workers' : 2,
}


# ------------------------------------------------------------------ #
# Train
# ------------------------------------------------------------------ #
def train(cfg):
    print("=" * 55)
    print("YOLOv8m Training — VisDrone 2019")
    print("=" * 55)
    print(f"  Model   : {cfg['model']}")
    print(f"  Epochs  : {cfg['epochs']}")
    print(f"  ImgSize : {cfg['imgsz']}")
    print(f"  Batch   : {cfg['batch']}")
    print(f"  Device  : {cfg['device']}")
    print()

    model = YOLO(cfg['model'])
    model.train(
        data      = cfg['data'],
        epochs    = cfg['epochs'],
        imgsz     = cfg['imgsz'],
        batch     = cfg['batch'],
        patience  = cfg['patience'],
        mosaic    = cfg['mosaic'],
        hsv_h     = cfg['hsv_h'],
        hsv_s     = cfg['hsv_s'],
        hsv_v     = cfg['hsv_v'],
        fliplr    = cfg['fliplr'],
        scale     = cfg['scale'],
        translate = cfg['translate'],
        degrees   = cfg['degrees'],
        project   = cfg['project'],
        name      = cfg['name'],
        device    = cfg['device'],
        workers   = cfg['workers'],
    )

    best = Path(cfg['project']) / cfg['name'] / 'weights' / 'best.pt'
    print(f"\nTraining complete!")
    print(f"Best weights → {best}")
    return best


# ------------------------------------------------------------------ #
# Validate
# ------------------------------------------------------------------ #
def validate(weights_path, data_yaml):
    print("\n" + "=" * 55)
    print("Validation Results")
    print("=" * 55)

    model   = YOLO(str(weights_path))
    metrics = model.val(data=data_yaml, split='val', verbose=True)

    print(f"\n  mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95  : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")

    return metrics


# ------------------------------------------------------------------ #
# Plot training curves
# ------------------------------------------------------------------ #
def plot_curves(run_dir):
    csv_path = Path(run_dir) / 'results.csv'
    if not csv_path.exists():
        return

    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('YOLOv8 Training Curves — VisDrone', fontsize=14)

    plots = [
        ('train/box_loss',       'val/box_loss',       'Box Loss'),
        ('train/cls_loss',       'val/cls_loss',       'Class Loss'),
        ('train/dfl_loss',       'val/dfl_loss',       'DFL Loss'),
        ('metrics/precision(B)', None,                  'Precision'),
        ('metrics/recall(B)',    None,                  'Recall'),
        ('metrics/mAP50(B)',     None,                  'mAP@0.5'),
    ]

    for ax, (t_col, v_col, title) in zip(axes.flatten(), plots):
        if t_col in df.columns:
            ax.plot(df['epoch'], df[t_col], label='train', color='#5563d4', linewidth=1.5)
        if v_col and v_col in df.columns:
            ax.plot(df['epoch'], df[v_col], label='val',
                    color='#e25c3c', linestyle='--', linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = Path(run_dir) / 'training_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Training curves saved → {out}")
    plt.show()


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',   default=CFG['model'])
    p.add_argument('--data',    default=CFG['data'])
    p.add_argument('--epochs',  type=int, default=CFG['epochs'])
    p.add_argument('--imgsz',   type=int, default=CFG['imgsz'])
    p.add_argument('--batch',   type=int, default=CFG['batch'])
    p.add_argument('--device',  default=CFG['device'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = {**CFG,
            'model' : args.model,
            'data'  : args.data,
            'epochs': args.epochs,
            'imgsz' : args.imgsz,
            'batch' : args.batch,
            'device': args.device}

    best    = train(cfg)
    metrics = validate(best, cfg['data'])
    run_dir = Path(cfg['project']) / cfg['name']
    plot_curves(run_dir)
