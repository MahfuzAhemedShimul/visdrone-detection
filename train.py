"""
Task 2: YOLOv8 Fine-tuning on VisDrone
Fine-tunes YOLOv8m on person + car classes with drone-optimized settings.

Run: python train.py
     python train.py --model yolov8s.pt --epochs 30  (faster, for testing)
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
DEFAULT_CONFIG = {
    "model"     : "yolov8m.pt",   # m = medium; use yolov8s.pt for faster runs
    "data"      : "./visdrone_yolo/data.yaml",
    "epochs"    : 50,
    "imgsz"     : 1280,           # large res catches tiny drone objects
    "batch"     : 8,              # reduce to 4 if OOM on T4
    "patience"  : 15,             # early stopping
    "lr0"       : 0.01,
    "lrf"       : 0.001,
    "warmup_epochs": 3,
    "mosaic"    : 1.0,            # mosaic augmentation — great for small objects
    "mixup"     : 0.1,
    "copy_paste": 0.1,
    "hsv_h"     : 0.015,
    "hsv_s"     : 0.7,
    "hsv_v"     : 0.4,
    "fliplr"    : 0.5,
    "scale"     : 0.5,
    "translate" : 0.1,
    "degrees"   : 0.0,            # aerial images rarely need rotation
    "project"   : "runs/visdrone",
    "name"      : "yolov8m_exp",
    "exist_ok"  : True,
    "save_period": 10,            # checkpoint every N epochs
    "device"    : 0,              # GPU 0; set "cpu" for CPU-only
    "workers"   : 4,
    "verbose"   : True,
}


# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #
def train(cfg: dict):
    print("=" * 55)
    print("YOLOv8 Training — VisDrone")
    print("=" * 55)
    for k, v in cfg.items():
        print(f"  {k:<18} = {v}")
    print()

    # Load model (downloads pretrained weights automatically)
    model = YOLO(cfg["model"])

    # Start training
    results = model.train(
        data        = cfg["data"],
        epochs      = cfg["epochs"],
        imgsz       = cfg["imgsz"],
        batch       = cfg["batch"],
        patience    = cfg["patience"],
        lr0         = cfg["lr0"],
        lrf         = cfg["lrf"],
        warmup_epochs = cfg["warmup_epochs"],
        mosaic      = cfg["mosaic"],
        mixup       = cfg["mixup"],
        copy_paste  = cfg["copy_paste"],
        hsv_h       = cfg["hsv_h"],
        hsv_s       = cfg["hsv_s"],
        hsv_v       = cfg["hsv_v"],
        fliplr      = cfg["fliplr"],
        scale       = cfg["scale"],
        translate   = cfg["translate"],
        degrees     = cfg["degrees"],
        project     = cfg["project"],
        name        = cfg["name"],
        exist_ok    = cfg["exist_ok"],
        save_period = cfg["save_period"],
        device      = cfg["device"],
        workers     = cfg["workers"],
        verbose     = cfg["verbose"],
    )

    best_weights = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights → {best_weights}")
    return model, results, best_weights


# ------------------------------------------------------------------ #
# Validation
# ------------------------------------------------------------------ #
def validate(model, data_yaml, best_weights):
    print("\n" + "=" * 55)
    print("Validation on val split")
    print("=" * 55)

    val_model = YOLO(str(best_weights))
    metrics   = val_model.val(data=data_yaml, split="val", verbose=True)

    names  = ["person", "car"]
    map50  = metrics.box.map50
    map5095= metrics.box.map
    prec   = metrics.box.mp
    rec    = metrics.box.mr

    print(f"\n  mAP@0.5      : {map50:.4f}")
    print(f"  mAP@0.5:0.95 : {map5095:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")

    # per-class
    print("\n  Per-class mAP@0.5:")
    for i, name in enumerate(names):
        try:
            print(f"    {name:<10} : {metrics.box.maps[i]:.4f}")
        except Exception:
            pass

    return metrics


# ------------------------------------------------------------------ #
# Plot training curves
# ------------------------------------------------------------------ #
def plot_results(run_dir: Path):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return

    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("YOLOv8 Training Curves — VisDrone", fontsize=14)

    pairs = [
        ("train/box_loss",  "val/box_loss",  "Box Loss"),
        ("train/cls_loss",  "val/cls_loss",  "Class Loss"),
        ("train/dfl_loss",  "val/dfl_loss",  "DFL Loss"),
        ("metrics/precision(B)", None,        "Precision"),
        ("metrics/recall(B)",    None,        "Recall"),
        ("metrics/mAP50(B)",     None,        "mAP@0.5"),
    ]

    for ax, (train_col, val_col, title) in zip(axes.flatten(), pairs):
        if train_col in df.columns:
            ax.plot(df["epoch"], df[train_col], label="train", color="#5563d4")
        if val_col and val_col in df.columns:
            ax.plot(df["epoch"], df[val_col], label="val", color="#e25c3c", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = run_dir / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nTraining curves saved → {out}")
    plt.show()


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 VisDrone training")
    p.add_argument("--model",   default=DEFAULT_CONFIG["model"])
    p.add_argument("--data",    default=DEFAULT_CONFIG["data"])
    p.add_argument("--epochs",  type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--imgsz",   type=int, default=DEFAULT_CONFIG["imgsz"])
    p.add_argument("--batch",   type=int, default=DEFAULT_CONFIG["batch"])
    p.add_argument("--device",  default=DEFAULT_CONFIG["device"])
    p.add_argument("--name",    default=DEFAULT_CONFIG["name"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = {**DEFAULT_CONFIG,
            "model": args.model,
            "data" : args.data,
            "epochs": args.epochs,
            "imgsz" : args.imgsz,
            "batch" : args.batch,
            "device": args.device,
            "name"  : args.name}

    model, results, best_weights = train(cfg)
    metrics  = validate(model, cfg["data"], best_weights)
    run_dir  = Path(cfg["project"]) / cfg["name"]
    plot_results(run_dir)
