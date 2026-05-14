"""
Task 5: Evaluation & Visualization
Runs val split evaluation, prints metrics, and generates visual reports.

Usage:
  python evaluate.py --weights runs/visdrone/yolov8m_exp/weights/best.pt
                     --data    visdrone_yolo/data.yaml
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

CLASS_NAMES  = ["person", "car"]
CLASS_COLORS_MPL = {0: "lime", 1: "dodgerblue"}
CLASS_COLORS_CV  = {0: (57, 255, 20), 1: (30, 144, 255)}
OUTPUT_DIR   = Path("outputs/evaluation")


# ------------------------------------------------------------------ #
# Quantitative evaluation
# ------------------------------------------------------------------ #
def run_validation(model, data_yaml: str, split="val") -> dict:
    print("=" * 55)
    print(f"Validation — split: {split}")
    print("=" * 55)

    metrics = model.val(data=data_yaml, split=split, verbose=True)

    results = {
        "mAP50"     : float(metrics.box.map50),
        "mAP50_95"  : float(metrics.box.map),
        "precision" : float(metrics.box.mp),
        "recall"    : float(metrics.box.mr),
    }

    # per-class mAP
    for i, name in enumerate(CLASS_NAMES):
        try:
            results[f"mAP50_{name}"] = float(metrics.box.maps[i])
        except Exception:
            pass

    print("\n  ── Summary ──────────────────────────")
    print(f"  mAP@0.5       : {results['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95  : {results['mAP50_95']:.4f}")
    print(f"  Precision     : {results['precision']:.4f}")
    print(f"  Recall        : {results['recall']:.4f}")
    for name in CLASS_NAMES:
        k = f"mAP50_{name}"
        if k in results:
            print(f"  mAP@0.5 {name:<8}: {results[k]:.4f}")

    return results


# ------------------------------------------------------------------ #
# FPS benchmark
# ------------------------------------------------------------------ #
def benchmark_fps(model, data_yaml: str, n_images=50):
    import yaml
    cfg     = yaml.safe_load(open(data_yaml))
    img_dir = Path(cfg.get("path", ".")) / cfg.get("val", "images/val")
    # remove "images" duplicate if val already has it
    if not img_dir.exists():
        img_dir = Path(cfg.get("path", ".")) / "images" / cfg.get("val", "val")

    exts  = {".jpg", ".jpeg", ".png"}
    files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts][:n_images]

    if not files:
        print("[WARN] No images found for FPS benchmark")
        return None

    times = []
    for f in files:
        img = cv2.imread(str(f))
        t0  = time.perf_counter()
        model(img, conf=0.25, verbose=False)
        times.append(time.perf_counter() - t0)

    avg_fps = 1.0 / np.mean(times)
    print(f"\n  FPS benchmark ({len(files)} images): {avg_fps:.1f} FPS")
    return avg_fps


# ------------------------------------------------------------------ #
# Visual: metrics bar chart
# ------------------------------------------------------------------ #
def plot_metrics(results: dict, out_dir: Path):
    labels = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
    values = [results["precision"], results["recall"],
              results["mAP50"],     results["mAP50_95"]]
    colors = ["#5a8dee", "#51d0a0", "#f4a44f", "#e8605a"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Detection Metrics — VisDrone Val Split", fontsize=13, pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # per-class mAP
    per_class = {k: v for k, v in results.items() if k.startswith("mAP50_") and k != "mAP50"}
    if per_class:
        ax2 = ax.inset_axes([0.72, 0.55, 0.25, 0.38])
        names = [k.replace("mAP50_", "") for k in per_class]
        vals  = list(per_class.values())
        ax2.barh(names, vals, color=["#51d0a0", "#5a8dee"], height=0.5)
        ax2.set_xlim(0, 1)
        ax2.set_title("mAP@0.5\nper class", fontsize=8)
        ax2.tick_params(labelsize=8)
        for i, v in enumerate(vals):
            ax2.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
        ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = out_dir / "metrics_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Metrics chart → {out}")
    plt.show()


# ------------------------------------------------------------------ #
# Visual: sample predictions grid
# ------------------------------------------------------------------ #
def plot_predictions(model, data_yaml: str, out_dir: Path, n=6, conf=0.25):
    import yaml
    cfg     = yaml.safe_load(open(data_yaml))
    img_dir = Path(cfg.get("path", ".")) / cfg.get("val", "images/val")
    if not img_dir.exists():
        img_dir = Path(cfg.get("path", ".")) / "images" / cfg.get("val", "val")

    exts  = {".jpg", ".jpeg", ".png"}
    files = [p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in exts][:n]

    if not files:
        print("[WARN] No val images for prediction grid")
        return

    cols = 3
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = np.array(axes).flatten()
    fig.suptitle("Sample Predictions on Val Images", fontsize=14, y=1.01)

    for ax, img_path in zip(axes, files):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        ax.imshow(img)

        results = model(img, conf=conf, verbose=False)[0]
        persons = 0
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color  = CLASS_COLORS_MPL[cls]
            label  = f"{CLASS_NAMES[cls]} {conf_val:.2f}"
            rect   = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                        linewidth=1.5, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1 - 3, label, color=color, fontsize=6, fontweight="bold")
            if cls == 0:
                persons += 1

        ax.set_title(f"{img_path.name}  |  humans: {persons}", fontsize=8)
        ax.axis("off")

    for ax in axes[len(files):]:
        ax.axis("off")

    plt.tight_layout()
    out = out_dir / "sample_predictions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Prediction grid → {out}")
    plt.show()


# ------------------------------------------------------------------ #
# Visual: counting histogram
# ------------------------------------------------------------------ #
def plot_count_distribution(model, data_yaml: str, out_dir: Path, max_images=200):
    import yaml
    cfg     = yaml.safe_load(open(data_yaml))
    img_dir = Path(cfg.get("path", ".")) / cfg.get("val", "images/val")
    if not img_dir.exists():
        img_dir = Path(cfg.get("path", ".")) / "images" / cfg.get("val", "val")

    exts  = {".jpg", ".jpeg", ".png"}
    files = [p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in exts][:max_images]
    if not files:
        return

    person_counts = []
    car_counts    = []

    print(f"\nCounting objects in {len(files)} val images...")
    for img_path in files:
        img     = cv2.imread(str(img_path))
        results = model(img, conf=0.25, verbose=False)[0]
        p = sum(1 for b in results.boxes if int(b.cls[0]) == 0)
        c = sum(1 for b in results.boxes if int(b.cls[0]) == 1)
        person_counts.append(p)
        car_counts.append(c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Predicted Object Count Distribution — Val Split", fontsize=12)

    for ax, counts, label, color in [
        (ax1, person_counts, "Persons per image", "lime"),
        (ax2, car_counts,    "Cars per image",    "dodgerblue"),
    ]:
        ax.hist(counts, bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(counts), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {np.mean(counts):.1f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Number of images")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = out_dir / "count_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Count distribution → {out}")
    plt.show()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and visualize VisDrone model")
    p.add_argument("--weights", default="runs/visdrone/yolov8m_exp/weights/best.pt")
    p.add_argument("--data",    default="visdrone_yolo/data.yaml")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--split",   default="val")
    p.add_argument("--fps-bench", action="store_true", help="run FPS benchmark")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.weights}\n")
    model = YOLO(args.weights)

    # 1. Quantitative metrics
    results = run_validation(model, args.data, args.split)

    # 2. FPS benchmark
    if args.fps_bench:
        fps = benchmark_fps(model, args.data)
        if fps:
            results["fps"] = fps

    # 3. Save metrics to txt
    with open(out_dir / "metrics.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\n  Metrics saved → {out_dir}/metrics.txt")

    # 4. Visualizations
    plot_metrics(results, out_dir)
    plot_predictions(model, args.data, out_dir, n=6, conf=args.conf)
    plot_count_distribution(model, args.data, out_dir)

    print("\n[Done] All evaluation outputs saved to:", out_dir)
