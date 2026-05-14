"""
Tasks 3 & 4: Detection + Human Counting + (Optional) ByteTrack Tracking

Usage:
  # Single image
  python detect.py --source image.jpg --weights runs/visdrone/yolov8m_exp/weights/best.pt

  # Folder of images
  python detect.py --source ./test_images/ --weights best.pt

  # Video with tracking (Task 4 bonus)
  python detect.py --source video.mp4 --weights best.pt --track

  # Webcam
  python detect.py --source 0 --weights best.pt --track
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
CLASS_NAMES  = ["person", "car"]
CLASS_COLORS = {
    0: (57, 255, 20),    # person → bright green (BGR)
    1: (30, 144, 255),   # car    → dodger blue  (BGR)
}
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
OUTPUT_DIR     = Path("outputs/detections")


# ------------------------------------------------------------------ #
# Drawing utilities
# ------------------------------------------------------------------ #
def draw_box(frame, x1, y1, x2, y2, label, color, track_id=None):
    """Draw bounding box with filled label tag."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    tag = f"#{track_id} {label}" if track_id is not None else label
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    tag_y = max(y1 - 4, th + 4)
    cv2.rectangle(frame, (x1, tag_y - th - 4), (x1 + tw + 6, tag_y + 2), color, -1)
    cv2.putText(frame, tag, (x1 + 3, tag_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_count_overlay(frame, person_count, car_count, fps=None):
    """Draw semi-transparent count panel in top-left corner."""
    H, W = frame.shape[:2]
    panel_h = 110 if fps is None else 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (260, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Humans : {person_count}", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, CLASS_COLORS[0], 2, cv2.LINE_AA)
    cv2.putText(frame, f"Cars   : {car_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, CLASS_COLORS[1], 2, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(frame, f"FPS    : {fps:.1f}", (20, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 1, cv2.LINE_AA)


# ------------------------------------------------------------------ #
# Single-image inference (Task 3)
# ------------------------------------------------------------------ #
def detect_image(model, img_path: Path, save_dir: Path) -> dict:
    img   = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARN] Cannot read {img_path}")
        return {}

    t0      = time.perf_counter()
    results = model(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
    fps     = 1.0 / (time.perf_counter() - t0)

    person_count = 0
    car_count    = 0

    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS[cls]
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        draw_box(img, x1, y1, x2, y2, label, color)
        if cls == 0: person_count += 1
        else:        car_count    += 1

    draw_count_overlay(img, person_count, car_count, fps)

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / img_path.name
    cv2.imwrite(str(out_path), img)

    stats = {"file": img_path.name, "persons": person_count, "cars": car_count, "fps": fps}
    print(f"  {img_path.name:30s}  persons={person_count:3d}  cars={car_count:3d}  fps={fps:.1f}")
    return stats


# ------------------------------------------------------------------ #
# Video inference with ByteTrack (Tasks 3 + 4)
# ------------------------------------------------------------------ #
def detect_video(model, source, save_dir: Path, use_tracking: bool):
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out_name = (Path(source).stem if not isinstance(source, int) else "webcam")
    suffix   = "_tracked" if use_tracking else "_detected"
    out_path = save_dir / f"{out_name}{suffix}.mp4"
    writer   = cv2.VideoWriter(str(out_path),
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (W, H))

    frame_idx    = 0
    total_fps    = []
    mode_str     = "ByteTrack" if use_tracking else "Detect"
    print(f"\n[{mode_str}] Processing: {source}")
    print(f"  Resolution  : {W}x{H} @ {fps:.1f}fps")
    print(f"  Output      : {out_path}")

    # For non-tracking mode, keep it simple
    run_fn = model.track if use_tracking else model

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        if use_tracking:
            results = model.track(
                frame,
                conf    = CONF_THRESHOLD,
                iou     = IOU_THRESHOLD,
                tracker = "bytetrack.yaml",
                persist = True,
                verbose = False,
            )[0]
        else:
            results = model(
                frame,
                conf    = CONF_THRESHOLD,
                iou     = IOU_THRESHOLD,
                verbose = False,
            )[0]

        inf_fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        total_fps.append(inf_fps)

        person_count = 0
        car_count    = 0

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS[cls]
            label = f"{CLASS_NAMES[cls]} {conf:.2f}"

            track_id = None
            if use_tracking and box.id is not None:
                track_id = int(box.id[0])

            draw_box(frame, x1, y1, x2, y2, label, color, track_id)
            if cls == 0: person_count += 1
            else:        car_count    += 1

        draw_count_overlay(frame, person_count, car_count, inf_fps)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx:5d}  persons={person_count}  cars={car_count}  fps={inf_fps:.1f}")

        # live preview (press q to quit)
        cv2.imshow("VisDrone Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    avg_fps = float(np.mean(total_fps)) if total_fps else 0
    print(f"\n  Done. Frames={frame_idx}  Avg FPS={avg_fps:.1f}")
    print(f"  Saved → {out_path}")


# ------------------------------------------------------------------ #
# Batch image processing
# ------------------------------------------------------------------ #
def detect_folder(model, folder: Path, save_dir: Path):
    exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts]

    if not files:
        print(f"[WARN] No images found in {folder}")
        return

    print(f"\n[Detect] {len(files)} images in {folder}")
    all_stats = []
    for f in files:
        stats = detect_image(model, f, save_dir)
        if stats:
            all_stats.append(stats)

    # Summary
    total_persons = sum(s["persons"] for s in all_stats)
    total_cars    = sum(s["cars"]    for s in all_stats)
    avg_fps       = float(np.mean([s["fps"] for s in all_stats])) if all_stats else 0
    print(f"\n{'=' * 45}")
    print(f"  Images processed : {len(all_stats)}")
    print(f"  Total humans     : {total_persons}")
    print(f"  Total cars       : {total_cars}")
    print(f"  Avg FPS          : {avg_fps:.1f}")
    print(f"  Results in       : {save_dir}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser(description="VisDrone Detect + Count (+ Track)")
    p.add_argument("--source",  required=True,
                   help="image path | image folder | video path | webcam index (0)")
    p.add_argument("--weights", default="runs/visdrone/yolov8m_exp/weights/best.pt",
                   help="path to trained weights")
    p.add_argument("--track",   action="store_true",
                   help="enable ByteTrack object tracking (video/webcam only)")
    p.add_argument("--conf",    type=float, default=CONF_THRESHOLD)
    p.add_argument("--iou",     type=float, default=IOU_THRESHOLD)
    p.add_argument("--output",  default=str(OUTPUT_DIR))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    global CONF_THRESHOLD, IOU_THRESHOLD
    CONF_THRESHOLD = args.conf
    IOU_THRESHOLD  = args.iou

    print(f"\nLoading weights: {args.weights}")
    model     = YOLO(args.weights)
    save_dir  = Path(args.output)
    source    = args.source

    # Determine input type
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    try:
        src_int = int(source)   # webcam index
        detect_video(model, src_int, save_dir, args.track)
    except ValueError:
        src_path = Path(source)
        if src_path.is_dir():
            detect_folder(model, src_path, save_dir)
        elif src_path.suffix.lower() in VIDEO_EXTS:
            detect_video(model, src_path, save_dir, args.track)
        else:
            detect_image(model, src_path, save_dir)
