"""
Task 3: Human & Car Detection with Counting

Detects humans and cars in drone images/videos.
Draws bounding boxes and displays live human + car counts.

Usage:
  # Single image
  python detect.py --source image.jpg --weights best.pt

  # Folder of images
  python detect.py --source ./images/ --weights best.pt

  # Video
  python detect.py --source video.mp4 --weights best.pt

  # With ByteTrack tracking (bonus)
  python detect.py --source video.mp4 --weights best.pt --track
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------------ #
# Class configuration (VisDrone 10-class)
# ------------------------------------------------------------------ #
CLASS_NAMES = ['pedestrian','people','bicycle','car','van',
               'truck','tricycle','awning-tricycle','bus','motor']

# Classes counted as "humans"
PERSON_CLASSES = {0, 1}        # pedestrian, people

# Classes counted as "cars"
CAR_CLASSES    = {3, 4}        # car, van

# Box colors per class (BGR)
CLASS_COLORS = {
    0: (57,  255, 20),    # pedestrian  → green
    1: (0,   255, 128),   # people      → teal
    2: (255, 255, 0),     # bicycle     → yellow
    3: (30,  144, 255),   # car         → blue
    4: (255, 144, 30),    # van         → orange
    5: (255, 50,  50),    # truck       → red
    6: (200, 0,   200),   # tricycle    → purple
    7: (255, 200, 0),     # awning-tri  → gold
    8: (0,   200, 255),   # bus         → cyan
    9: (180, 180, 180),   # motor       → gray
}

CONF  = 0.25
IOU   = 0.45
OUT   = Path('outputs/detections')


# ------------------------------------------------------------------ #
# Drawing
# ------------------------------------------------------------------ #
def draw_box(img, x1, y1, x2, y2, label, color, track_id=None):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    tag = f'#{track_id} {label}' if track_id is not None else label
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
    cv2.putText(img, tag, (x1 + 3, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_overlay(img, persons, cars, fps=None):
    """Semi-transparent count panel."""
    h = 115 if fps is None else 140
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (255, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, f'Humans : {persons}', (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (57, 255, 20), 2, cv2.LINE_AA)
    cv2.putText(img, f'Cars   : {cars}', (18, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 144, 255), 2, cv2.LINE_AA)
    if fps:
        cv2.putText(img, f'FPS    : {fps:.1f}', (18, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 1, cv2.LINE_AA)


# ------------------------------------------------------------------ #
# Single image
# ------------------------------------------------------------------ #
def detect_image(model, img_path, save_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f'[WARN] Cannot read {img_path}')
        return

    t0      = time.perf_counter()
    results = model(img, conf=CONF, iou=IOU, verbose=False)[0]
    fps     = 1.0 / max(time.perf_counter() - t0, 1e-6)

    persons = cars = 0
    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        label = f'{CLASS_NAMES[cls]} {conf:.2f}'
        draw_box(img, x1, y1, x2, y2, label, color)
        if cls in PERSON_CLASSES: persons += 1
        elif cls in CAR_CLASSES:  cars    += 1

    draw_overlay(img, persons, cars, fps)

    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / Path(img_path).name), img)
    print(f'  {Path(img_path).name:<35} humans={persons:3d}  cars={cars:3d}  fps={fps:.1f}')


# ------------------------------------------------------------------ #
# Video / webcam
# ------------------------------------------------------------------ #
def detect_video(model, source, save_dir, use_track):
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        print(f'[ERROR] Cannot open {source}')
        return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    name   = Path(source).stem if not isinstance(source, int) else 'webcam'
    suffix = '_tracked' if use_track else '_detected'
    out_p  = save_dir / f'{name}{suffix}.mp4'
    writer = cv2.VideoWriter(str(out_p), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    frame_idx  = 0
    all_fps    = []
    mode       = 'ByteTrack' if use_track else 'Detect'
    print(f'\n[{mode}] {source}  ({W}x{H} @ {fps:.0f}fps)')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        if use_track:
            res = model.track(frame, conf=CONF, iou=IOU,
                              tracker='bytetrack.yaml', persist=True, verbose=False)[0]
        else:
            res = model(frame, conf=CONF, iou=IOU, verbose=False)[0]

        inf_fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        all_fps.append(inf_fps)

        persons = cars = 0
        for box in res.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color    = CLASS_COLORS.get(cls, (255, 255, 255))
            label    = f'{CLASS_NAMES[cls]} {conf:.2f}'
            track_id = int(box.id[0]) if use_track and box.id is not None else None
            draw_box(frame, x1, y1, x2, y2, label, color, track_id)
            if cls in PERSON_CLASSES: persons += 1
            elif cls in CAR_CLASSES:  cars    += 1

        draw_overlay(frame, persons, cars, inf_fps)
        writer.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f'  frame {frame_idx:5d}  humans={persons}  cars={cars}  fps={inf_fps:.1f}')

        cv2.imshow('VisDrone Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f'\n  Saved → {out_p}')
    print(f'  Avg FPS: {np.mean(all_fps):.1f}')


# ------------------------------------------------------------------ #
# Folder
# ------------------------------------------------------------------ #
def detect_folder(model, folder, save_dir):
    exts  = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [p for p in sorted(Path(folder).rglob('*')) if p.suffix.lower() in exts]
    if not files:
        print(f'[WARN] No images in {folder}')
        return
    print(f'\n[Detect] {len(files)} images')
    for f in files:
        detect_image(model, f, save_dir)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source',  required=True)
    p.add_argument('--weights', default='runs/visdrone/yolov8m_exp/weights/best.pt')
    p.add_argument('--track',   action='store_true')
    p.add_argument('--conf',    type=float, default=CONF)
    p.add_argument('--output',  default=str(OUT))
    return p.parse_args()


if __name__ == '__main__':
    args     = parse_args()
    CONF     = args.conf
    save_dir = Path(args.output)

    print(f'Loading: {args.weights}')
    model = YOLO(args.weights)

    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    try:
        src = int(args.source)
        detect_video(model, src, save_dir, args.track)
    except ValueError:
        src = Path(args.source)
        if src.is_dir():
            detect_folder(model, src, save_dir)
        elif src.suffix.lower() in VIDEO_EXTS:
            detect_video(model, src, save_dir, args.track)
        else:
            detect_image(model, src, save_dir)
