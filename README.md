# Drone Human Detection & Counting System

A computer vision pipeline for detecting humans and cars in drone/aerial images using **YOLOv8** fine-tuned on the **VisDrone 2019** dataset. Includes counting logic, bounding box visualization, and optional **ByteTrack** object tracking.

> Antlings Internship Program — Technical Assessment (AI/ML)

---

## Demo

| Detection + Counting | ByteTrack Tracking |
|---|---|
| *(outputs/detections/)* | *(outputs/detections/*_tracked.mp4)* |

---

## Features

- **Task 1** — VisDrone dataset parsing and YOLO-format conversion with augmentation pipeline
- **Task 2** — YOLOv8m fine-tuned for `person` and `car` on drone imagery
- **Task 3** — Real-time detection with bounding boxes and live human count overlay
- **Task 4** — (Bonus) ByteTrack multi-object tracking with unique track IDs
- **Task 5** — Full evaluation: mAP, precision, recall, FPS, prediction grids

---

## Project Structure

```
visdrone-detection/
├── prepare_dataset.py   # Task 1: convert VisDrone → YOLO format + visualizations
├── train.py             # Task 2: fine-tune YOLOv8 with drone-optimized settings
├── detect.py            # Tasks 3 & 4: inference + counting + ByteTrack tracking
├── evaluate.py          # Task 5: metrics + visual reports
├── requirements.txt
├── outputs/
│   ├── sample_visualizations.png
│   ├── detections/
│   └── evaluation/
│       ├── metrics.txt
│       ├── metrics_chart.png
│       ├── sample_predictions.png
│       └── count_distribution.png
└── runs/
    └── visdrone/
        └── yolov8m_exp/
            └── weights/
                └── best.pt
```

---

## Setup

```bash
git clone https://github.com/<your-username>/visdrone-detection.git
cd visdrone-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**GPU recommended.** Tested on NVIDIA T4 (Kaggle / Google Colab) and RTX 3060.

---

## Dataset

Download from Kaggle:
```
https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset
```

Or via Kaggle CLI:
```bash
kaggle datasets download -d banuprasadb/visdrone-dataset
unzip visdrone-dataset.zip
```

**VisDrone annotation format** (space-separated per line):
```
x_min, y_min, width, height, score, category_id, truncation, occlusion
```

**Classes used** (others discarded):

| VisDrone ID | VisDrone Name | Our Label |
|---|---|---|
| 1 | pedestrian | person (0) |
| 2 | people | person (0) |
| 4 | car | car (1) |
| 5 | van | car (1) |

---

## Task 1 — Dataset Preprocessing

```bash
python prepare_dataset.py
```

This script:
1. Parses VisDrone `.txt` annotations
2. Filters to `person` and `car` classes
3. Converts coordinates to YOLO normalized format `(cx, cy, w, h)`
4. Creates an 85/15 train/val split
5. Writes `data.yaml`
6. Saves sample visualization images to `outputs/sample_visualizations.png`

**Dataset challenges noticed:**
- Very small objects (many boxes < 10×10 px in 1920×1080 images)
- Dense crowds causing heavy occlusion
- Varying altitude causing scale inconsistency
- Class imbalance: many more persons than cars
- Some images taken at night or in low contrast

**Augmentation strategy** (applied during training via Ultralytics):
- Mosaic (4-image tiling) — especially effective for tiny-object detection
- HSV jitter — handles lighting/exposure variation
- Horizontal flip
- Random scale (±50%) — simulates altitude variation
- Mixup and copy-paste

---

## Task 2 — Model Training

```bash
python train.py
# or customize:
python train.py --model yolov8s.pt --epochs 30 --batch 16 --device 0
```

**Model choice: YOLOv8m**

| Model | Params | mAP@0.5 (COCO) | Speed |
|---|---|---|---|
| YOLOv8n | 3.2M | 37.3 | fastest |
| **YOLOv8m** | **25.9M** | **50.2** | **balanced ✓** |
| YOLOv8l | 43.7M | 52.9 | slower |

YOLOv8m offers the best balance between accuracy (critical for tiny drone objects) and training speed within a Kaggle T4 budget.

**Key training settings:**

| Parameter | Value | Reason |
|---|---|---|
| `imgsz` | 1280 | Larger input preserves tiny objects |
| `mosaic` | 1.0 | Best augmentation for small objects |
| `batch` | 8 | Fits T4 16GB at 1280px |
| `epochs` | 50 | Converges well with early stopping |
| `lr0` | 0.01 | Standard SGD start |

**Training command on Kaggle:**
```python
from ultralytics import YOLO
model = YOLO("yolov8m.pt")
model.train(data="visdrone_yolo/data.yaml", epochs=50, imgsz=1280, batch=8)
```

---

## Task 3 — Detection & Human Counting

```bash
# Single image
python detect.py --source test.jpg --weights runs/visdrone/yolov8m_exp/weights/best.pt

# Folder of images
python detect.py --source ./test_images/ --weights best.pt

# Video file
python detect.py --source drone_video.mp4 --weights best.pt
```

**Counting logic:**  
After inference, bounding boxes are filtered by class index. `person` count = number of boxes where `class == 0`. Simple and fast — no centroid tracking or zone logic needed for images.

**Output overlay:**
- Green boxes = persons
- Blue boxes = cars
- Semi-transparent panel showing live counts + FPS

---

## Task 4 (Bonus) — ByteTrack Object Tracking

```bash
python detect.py --source drone_video.mp4 --weights best.pt --track
```

**Why ByteTrack?**
- Built directly into Ultralytics — no extra installation
- Handles partial occlusion well
- Associates detections across frames using IoU + confidence score
- Each person/car gets a unique persistent ID across frames

**Output:** annotated video with track IDs (`#ID class conf`) above each box.

---

## Task 5 — Evaluation

```bash
python evaluate.py --weights runs/visdrone/yolov8m_exp/weights/best.pt \
                   --data    visdrone_yolo/data.yaml \
                   --fps-bench
```

**Metrics generated:**
- mAP@0.5 (standard detection threshold)
- mAP@0.5:0.95 (COCO-style stricter metric)
- Precision and Recall
- FPS (inference speed on T4)
- Per-class mAP for `person` and `car`

**Visual outputs:**
- `metrics_chart.png` — bar chart of all metrics
- `sample_predictions.png` — grid of 6 val images with predicted boxes
- `count_distribution.png` — histogram of predicted object counts across val set

---

## Results

*(Fill in after training)*

| Metric | person | car | Overall |
|---|---|---|---|
| mAP@0.5 | — | — | — |
| mAP@0.5:0.95 | — | — | — |
| Precision | — | — | — |
| Recall | — | — | — |
| FPS (T4) | — | — | — |

---

## Strengths & Limitations

**Strengths:**
- YOLOv8m pre-trained on COCO gives strong general detection priors, accelerating convergence
- `imgsz=1280` preserves detail for tiny objects relative to standard 640px input
- Mosaic augmentation is particularly effective for dense small-object scenes
- ByteTrack tracking adds minimal latency (built-in, no extra model)

**Limitations:**
- Very small objects (< 8px) are still frequently missed — would need SAHI tiling for improvement
- Night / low-contrast images degrade detection quality
- Class imbalance (person >> car) can bias precision/recall tradeoff
- Single-scale inference; a multi-scale approach (SAHI) would help dense scenes

**Challenges:**
- VisDrone's annotation includes `score=0` (ignored/occluded) entries that must be filtered
- GPU memory management at `imgsz=1280` — batch size must be tuned per GPU
- Tracking ID switches during heavy occlusion (inherent ByteTrack limitation)

---

## Improvements (if more time)

- **SAHI** (Sliced Inference) — run inference on overlapping tiles and merge results for sub-10px objects
- **RT-DETR** — transformer-based detector, stronger at dense scenes
- **BoT-SORT** — tracking with ReID features, fewer ID switches
- **Custom anchor tuning** — YOLOv8's default anchors are COCO-tuned; re-clustering on VisDrone's small boxes would help
- **Counting zones** — polygon-based ROI counting rather than whole-frame totals

---

## References

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- Zhu et al., "Vision Meets Drones: Past, Present and Future" (VisDrone benchmark paper)
