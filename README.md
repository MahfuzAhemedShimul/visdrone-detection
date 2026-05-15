# Drone Human Detection & Counting System

A computer vision pipeline for detecting humans and cars in drone/aerial images using **YOLOv8m** fine-tuned on the **VisDrone 2019** dataset. Includes human counting logic, bounding box visualization, and full evaluation metrics.

> Antlings Internship Program — Technical Assessment (AI/ML)

---

## Demo Results

### Detection + Counting Output
![Detection Results](outputs/detection_results.png)

### Training Curves
![Training Curves](outputs/results.png)

---

## Results

| Metric | Score |
|---|---|
| **mAP@0.5** | **0.4273** |
| **mAP@0.5:0.95** | **0.2521** |
| **Precision** | **0.5459** |
| **Recall** | **0.4341** |

### Per-Class mAP@0.5

| Class | mAP@0.5 |
|---|---|
| 🚗 car | **0.806** |
| 🚌 bus | 0.599 |
| 🚐 van | 0.472 |
| 🚛 truck | 0.413 |
| 🛵 motor | 0.500 |
| 🚶 pedestrian | 0.473 |
| 👥 people | 0.362 |
| 🚲 bicycle | 0.174 |
| 🛺 tricycle | 0.315 |
| 🛺 awning-tricycle | 0.159 |

---

## Features

- **Task 1** — VisDrone dataset understanding and preprocessing
- **Task 2** — YOLOv8m fine-tuned on all 10 VisDrone classes
- **Task 3** — Real-time detection with bounding boxes and live human + car count overlay
- **Task 5** — Full evaluation: mAP, precision, recall, per-class metrics

---

## Project Structure

```
visdrone-detection/
├── prepare_dataset.py     # Task 1: dataset understanding + preprocessing
├── train.py               # Task 2: YOLOv8m fine-tuning
├── detect.py              # Task 3: inference + human counting + visualization
├── evaluate.py            # Task 5: metrics + visual reports
├── kaggle_notebook.ipynb  # End-to-end notebook (run on Kaggle)
├── requirements.txt
└── outputs/
    ├── detection_results.png
    └── results.png
```

---

## Dataset

**VisDrone 2019 Detection Dataset**
- 6,471 training images
- 548 validation images
- 1,580 test images
- Captured by drone cameras across 14 cities in China
- 10 object classes

Download: https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset

### Dataset Structure (on Kaggle)
```
VisDrone_Dataset/
├── VisDrone2019-DET-train/
│   ├── images/       ← 6471 .jpg files
│   └── labels/       ← 6471 .txt files (YOLO format)
├── VisDrone2019-DET-val/
│   ├── images/       ← 548 .jpg files
│   └── labels/       ← 548 .txt files
└── VisDrone2019-DET-test-dev/
```

### Classes Used

| ID | Class | Mapped To |
|---|---|---|
| 0 | pedestrian | person (human) |
| 1 | people | person (human) |
| 3 | car | car |
| 4 | van | car |

---

## Setup

```bash
git clone https://github.com/MahfuzAhemedShimul/visdrone-detection.git
cd visdrone-detection
pip install -r requirements.txt
```

---

## Task 1 — Dataset Understanding

**Key observations:**
- Very small objects — many humans appear as < 10×10 pixels from drone altitude
- Dense crowds with heavy occlusion between objects
- High class imbalance — far more pedestrians than vehicles
- Varying drone altitude causes significant scale differences
- Some images have poor lighting (night, overcast)

**Augmentation strategy applied during training:**
- Mosaic (4-image tiling) — most effective for tiny objects
- HSV jitter — handles lighting variation
- Horizontal flip
- Random scale ±50% — simulates altitude variation

---

## Task 2 — Model Training

**Model:** YOLOv8m (25.8M parameters, pretrained on COCO)

**Training configuration:**

| Parameter | Value | Reason |
|---|---|---|
| `imgsz` | 640 | Stable on T4 GPU |
| `batch` | 16 | Fits 14.9GB VRAM |
| `epochs` | 50 | Full convergence |
| `mosaic` | 1.0 | Best for small objects |
| `patience` | 15 | Early stopping |
| `optimizer` | AdamW | Auto-selected |

**Run training:**
```bash
python train.py
```

Or on Kaggle, open `kaggle_notebook.ipynb` and run all cells.

---

## Task 3 — Detection & Human Counting

```bash
# Single image
python detect.py --source image.jpg --weights runs/visdrone/yolov8m_exp/weights/best.pt

# Folder of images
python detect.py --source ./test_images/ --weights best.pt

# Video
python detect.py --source video.mp4 --weights best.pt
```

**Counting logic:**
- Green boxes = humans (pedestrian + people classes)
- Orange boxes = cars (car + van classes)
- Count displayed live in top-left corner of each image

**Sample output:**
- Sports court image: **126 humans detected**
- Road intersection: **12 humans, 37 cars detected**
- City street: **8 humans, 36 cars detected**

---

## Task 5 — Evaluation

```bash
python evaluate.py \
  --weights runs/visdrone/yolov8m_exp/weights/best.pt \
  --data    visdrone_yolo/data.yaml
```

**Inference speed:** ~13.6ms per image (≈73 FPS on T4 GPU)

---

## Strengths & Limitations

### Strengths
- YOLOv8m pretrained on COCO provides strong general detection priors
- Mosaic augmentation significantly helps small object detection
- Car detection achieves excellent 0.806 mAP
- Fast inference at ~73 FPS suitable for real-time drone applications

### Limitations
- Very small objects (< 8px) frequently missed — SAHI tiling would help
- People class lower mAP (0.362) due to extreme occlusion in crowds
- Bicycle hardest class (0.174) — looks similar to motorbikes from above
- Model trained at 640px — larger input would improve tiny object recall

### Challenges Faced
- VisDrone annotations use `score=0` for ignored regions that must be filtered
- GPU VRAM limitations required reducing from imgsz=1280 to imgsz=640
- Heavy class imbalance affects precision/recall tradeoff
- Dense scenes cause overlapping labels and missed detections

---

## Future Improvements

- **SAHI** (Sliced Inference) — tile images for sub-10px object detection
- **RT-DETR** — transformer-based detector, stronger on dense scenes
- **Larger input size** — imgsz=1280 with more GPU memory
- **More epochs** — model was still improving at epoch 50
- **Custom augmentation** — targeted augmentation for tiny objects

---

## References

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Zhu et al., "Vision Meets Drones: Past, Present and Future"
- [VisDrone Challenge Benchmark](http://aiskyeye.com/)
