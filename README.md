# 🚁 Drone Human Detection & Counting System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8m-Ultralytics-orange)
![Dataset](https://img.shields.io/badge/Dataset-VisDrone2019-green)
![mAP](https://img.shields.io/badge/mAP@0.5-0.4273-red)

A complete computer vision pipeline that detects **humans and cars** in drone/aerial images using **YOLOv8m** fine-tuned on the **VisDrone 2019** dataset. The system draws bounding boxes, counts total humans, and evaluates model performance with standard metrics.

> Built for the Antlings Internship Program — Technical Assessment (AI/ML)

---

## 📸 Results

### Detection + Human Counting
![Detection Results](outputs/detection_results.png)

### Training Curves (50 Epochs)
![Training Curves](outputs/results.png)

### Validation Batch Predictions
![Val Predictions](outputs/val_batch0_pred.jpg)

---

## 📊 Evaluation Metrics

| Metric | Score |
|---|---|
| **mAP@0.5** | **0.4273** |
| **mAP@0.5:0.95** | **0.2521** |
| **Precision** | **0.5459** |
| **Recall** | **0.4341** |
| **Inference Speed** | ~13.6ms/image (~73 FPS on T4) |

### Per-Class mAP@0.5

| Class | mAP@0.5 |
|---|---|
| 🚗 Car | **0.806** |
| 🚌 Bus | 0.599 |
| 🚐 Van | 0.472 |
| 🛵 Motor | 0.500 |
| 🚛 Truck | 0.413 |
| 🚶 Pedestrian | 0.473 |
| 👥 People | 0.362 |
| 🚲 Bicycle | 0.174 |
| 🛺 Tricycle | 0.315 |
| 🛺 Awning-Tricycle | 0.159 |

---

## 🗂️ Project Structure

```
visdrone-detection/
├── prepare_dataset.py      # Task 1: dataset understanding + YOLO conversion
├── train.py                # Task 2: YOLOv8m fine-tuning script
├── detect.py               # Task 3: inference + human counting + visualization
├── evaluate.py             # Task 5: mAP, precision, recall, charts
├── kaggle_notebook.ipynb   # End-to-end notebook (run directly on Kaggle)
├── requirements.txt        # All dependencies
├── README.md
└── outputs/
    ├── detection_results.png
    ├── results.png
    └── val_batch0_pred.jpg
```

---

## 📦 Dataset

**VisDrone 2019 Detection Dataset**

| Split | Images | Instances |
|---|---|---|
| Train | 6,471 | ~400,000 |
| Val | 548 | ~38,759 |
| Test-dev | 1,610 | — |

Download: https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset

### Dataset Folder Structure on Kaggle
```
VisDrone_Dataset/
├── VisDrone2019-DET-train/
│   ├── images/        ← 6471 drone images (.jpg)
│   └── labels/        ← 6471 annotations (YOLO format)
├── VisDrone2019-DET-val/
│   ├── images/        ← 548 images
│   └── labels/        ← 548 annotations
└── VisDrone2019-DET-test-dev/
```

### VisDrone Classes

| ID | Class | Role in this project |
|---|---|---|
| 0 | pedestrian | ✅ Counted as human |
| 1 | people | ✅ Counted as human |
| 2 | bicycle | Detected |
| 3 | car | ✅ Counted as car |
| 4 | van | ✅ Counted as car |
| 5 | truck | Detected |
| 6 | tricycle | Detected |
| 7 | awning-tricycle | Detected |
| 8 | bus | Detected |
| 9 | motor | Detected |

---

## ⚙️ Setup

```bash
git clone https://github.com/MahfuzAhemedShimul/visdrone-detection.git
cd visdrone-detection
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA GPU recommended

---

## 🔧 Task 1 — Dataset Understanding & Preprocessing

```bash
python prepare_dataset.py
```

**Dataset challenges identified:**
- Extremely small objects — many humans are < 10×10 pixels at drone altitude
- Dense crowds with heavy mutual occlusion
- High class imbalance — pedestrians far outnumber vehicles
- Scale variation from different drone altitudes
- Lighting inconsistencies (night, overcast, shadows)

**Preprocessing steps:**
1. Parse VisDrone annotation format (`x, y, w, h, score, class_id, truncation, occlusion`)
2. Filter `score=0` entries (ignored/occluded regions)
3. Convert to YOLO normalized format `(class cx cy w h)`
4. 85/15 train/val split
5. Save sample visualizations

**Augmentation strategy (applied during training):**
- Mosaic tiling — most effective technique for tiny objects
- HSV jitter — handles drone lighting variation
- Horizontal flip
- Random scale ±50% — simulates altitude change

---

## 🏋️ Task 2 — Model Training

**Model choice: YOLOv8m**

| Model | Params | Reason |
|---|---|---|
| YOLOv8n | 3.2M | Too small for tiny objects |
| **YOLOv8m** | **25.8M** | ✅ Best balance of accuracy + speed |
| YOLOv8l | 43.7M | Too slow for T4 GPU at useful batch size |

**Training configuration:**

| Parameter | Value | Reason |
|---|---|---|
| `imgsz` | 640 | Stable on T4 GPU (14.9GB VRAM) |
| `batch` | 16 | Fits VRAM comfortably |
| `epochs` | 50 | Full convergence |
| `mosaic` | 1.0 | Critical for small objects |
| `patience` | 15 | Early stopping |
| `optimizer` | AdamW (auto) | Best for this task |
| `lr0` | 0.000714 | Auto-tuned by Ultralytics |

```bash
python train.py
# or on Kaggle: open kaggle_notebook.ipynb
```

**Training observations:**
- Loss converged smoothly over 50 epochs
- mAP@0.5 improved from 0.14 → 0.4273
- No overfitting — val loss tracked train loss closely
- GPU memory peaked at ~11.6G (safe margin on T4)

---

## 🎯 Task 3 — Detection & Human Counting

```bash
# Single image
python detect.py --source image.jpg \
                 --weights runs/visdrone/yolov8m_exp/weights/best.pt

# Folder of images
python detect.py --source ./test_images/ --weights best.pt

# Video file
python detect.py --source drone_video.mp4 --weights best.pt
```

**How counting works:**
```python
PERSON_CLASSES = [0, 1]   # pedestrian + people → counted as "humans"
CAR_CLASSES    = [3, 4]   # car + van → counted as "cars"

person_count = sum(1 for box in results.boxes if int(box.cls[0]) in PERSON_CLASSES)
car_count    = sum(1 for box in results.boxes if int(box.cls[0]) in CAR_CLASSES)
```

**Visual output:**
- 🟩 Green boxes = humans
- 🟧 Orange boxes = cars
- Count overlay displayed top-left corner

**Sample detections:**
- Sports court: **126 humans** detected
- Road intersection: **12 humans, 37 cars**
- City street: **8 humans, 36 cars**

---

## 📈 Task 5 — Evaluation & Visualization

```bash
python evaluate.py \
  --weights runs/visdrone/yolov8m_exp/weights/best.pt \
  --data    visdrone_yolo/data.yaml
```

Generates:
- Metrics summary (mAP, precision, recall)
- Per-class performance breakdown
- Training curve plots
- Sample prediction grid
- Count distribution histogram

---

## ✅ Strengths

- YOLOv8m pretrained on COCO provides strong detection priors — fast convergence
- Mosaic augmentation significantly boosts small object detection
- Car detection achieves excellent **0.806 mAP** — near production quality
- Fast inference at ~73 FPS — suitable for real-time drone applications
- Clean modular code — easy to extend or modify

---

## ⚠️ Limitations

- Very small objects (< 8px height) frequently missed — needs SAHI tiling
- `people` class lower mAP (0.362) due to extreme occlusion in crowd scenes
- `bicycle` hardest class (0.174) — visually similar to motorbikes from above
- Trained at imgsz=640 — larger input (1280) would improve tiny object recall
- Model not tested on unseen geographic locations (may not generalize)

---

## 🔮 Future Improvements

- **SAHI** — Sliced inference for sub-10px object detection
- **RT-DETR** — Transformer-based detector, stronger on dense scenes
- **imgsz=1280** — With better GPU, larger input improves small objects
- **ByteTrack** — Add multi-object tracking for video inputs
- **Custom anchor tuning** — Re-cluster anchors on VisDrone's small boxes

---

## 📚 References

- [VisDrone Dataset Paper](https://github.com/VisDrone/VisDrone-Dataset)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- Zhu et al., "Vision Meets Drones: Past, Present and Future", ECCV 2018
- [VisDrone Challenge Leaderboard](http://aiskyeye.com/)
