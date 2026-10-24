# Deep Learning-Based Rubber Detection: Industrial Quality Inspection & Plantation Monitoring

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8%20%7C%20YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-Academic%20Research-green.svg)](#acknowledgments)

---

## 📌 Executive Summary

This repository hosts the research and benchmark implementation of **Deep Learning-Based Rubber Sheet & Tree Tapping Detection for Industrial Quality Inspection and Plantation Monitoring**. 

Carried out as an internship research project at the **Indian Institute of Information Technology Kottayam (IIIT Kottayam)** under the mentorship of **Dr. Dhakshayani** by **Sanjay S** (*Chinmaya Vishwa Vidyapeeth*), this work systematically evaluates cutting-edge one-stage object detectors (**YOLOv8**, **YOLO11n**, **YOLO11s**) against a classic two-stage detector (**Faster R-CNN with ResNet-50-FPN**) on custom-collected rubber domain datasets.

### Primary Objectives
1. **Industrial Quality Inspection**: Automate surface defect inspection and boundary detection of manufactured rubber sheets in factory production lines under challenging illumination, shadow, and dust conditions.
2. **Plantation Agriculture Monitoring**: Detect latex tapping cuts, latex flow lines, and tree bark on *Hevea brasiliensis* rubber trees to assist automated harvesting and yield monitoring.
3. **Architectural Benchmarking**: Quantify trade-offs between inference latency, training duration, bounding box precision, recall, and mean Average Precision (mAP@50 and mAP@50-95).

---

## 📊 Benchmark Results & Comparative Analysis

All four models were trained and validated on identical dataset splits using NVIDIA GPU hardware. 

| Model | Detector Type | Framework | Resolution | Precision | Recall | mAP@50 | mAP@50-95 | Training Time | Recommended Use Case |
|---|---|---|---|---|---|---|---|---|---|
| **YOLO11s** | One-Stage | Ultralytics | 640 × 640 | **0.87** | **0.71** | **0.80** | **0.43** | 10 hrs | 🏆 **Best Overall Accuracy** (High-precision industrial sorting) |
| **YOLO11n** | One-Stage | Ultralytics | 640 × 640 | **0.85** | **0.70** | **0.79** | **0.42** | 8 hrs | ⚡ **Best for Edge/Real-Time** (High FPS, low-power devices) |
| **YOLOv8** | One-Stage | Ultralytics | 640 × 640 | 0.70 | 0.59 | 0.62 | 0.29 | **6 hrs** | Baseline one-stage anchor-free model |
| **Faster R-CNN** | Two-Stage (ResNet-50-FPN) | PyTorch / torchvision | 800 × 800 | *N/A* | 0.254 | 0.365 | 0.209 | 20 hrs | Two-stage anchor-based baseline |

```
mAP@50 Comparison:
YOLO11s      [████████████████████] 0.80  (Winner)
YOLO11n      [███████████████████░] 0.79  (Optimal efficiency)
YOLOv8       [███████████████░░░░░] 0.62  
Faster R-CNN [█████████░░░░░░░░░░░] 0.365 
```

### Key Research Insights
- **Superiority of YOLO11**: YOLO11 models substantially outperformed both YOLOv8 and Faster R-CNN across all key detection metrics. The integration of **C3k2** feature extractors and **SPFF** (Spatial Pyramid Pooling - Fast with Fusion) enables richer multi-scale contextual features that are crucial for distinguishing subtle rubber contours from background clutter.
- **YOLO11s vs. YOLO11n**: YOLO11s delivers the highest overall accuracy (mAP@50 of **0.80** and precision of **0.87**). YOLO11n trails by only **0.01** mAP while training 25% faster (8 hours vs. 10 hours) and requiring significantly less compute, making it the ideal candidate for battery-powered or embedded edge vision systems.
- **Why Faster R-CNN Underperformed**: 
  - Required the longest training duration (**20 hours**).
  - Its Region Proposal Network (RPN) struggled with the elongated, narrow geometries typical of rubber sheets and latex flow cuts.
  - Slower convergence with high false-positive rates on complex wood bark and factory floor textures.

---

## 🗂️ Dataset & Annotation Methodology

The custom rubber dataset was captured across natural plantation environments and factory floors, encompassing diverse ambient lighting, viewing angles, camera distances, and surface textures.

```
                    Raw Images (Factory & Plantation)
                                  │
                                  ▼
                     Annotation via CVAT Tool
                    (Bounding Boxes & Polygons)
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
           YOLO Format (.txt)           COCO Format (.json)
           - [x_center, y_center, w, h]  - Via datasetconverter.py
           - Classes: latex_flow (0),    - Normalized bounding boxes
                      tree_bark (1),     - Formatted for Faster R-CNN
                      rubber_sheet
                   │                             │
                   ▼                             ▼
          Ultralytics YOLO               PyTorch Faster R-CNN
        (YOLOv8, 11n, 11s)               (ResNet-50-FPN)
```

- **Annotation Platform**: [CVAT](https://github.com/cvat-ai/cvat) (Computer Vision Annotation Tool).
- **Target Classes**:
  - `latex_flow` (Class 0): Stream of fresh milky latex sap draining along tree tapping grooves into collection cups.
  - `tree_bark` (Class 1): Intact and harvested trunk bark tissue of the rubber tree.
  - `rubber_sheet`: Manufactured natural rubber sheets undergoing quality evaluation.
- **Dataset Partition**:
  - **Training Set**: ~70% – 80% (augmented with horizontal/vertical flips, rotations, and illumination perturbations)
  - **Validation Set**: ~15% – 20% (held-out validation images)
  - **Test Set**: ~15% (unseen test samples for final evaluation)

---

## 📁 Repository Directory Structure

```text
Rubber Training/
│
├── Deep Learning-Based Rubber Sheet Detection for Industrial Quality Inspection2 .pdf
│   └── Formal academic research paper submitted to IIIT Kottayam.
│
├── Graphs/                                  # Comparative visualization modules
│   ├── map50.py                             # Bar chart script comparing mAP@50
│   ├── map95.py                             # Bar chart script comparing mAP@50-95
│   ├── recall.py                            # Bar chart script comparing recall
│   ├── time.py                              # Bar chart script comparing training durations
│   ├── Map50.png                            # Generated mAP@50 comparative chart
│   ├── Map95.png                            # Generated mAP@50-95 comparative chart
│   ├── Recall.png                           # Generated Recall comparative chart
│   └── Time.png                             # Generated Training Time comparative chart
│
├── Images/                                  # Raw dataset samples & field captures
│   └── [768+ JPEG/PNG images]               # High-res plantation & factory images
│
├── Model Results/                           # YOLO evaluation outputs and ground-truths
│   └── 1/
│       ├── labels/                          # YOLO format test ground truth labels (.txt)
│       └── labels.zip                       # Compressed label backup
│
├── RCNN/                                    # Faster R-CNN pipeline (PyTorch / torchvision)
│   ├── datasetall/                          # Consolidated training data
│   │   ├── data.yaml                        # Class mapping (0: latex_flow, 1: tree_bark)
│   │   ├── images/                          # Split image directories
│   │   ├── labels/                          # YOLO format bounding boxes
│   │   └── train.txt                        # Image index list
│   ├── datasetall.zip                       # Full zipped raw dataset archive
│   ├── rcnndataset/                         # COCO-converted dataset for Faster R-CNN
│   │   ├── train/ (images, annotations.json)
│   │   └── val/   (images, annotations.json)
│   ├── curves/                              # Evaluation plots & PR curves
│   │   ├── all_curves_combined.png          # Overlay of F1, Precision, Recall, PR
│   │   ├── boxF1_curve.png                  # F1-Confidence curve
│   │   ├── boxP_curve.png                   # Precision-Confidence curve
│   │   ├── boxPR_curve.png                  # Precision-Recall curve
│   │   ├── boxR_curve.png                   # Recall-Confidence curve
│   │   └── box_curves_summary.png           # 2x2 grid summary of all curves
│   ├── datasetconverter.py                  # Converts YOLO txt annotations -> COCO JSON
│   ├── eval.py                              # Evaluates Faster R-CNN using torchmetrics
│   ├── single.py                            # Generates combined single graph for R-CNN
│   ├── split.py                             # 80/20 train-validation dataset split script
│   └── faster_rcnn_rubber.pth               # Trained Faster R-CNN weights (ResNet-50)
│
├── yolov8n.pt                               # Base YOLOv8 nano model weights
├── yolov8n-seg.pt                           # Base YOLOv8 segmentation weights
├── .gitignore                               # Git ignore configuration
└── README.md                                # Comprehensive project documentation
```

---

## ⚙️ Installation & Environment Setup

### 1. Prerequisites
- **Python**: Version `3.10` or higher
- **GPU**: NVIDIA CUDA-capable GPU (tested on NVIDIA GeForce RTX 4050 Laptop GPU / CUDA 12.x)

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/rubber-detection-deep-learning.git
cd rubber-detection-deep-learning
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust cuda version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Ultralytics and Computer Vision utilities
pip install ultralytics torchmetrics pycocotools opencv-python matplotlib pillow numpy
```

---

## 🚀 Execution Guide & Workflows

### 1. Train YOLO Models (YOLO11s, YOLO11n, YOLOv8)
To train the top-performing **YOLO11s** model using Ultralytics:
```bash
yolo detect train \
  data="RCNN/datasetall/data.yaml" \
  model=yolo11s.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=rubber_yolo11s
```

To train the lightweight edge **YOLO11n** model:
```bash
yolo detect train \
  data="RCNN/datasetall/data.yaml" \
  model=yolo11n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=rubber_yolo11n
```

### 2. Prepare Data for Faster R-CNN
If starting from raw YOLO labels, run the split and conversion scripts:
```bash
# 1. Split images and labels into train/val subsets (80/20)
python RCNN/split.py

# 2. Convert YOLO bounding boxes into COCO formatted annotations.json
python RCNN/datasetconverter.py
```

### 3. Train Faster R-CNN
Train the Faster R-CNN ResNet-50-FPN model:
```bash
python RCNN/rcnn_model.py
```
*Note: Configured for 20 epochs using AdamW optimizer ($lr=10^{-4}$), batch size 2, and saves checkpoints to `faster_rcnn_rubber.pth`.*

### 4. Evaluate Faster R-CNN & Generate Curves
Compute mAP@50, mAP@50-95, Recall, and export performance curves:
```bash
# Generate individual curves (F1, Precision, Recall, PR, Summary)
python RCNN/eval.py

# Generate single combined overlay graph
python RCNN/single.py
```
Output charts are saved automatically to `RCNN/curves/`.

### 5. Generate Comparative Benchmark Graphs
To regenerate the paper's comparison charts:
```bash
python Graphs/map50.py    # Generates mAP@50 bar chart
python Graphs/map95.py    # Generates mAP@50-95 bar chart
python Graphs/recall.py   # Generates Recall bar chart
python Graphs/time.py     # Generates Training duration chart
```

---

## 📈 Visualizations & Curves

### Faster R-CNN Evaluation Curves
- **Combined Curves**: `RCNN/curves/all_curves_combined.png`
- **F1 vs. Confidence**: `RCNN/curves/boxF1_curve.png`
- **Precision vs. Recall**: `RCNN/curves/boxPR_curve.png`
- **Metric Summary Grid**: `RCNN/curves/box_curves_summary.png`

### Multi-Model Benchmarks
- **mAP@50 Comparison**: `Graphs/Map50.png`
- **mAP@50-95 Comparison**: `Graphs/Map95.png`
- **Recall Comparison**: `Graphs/Recall.png`
- **Training Time**: `Graphs/Time.png`

---

## 🏆 Recommendations for Deployment

1. **For Factory Automation & Quality Inspection**:
   Deploy **YOLO11s**. It achieved the highest precision (0.87) and mAP@50 (0.80), minimizing false accepts of defective sheets while detecting subtle defects under varying factory lighting.
2. **For Field / Mobile Plantation Monitoring**:
   Deploy **YOLO11n**. Its compact parameter footprint enables real-time edge inference on embedded devices (e.g., Raspberry Pi, Jetson Nano, mobile inspection rigs) while maintaining near-peak detection metrics (0.79 mAP@50).
3. **Avoid Heavy Two-Stage Detectors in Low-Sample Regimes**:
   Faster R-CNN proved computationally heavy (20 hrs training) and poorly adapted to fine, irregular rubber tree tapping cuts, highlighting the advantage of anchor-free, multi-scale modern YOLO architectures.

---

## 📜 Citation & Academic Details

If you utilize this dataset, methodology, or code in your research or industrial application, please cite:

```bibtex
@techreport{sanjay2026rubberdetection,
  title       = {Deep Learning-Based Rubber Sheet Detection for Industrial Quality Inspection},
  author      = {Sanjay S},
  institution = {Indian Institute of Information Technology Kottayam (IIIT Kottayam) \& Chinmaya Vishwa Vidyapeeth},
  advisor     = {Dr. Dhakshayani},
  year        = {2026},
  month       = {April--May},
  type        = {Research Internship Report}
}
```

---

## 🤝 Acknowledgments
- **Dr. Dhakshayani** for academic guidance and research mentorship at **Indian Institute of Information Technology Kottayam (IIIT Kottayam)**.
- **Chinmaya Vishwa Vidyapeeth** for institutional support.
- [Ultralytics](https://github.com/ultralytics/ultralytics) and [PyTorch Torchvision](https://pytorch.org/vision/) teams for open-source computer vision frameworks.
