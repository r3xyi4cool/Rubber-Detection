import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================
# PATHS
# ============================================
DATASET_PATH = r"C:\Users\Sanjay S\Videos\Rubber Training\RCNN\rcnndataset"
MODEL_PATH   = r"faster_rcnn_rubber.pth"
VAL_IMAGES   = os.path.join(DATASET_PATH, "val", "images")
VAL_ANN      = os.path.join(DATASET_PATH, "val", "annotations.json")
CURVES_DIR   = Path("curves")          # output folder for PNGs
CURVES_DIR.mkdir(exist_ok=True)

# ============================================
# DEVICE
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================
# TRANSFORMS
# ============================================
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True)
])

# ============================================
# DATASET
# ============================================
class CustomCocoDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        boxes, labels = [], []
        for obj in target:
            x, y, w, h = obj["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(obj["category_id"])

        boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)
        return img, target

# ============================================
# DATALOADER
# ============================================
val_dataset = CustomCocoDataset(VAL_IMAGES, VAL_ANN, transforms=transform)

def collate_fn(batch):
    return tuple(zip(*batch))

val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
)

# ============================================
# MODEL
# ============================================
model = fasterrcnn_resnet50_fpn(weights=None)
num_classes = 3
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = (
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

# ============================================
# COLLECT RAW PREDICTIONS + GROUND TRUTHS
# ============================================
all_scores, all_pred_labels = [], []
all_gt_labels = []
iou_thresholds = np.linspace(0.5, 0.95, 10)

# Per-image storage for curve computation
all_preds, all_gts = [], []

metric = MeanAveragePrecision()

with torch.no_grad():
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        preds_batch, gts_batch = [], []
        for output, target in zip(outputs, targets):
            p = {
                "boxes":  output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu(),
            }
            g = {
                "boxes":  target["boxes"],
                "labels": target["labels"],
            }
            preds_batch.append(p)
            gts_batch.append(g)
            all_preds.append(p)
            all_gts.append(g)

        metric.update(preds_batch, gts_batch)

results = metric.compute()
print("\n===== Faster R-CNN Evaluation =====")
print(f"mAP50    : {results['map_50']:.4f}")
print(f"mAP50-95 : {results['map']:.4f}")
print(f"Recall   : {results['mar_100']:.4f}")

# ============================================
# HELPER – IoU between two boxes (xyxy)
# ============================================
def box_iou_single(b1, b2):
    """b1, b2 : (4,) tensors [x1,y1,x2,y2]"""
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

# ============================================
# BUILD MATCHED DETECTIONS AT IoU 0.5
# ============================================
def build_detections(preds, gts, iou_thresh=0.5):
    """
    Returns arrays of (score, tp_flag) sorted by score descending,
    and total number of ground-truth boxes.
    """
    det_scores, det_tp = [], []
    n_gt = 0

    for p, g in zip(preds, gts):
        boxes_p = p["boxes"];  scores_p = p["scores"];  labels_p = p["labels"]
        boxes_g = g["boxes"];  labels_g = g["labels"]
        n_gt += len(boxes_g)

        matched_gt = set()
        # sort predictions by descending score
        order = scores_p.argsort(descending=True)
        for idx in order:
            score = scores_p[idx].item()
            pb    = boxes_p[idx]
            pl    = labels_p[idx].item()
            best_iou, best_j = 0.0, -1
            for j, (gb, gl) in enumerate(zip(boxes_g, labels_g)):
                if gl.item() != pl:
                    continue
                iou = box_iou_single(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j not in matched_gt:
                det_scores.append(score)
                det_tp.append(1)
                matched_gt.add(best_j)
            else:
                det_scores.append(score)
                det_tp.append(0)

    det_scores = np.array(det_scores)
    det_tp     = np.array(det_tp)
    order      = np.argsort(-det_scores)
    return det_scores[order], det_tp[order], n_gt

# ============================================
# PRECISION / RECALL / F1 CURVES
# ============================================
scores_sorted, tp_sorted, n_gt = build_detections(all_preds, all_gts)

cum_tp = np.cumsum(tp_sorted)
cum_fp = np.cumsum(1 - tp_sorted)
n_det  = np.arange(1, len(tp_sorted) + 1)

precision_curve = cum_tp / n_det                        # P at each threshold
recall_curve    = cum_tp / (n_gt + 1e-9)                # R at each threshold
f1_curve        = (2 * precision_curve * recall_curve /
                   (precision_curve + recall_curve + 1e-9))

# Confidence thresholds (use the actual scores as thresholds)
conf_curve = scores_sorted

# ============================================
# PLOT STYLE HELPERS
# ============================================
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.grid":        True,
    "grid.color":       "#dee2e6",
    "grid.linewidth":   0.7,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
}

def styled_fig(title, xlabel, ylabel, filename):
    """Create a figure, save and close it."""
    return title, xlabel, ylabel, CURVES_DIR / filename

def save_curve(x, y, title, xlabel, ylabel, filename,
               color="#0077b6", label=None, extra_lines=None):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, y, color=color, linewidth=2,
                label=label or title)
        if extra_lines:
            for (ex, ey, ec, el) in extra_lines:
                ax.plot(ex, ey, color=ec, linewidth=1.5,
                        linestyle="--", label=el)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        if label or extra_lines:
            ax.legend(fontsize=10)
        out = CURVES_DIR / filename
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")

# ============================================
# 1.  box_F1_curve  –  F1 vs Confidence
# ============================================
save_curve(
    conf_curve, f1_curve,
    title    = "Box F1-Confidence Curve",
    xlabel   = "Confidence Threshold",
    ylabel   = "F1 Score",
    filename = "boxF1_curve.png",
    color    = "#0077b6",
    label    = f"all classes  (best F1={f1_curve.max():.2f})"
)

# ============================================
# 2.  box_P_curve  –  Precision vs Confidence
# ============================================
save_curve(
    conf_curve, precision_curve,
    title    = "Box Precision-Confidence Curve",
    xlabel   = "Confidence Threshold",
    ylabel   = "Precision",
    filename = "boxP_curve.png",
    color    = "#e63946",
    label    = "Precision"
)

# ============================================
# 3.  box_PR_curve  –  Precision vs Recall
# ============================================
save_curve(
    recall_curve, precision_curve,
    title    = "Box Precision-Recall Curve",
    xlabel   = "Recall",
    ylabel   = "Precision",
    filename = "boxPR_curve.png",
    color    = "#2a9d8f",
    label    = f"mAP@0.5 = {results['map_50']:.4f}"
)

# ============================================
# 4.  box_R_curve  –  Recall vs Confidence
# ============================================
save_curve(
    conf_curve, recall_curve,
    title    = "Box Recall-Confidence Curve",
    xlabel   = "Confidence Threshold",
    ylabel   = "Recall",
    filename = "boxR_curve.png",
    color    = "#f4a261",
    label    = "Recall"
)

# ============================================
# 5.  BONUS – all four on one summary grid
# ============================================
with plt.rc_context(STYLE):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Faster R-CNN  –  Bounding-Box Evaluation Curves",
                 fontsize=15, fontweight="bold", y=1.01)

    configs = [
        (axes[0,0], conf_curve,   f1_curve,        "#0077b6",
         "Box F1-Confidence",     "Confidence", "F1"),
        (axes[0,1], conf_curve,   precision_curve,  "#e63946",
         "Box Precision-Confidence","Confidence", "Precision"),
        (axes[1,0], recall_curve, precision_curve,  "#2a9d8f",
         "Box Precision-Recall",  "Recall",      "Precision"),
        (axes[1,1], conf_curve,   recall_curve,     "#f4a261",
         "Box Recall-Confidence", "Confidence",  "Recall"),
    ]

    for ax, x, y, col, title, xl, yl in configs:
        ax.plot(x, y, color=col, linewidth=2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

    out = CURVES_DIR / "box_curves_summary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

print("\nAll curve plots saved to:", CURVES_DIR.resolve())