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
CURVES_DIR   = Path("curves")
CURVES_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True)
])

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

val_dataset = CustomCocoDataset(VAL_IMAGES, VAL_ANN, transforms=transform)

def collate_fn(batch):
    return tuple(zip(*batch))

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

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
print(f"\nmAP50    : {results['map_50']:.4f}")
print(f"mAP50-95 : {results['map']:.4f}")
print(f"Recall   : {results['mar_100']:.4f}")

def box_iou_single(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def build_detections(preds, gts, iou_thresh=0.5):
    det_scores, det_tp = [], []
    n_gt = 0
    for p, g in zip(preds, gts):
        boxes_p = p["boxes"]; scores_p = p["scores"]; labels_p = p["labels"]
        boxes_g = g["boxes"]; labels_g = g["labels"]
        n_gt += len(boxes_g)
        matched_gt = set()
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
                det_scores.append(score); det_tp.append(1)
                matched_gt.add(best_j)
            else:
                det_scores.append(score); det_tp.append(0)
    det_scores = np.array(det_scores)
    det_tp     = np.array(det_tp)
    order      = np.argsort(-det_scores)
    return det_scores[order], det_tp[order], n_gt

scores_sorted, tp_sorted, n_gt = build_detections(all_preds, all_gts)

cum_tp = np.cumsum(tp_sorted)
cum_fp = np.cumsum(1 - tp_sorted)
n_det  = np.arange(1, len(tp_sorted) + 1)

precision_curve = cum_tp / n_det
recall_curve    = cum_tp / (n_gt + 1e-9)
f1_curve        = (2 * precision_curve * recall_curve /
                   (precision_curve + recall_curve + 1e-9))
conf_curve      = scores_sorted

# ============================================
# SINGLE COMBINED GRAPH – all curves together
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

with plt.rc_context(STYLE):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ── Confidence-based curves (x = confidence threshold) ──
    l1, = ax1.plot(conf_curve, f1_curve,
                   color="#0077b6", linewidth=2,
                   label=f"F1  (best={f1_curve.max():.2f})")
    l2, = ax1.plot(conf_curve, precision_curve,
                   color="#e63946", linewidth=2,
                   label="Precision")
    l3, = ax1.plot(conf_curve, recall_curve,
                   color="#f4a261", linewidth=2,
                   label="Recall")

    ax1.set_xlabel("Confidence Threshold  /  Recall  (PR curve)",
                   fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])

    # ── PR curve (x = recall, y = precision) overlaid ──
    # Since recall values are also in [0,1] they share the same x-axis range,
    # but we mark them with a distinct style so the viewer can distinguish.
    l4, = ax1.plot(recall_curve, precision_curve,
                   color="#2a9d8f", linewidth=2, linestyle="--",
                   label=f"PR curve  (mAP@0.5={results['map_50']:.4f})")

    # ── Legend & title ──
    ax1.legend(handles=[l1, l2, l3, l4],
               loc="lower left", fontsize=10,
               framealpha=0.9)

    ax1.set_title("Faster R-CNN – All Evaluation Curves (Combined)",
                  fontsize=14, fontweight="bold", pad=12)

    # Annotation: best F1 point
    best_idx = np.argmax(f1_curve)
    ax1.annotate(
        f"Best F1 @ conf={conf_curve[best_idx]:.2f}",
        xy=(conf_curve[best_idx], f1_curve[best_idx]),
        xytext=(conf_curve[best_idx] - 0.25, f1_curve[best_idx] + 0.08),
        arrowprops=dict(arrowstyle="->", color="#0077b6"),
        fontsize=9, color="#0077b6"
    )

    out = CURVES_DIR / "all_curves_combined.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved combined graph → {out}")