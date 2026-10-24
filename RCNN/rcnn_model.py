import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader

# ============================================
# DATASET PATHS
# ============================================

DATASET_PATH = r"C:\Users\Sanjay S\Videos\Rubber Training\RCNN\rcnndataset"

TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "images")
TRAIN_ANN = os.path.join(DATASET_PATH, "train", "annotations.json")

VAL_IMAGES = os.path.join(DATASET_PATH, "val", "images")
VAL_ANN = os.path.join(DATASET_PATH, "val", "annotations.json")

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
# CUSTOM COCO DATASET
# ============================================

class CustomCocoDataset(CocoDetection):

    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx):

        img, target = super().__getitem__(idx)

        boxes = []
        labels = []

        for obj in target:

            bbox = obj['bbox']

            x, y, w, h = bbox

            # Convert COCO bbox -> Faster RCNN bbox
            boxes.append([
                x,
                y,
                x + w,
                y + h
            ])

            # IMPORTANT FIX
            # DO NOT add +1
            labels.append(obj['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

# ============================================
# LOAD DATASETS
# ============================================

train_dataset = CustomCocoDataset(
    TRAIN_IMAGES,
    TRAIN_ANN,
    transforms=transform
)

val_dataset = CustomCocoDataset(
    VAL_IMAGES,
    VAL_ANN,
    transforms=transform
)

# ============================================
# COLLATE FUNCTION
# ============================================

def collate_fn(batch):
    return tuple(zip(*batch))

# ============================================
# DATALOADERS
# ============================================

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)

# ============================================
# LOAD FASTER RCNN MODEL
# ============================================

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

# ============================================
# NUMBER OF CLASSES
#
# background + latex_flow + tree_bark
# ============================================

num_classes = 3

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    num_classes
)

model.to(device)

# ============================================
# OPTIMIZER
# ============================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001
)

# ============================================
# TRAINING LOOP
# ============================================

num_epochs = 20

for epoch in range(num_epochs):

    model.train()

    epoch_loss = 0

    for images, targets in train_loader:

        images = [img.to(device) for img in images]

        targets = [
            {
                k: v.to(device)
                for k, v in t.items()
            }
            for t in targets
        ]

        # Forward pass
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        epoch_loss += losses.item()

    avg_loss = epoch_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

# ============================================
# SAVE MODEL
# ============================================

torch.save(model.state_dict(), "faster_rcnn_rubber.pth")

print("\nTraining completed!")
print("Model saved as faster_rcnn_rubber.pth")