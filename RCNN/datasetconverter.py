import os
import json
import shutil
from PIL import Image

# =========================
# CONFIG
# =========================

dataset_path = r"C:\Users\Sanjay S\Videos\Rubber Training\RCNN\datasetall"

output_dataset = r"C:\Users\Sanjay S\Videos\Rubber Training\RCNN\rcnndataset"

splits = ["train", "val"]  # change to ["train", "val"] if needed

class_names = ["rubber"]

# =========================

# Create output folder
os.makedirs(output_dataset, exist_ok=True)

for split in splits:

    images_path = os.path.join(dataset_path, "images", split)
    labels_path = os.path.join(dataset_path, "labels", split)

    output_split = os.path.join(output_dataset, split)
    output_images = os.path.join(output_split, "images")

    os.makedirs(output_images, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Categories
    for idx, name in enumerate(class_names):
        coco["categories"].append({
            "id": idx,
            "name": name
        })

    annotation_id = 0
    image_id = 0

    for image_file in os.listdir(images_path):

        if not image_file.endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(images_path, image_file)

        # Copy image to new dataset
        shutil.copy(image_path, output_images)

        img = Image.open(image_path)
        width, height = img.size

        coco["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })

        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_path, label_file)

        if os.path.exists(label_path):

            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                class_id = int(parts[0])

                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                bbox_width = float(parts[3]) * width
                bbox_height = float(parts[4]) * height

                x_min = x_center - (bbox_width / 2)
                y_min = y_center - (bbox_height / 2)

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [
                        x_min,
                        y_min,
                        bbox_width,
                        bbox_height
                    ],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })

                annotation_id += 1

        image_id += 1

    annotation_output = os.path.join(output_split, "annotations.json")

    with open(annotation_output, "w") as f:
        json.dump(coco, f)

    print(f"{split} conversion completed!")

print("\nRCNN dataset created successfully!")