import os
import random
import shutil

# Paths
images_path = "datasetall/images/train"
labels_path = "datasetall/labels/train"

# Output folders
train_img_out = "datasetall/images/train_split"
val_img_out = "datasetall/images/val"

train_lbl_out = "datasetall/labels/train_split"
val_lbl_out = "datasetall/labels/val"

# Create folders
for folder in [train_img_out, val_img_out, train_lbl_out, val_lbl_out]:
    os.makedirs(folder, exist_ok=True)

# Get all image files
images = [f for f in os.listdir(images_path)
          if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]

# Shuffle images
random.shuffle(images)

# 80-20 split
split_index = int(len(images) * 0.8)

train_images = images[:split_index]
val_images = images[split_index:]

def move_files(image_list, img_dest, lbl_dest):
    for img_file in image_list:
        label_file = os.path.splitext(img_file)[0] + ".txt"

        src_img = os.path.join(images_path, img_file)
        src_lbl = os.path.join(labels_path, label_file)

        dst_img = os.path.join(img_dest, img_file)
        dst_lbl = os.path.join(lbl_dest, label_file)

        shutil.copy(src_img, dst_img)

        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)

# Copy train files
move_files(train_images, train_img_out, train_lbl_out)

# Copy validation files
move_files(val_images, val_img_out, val_lbl_out)

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print("Dataset split completed!")