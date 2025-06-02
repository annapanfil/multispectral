import random
import albumentations as A
import cv2
import numpy as np

from processing.load import load_aligned

# --- Augmentation Pipeline ---
transforms = A.Compose([
    # --- Environmental ---
    A.RandomShadow(p=0.3),  
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

    # --- Geometric ---
    A.ShiftScaleRotate(
        shift_limit=0.1,  # Small translations (10% of image size)
        scale_limit=0.05,   # +/- 10% scaling
        rotate_limit=30,   # Degrees
        border_mode=cv2.BORDER_REFLECT,  # Avoid edge artifacts
        p=0.5
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ElasticTransform(
        alpha=0.8, sigma=20, p=0.3  # Mild wave distortion
    ),
    A.RandomSizedBBoxSafeCrop(  # AtLeastOneBBox equivalent
        height=608, width=800, erosion_rate=0.2, p=0.5
    ),
    
    # --- Blur & Noise ---
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.MedianBlur(blur_limit=3, p=0.2),
        A.MotionBlur(blur_limit=(3, 5), p=0.4)  # Drone movement
    ], p=0.5),
    A.OneOf([
        A.MultiplicativeNoise(
            multiplier=(0.9, 1.1), per_channel=True, p=0.5  # Per-band noise
        ),
        A.GaussNoise(var_limit=(10, 30), p=0.5)
    ], p=0.5),
    
    # --- Occlusion ---
    A.CoarseDropout(
        max_holes=3, max_height=0.1, max_width=0.1,
        fill_value=0, p=0.3  # water is mostly close to black
    )
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

from pathlib import Path
import os

def pre_augment_dataset(base_dir, n_augment=2):
    """Generate augmented images/labels and save to train_aug."""

    os.makedirs(f"{base_dir}images/train_aug/", exist_ok=True)
    os.makedirs(f"{base_dir}labels/train_aug/", exist_ok=True)
    
    imgs_dir = f"{base_dir}/images/train/"
    for im_name in os.listdir(imgs_dir):
        if not im_name.endswith("ch0.tiff"):
            continue
        print(f"Processing {im_name}")
        im_name = "_".join(im_name.split(".")[0].split("_")[:-1])  # Remove channel number
        label_path = f"{base_dir}/labels/train/{im_name}_ch0.txt"
        image = load_aligned(imgs_dir, im_name).astype(np.uint8)
        image = cv2.resize(image, (800, 608))

        with open(label_path, 'r') as f:
            lines = [list(map(float, line.split())) for line in f.readlines()]
            bboxes = [line[1:] for line in lines]  # Only bbox coordinates
            class_labels = [int(line[0]) for line in lines]  # Class IDs
        
        # Apply augmentations `n_augment` times
        for i in range(n_augment):
            transformed = transforms(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            # Save augmented channels
            for ch in range(image.shape[2]):
                aug_img_path = f"{base_dir}/images/train_aug/{im_name}_aug{i}_ch{ch}.tiff"
                cv2.imwrite(aug_img_path, transformed["image"][:, :, ch])
                aug_label_path = f"{base_dir}/labels/train_aug/{im_name}_aug{i}_ch{ch}.txt"
                with open(aug_label_path, 'w') as f:
                    for class_id, bbox in zip(class_labels, transformed["bboxes"]):
                        f.write(f"{int(class_id)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
    
if __name__ == "__main__":
    random.seed(42)
    pre_augment_dataset(
        base_dir="/home/anna/Datasets/annotated/mandrac3/",
        n_augment=2  # Generate 2 augmented versions per original image
    )