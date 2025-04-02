"""Main file for detection and pile positioning."""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from ultralytics import YOLO

from detection.shapes import Rectangle
from detection.utils import get_real_piles_size, greedy_grouping

def main():
    """Main function for detection and pile positioning."""
    POOL_HEIGHT = 0.5  # m
    CAM_HFOV = np.deg2rad(49.6)  # rad
    CAM_VFOV = np.deg2rad(38.3)  # rad

    im_dir = "/home/anna/Datasets/created/pool-form1_pool-3-channels_random/images/val"
    model_path = "../models/pool-form1_pool-3-channels_random_best.pt"
    debug = False

    for im_name in os.listdir(im_dir):
        # Read the image
        # TODO: Read the raw channels, align them and convert to desired format
        image = cv2.imread(f"{im_dir}/{im_name}")
        altitude = int(im_name.split("_")[3])  # Extract the altitude from the image name

        # Predict from yolo
        model = YOLO(model_path)  # load a pretrained model

        results = model.predict(source=image, conf=0.3, save=False)

        pred_bbs = results[0].boxes.xyxy.cpu().numpy()
        pred_bbs = [Rectangle(*bb, "rect") for bb in pred_bbs]

        # Merge piles
        merged_bbs, merged_img = greedy_grouping(pred_bbs, image.shape[:2], resize_factor=1.5, visualize=True)

        # TODO: Get position in the world
        sizes = get_real_piles_size(image.shape[:2], altitude - POOL_HEIGHT, CAM_HFOV, CAM_VFOV, merged_bbs)

        # Show
        if debug:
            plt.imshow(merged_img)
            plt.show()

        for rect, size in zip(pred_bbs, sizes):
            rect.draw(image, color=(0, 255, 0), thickness=2)
            text = f"{size[0]*100:.0f}x{size[1]*100:.0f}" # cm
            cv2.putText(image, text, (rect.x_l, rect.y_b), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        plt.imshow(image)
        plt.show()
        
        # TODO: Send somewhere

if __name__ == "__main__":
    main()
