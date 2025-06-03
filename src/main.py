import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO

from src.shapes import Rectangle
from src.utils import get_real_piles_size, greedy_grouping, prepare_image, time_decorator
from src.processing.load import align_from_saved_matrices, get_irradiance, load_image_set
from src.processing.consts import *

def main():
    """Main function for detection and pile positioning. (NOT TESTED YET)""" #TODO: this will be the main file

    img_dir = "/home/anna/Datasets/raw_images/mandrac_2025_04_16/images/0004SET/005" 
    model_path = "../models/sea-form8_sea_aug-random_best.pt"
    warp_matrices_dir = "/home/anna/Datasets/annotated/warp_matrices"
    img_nr = "1049"
    panel_img_nr = "0436"
    altitude = 10
    debug = False
    POOL_HEIGHT = 0.5  # m

    new_image_size = (800, 608)
    formula = "(N - (E - N))"
    channels = ["N", "G", formula]
    is_complex = True if any(len(form) for form in channels) > 40 else False
            
    ##################################################
    start = time.time()

    # Read the channels, align and convert to desired format
    img_capt, panel_capt = time_decorator(load_image_set)(img_dir, img_nr, panel_img_nr, no_panchromatic=True)
    print(img_capt.images[0].raw().shape)

    for i, img in enumerate(img_capt.images):
        print(f"Camera matrix and distortion coeffs {i}:")
        print(img.cv2_camera_matrix())
        print(img.cv2_distortion_coeff())
    img_type = time_decorator(get_irradiance)(img_capt, panel_capt, display=False, vignetting=False)
    img_aligned = time_decorator(align_from_saved_matrices)(img_capt, img_type, warp_matrices_dir, altitude, allow_closest=True, reference_band=0)
    print(img_aligned.shape)
    print(type(img_aligned))
    image = time_decorator(prepare_image)(img_aligned, channels, is_complex, new_image_size)
    print(image.shape)
   
    # Predict
    model = YOLO(model_path)

    results = time_decorator(model.predict)(source=image, conf=0.5, save=False)
    pred_bbs = results[0].boxes.xyxy.cpu().numpy()
    pred_bbs = [Rectangle(*bb, "rect") for bb in pred_bbs]

    # Merge piles
    merged_bbs, merged_img, _ = time_decorator(greedy_grouping)(pred_bbs, image.shape[:2], resize_factor=1.5, visualize=True)

    # TODO: Get position in the world
    sizes = time_decorator(get_real_piles_size)(image.shape[:2], altitude - POOL_HEIGHT, CAM_HFOV, CAM_VFOV, merged_bbs)

    print("----\nWhole main took", time.time() - start, "s")

    # Show
    if debug:
        plt.imshow(merged_img)
        plt.show()

    for rect, size, pos in zip(merged_bbs, sizes):
        rect.draw(image, color=(0, 255, 0), thickness=2)
        text = f"{size[0]*100:.0f}x{size[1]*100:.0f}" # cm
        cv2.putText(image, text, (rect.x_l, rect.y_b), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.imshow(image)
    plt.show()

    #TODO: send positions to rostopic

if __name__ == "__main__":
    main()
