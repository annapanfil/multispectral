import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO

from detection.shapes import Rectangle
from detection.utils import get_real_piles_size, greedy_grouping, prepare_image, time_decorator
from processing.load import align_from_saved_matrices, get_irradiance, load_image_set, load_aligned

def main():
    """Main function for detection and pile positioning."""
    POOL_HEIGHT = 0.5  # m
    CAM_HFOV = np.deg2rad(49.6)  # rad
    CAM_VFOV = np.deg2rad(38.3)  # rad

    img_dir = "/home/anna/Datasets/pool/realistic_trash/0034SET/000" 
    model_path = "../models/pool-form1_pool-3-channels_random_best.pt"
    warp_matrices_dir = "/home/anna/Datasets/annotated/warp_matrices"
    img_nr = "0004"
    panel_img_nr = "0000"
    altitude = 4 # TODO: get altitude and position from the ros topic or bag file
    debug = True

    new_image_size = (800, 608)
    formula = "E # G"
    channels = [formula, "G", "E"] # wrong order, but thats how the model was trained #TODO: fix this
    is_complex = True if any(len(form) for form in channels) > 40 else False
            
    # Read the channels, align and convert to desired format

    img_capt, panel_capt = time_decorator(load_image_set)(img_dir, img_nr, panel_img_nr)
    img_type = time_decorator(get_irradiance)(img_capt, panel_capt, display=False)
    img_aligned = time_decorator(align_from_saved_matrices)(img_capt, img_type, warp_matrices_dir, altitude, allow_closest=True)
    
    image = time_decorator(prepare_image)(img_aligned, channels, is_complex, new_image_size)
   
    # Predict
    model = YOLO(model_path)

    results = time_decorator(model.predict)(source=image, conf=0.5, save=False)
    pred_bbs = results[0].boxes.xyxy.cpu().numpy()
    pred_bbs = [Rectangle(*bb, "rect") for bb in pred_bbs]

    # Merge piles
    merged_bbs, merged_img = time_decorator(greedy_grouping)(pred_bbs, image.shape[:2], resize_factor=1.5, visualize=True)

    # TODO: Get position in the world
    sizes = time_decorator(get_real_piles_size)(image.shape[:2], altitude - POOL_HEIGHT, CAM_HFOV, CAM_VFOV, merged_bbs)


    # Show
    if debug:
        plt.imshow(merged_img)
        plt.show()

    for rect, size in zip(merged_bbs, sizes):
        rect.draw(image, color=(0, 255, 0), thickness=2)
        text = f"{size[0]*100:.0f}x{size[1]*100:.0f}" # cm
        cv2.putText(image, text, (rect.x_l, rect.y_b), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.imshow(image)
    plt.show()
    
    # TODO: Send somewhere

if __name__ == "__main__":
    main()
