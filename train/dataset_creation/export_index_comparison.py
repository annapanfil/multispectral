"""Export aligned images in RGB and BReNir format for visual checking and further processing.

You can run with multiple configs at once with python3 -m train.dataset_creation.export_index_comparison --multirun processing="mandrac2025_5m,..."
The configs should be in the 'conf/processing' folder
"""

import os

from matplotlib import pyplot as plt
from dev.display import draw_yolo_boxes
import hydra
from pathlib import Path
import numpy as np
import cv2

from dev.visualise import get_components_view, get_index_view, save_image, CHANNELS
from src.processing.load import align_from_saved_matrices, align_iterative, find_images, get_altitude, get_panel_irradiance, load_aligned, load_image_set, get_irradiance, save_warp_matrices
from src.processing.evaluate_index import get_custom_index


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    cfg = cfg.processing
    # setup environment
    image_numbers = [f"{x:04}" for x in cfg.params.image_numbers]

    if not os.path.exists(cfg.paths.output):
        print(f"creating directory {cfg.paths.output}")
        os.makedirs(cfg.paths.output)

    # process images
    for i, image_nr in enumerate(image_numbers):
        altitude = get_altitude(cfg, image_nr, i)

        try:
            im_aligned = load_aligned(cfg.paths.images, image_nr)
            img_names = find_images(Path(cfg.paths.images), f"{int(image_nr):04d}")
        except ValueError:
         
            im_aligned = load_aligned(cfg.paths.images, f"{int(image_nr):04d}_{altitude}")
            img_names = find_images(Path(cfg.paths.images), f"{int(image_nr):04d}_{altitude}")  
    
        filename = Path(cfg.paths.output, f"{image_nr}_{altitude}")

        RGB_image = get_components_view(im_aligned, (2,1,0))
      
        RNDWI_image = get_index_view(im_aligned, CHANNELS["N"], CHANNELS["G"])
        
        SPI_image = get_custom_index("((2*N)-E)", im_aligned)

        # draw annotations
        SPI_image = cv2.cvtColor(SPI_image, cv2.COLOR_GRAY2RGB)
        SPI_image = (SPI_image * 255).astype(np.uint8)
        SPI_image, annots = draw_yolo_boxes(img_names[0], 
                        f"{img_names[0].replace('images', 'labels').rsplit('.')[0]}.txt",
                        ["pile"]*50, image=SPI_image, display=False)
        save_image(SPI_image, f"{filename}_form8.jpg", is_float=False)


        RNDWI_image = cv2.cvtColor(RNDWI_image, cv2.COLOR_GRAY2RGB)
        RNDWI_image = (RNDWI_image * 255).astype(np.uint8)
        RNDWI_image, annots = draw_yolo_boxes(img_names[0], 
                        f"{img_names[0].replace('images', 'labels').rsplit('.')[0]}.txt",
                        ["pile"]*50, image=RNDWI_image, display=False)
        save_image(RNDWI_image, f"{filename}_RNDWI.jpg", is_float=False)        

        RGB_image = np.ascontiguousarray(RGB_image)
        RGB_image = (RGB_image * 255).astype(np.uint8)
        RGB_image, annots = draw_yolo_boxes(img_names[0], 
                        f"{img_names[0].replace('images', 'labels').rsplit('.')[0]}.txt",
                        ["pile"]*50, image=RGB_image, display=False)
        save_image(RGB_image, f"{filename}_RGB.jpg", is_float=False)

        plt.subplots(1, 3, figsize=(30, 9))
        plt.subplot(1,3, 1)
        plt.axis('off') 
        plt.imshow(RGB_image)
        plt.subplot(1,3, 2)
        plt.axis('off') 
        plt.imshow(RNDWI_image, cmap="gray")
        plt.subplot(1,3, 3)
        plt.axis('off') 
        plt.imshow(SPI_image, cmap="gray")
        plt.tight_layout()
        plt.savefig(f"{filename}_comparison.jpg")
        plt.close()

if __name__ == "__main__":
    main()