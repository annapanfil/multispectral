import os

from matplotlib import pyplot as plt
import numpy as np

from display import draw_litter
import hydra
import cv2

from detection import find_litter, get_real_piles_size, group_contours, pool2abs_point
from shapes import Rectangle

POOL_HEIGHT = 1.5 #m

@hydra.main(config_path="../conf", config_name="testing", version_base=None)
def main(cfg):
    # print(cfg)
    im_names = [f"{x}_meanRE.png" for x in cfg.paths.im_numbers]
    verb=False

    if not os.path.exists(cfg.paths.out):
        os.makedirs(cfg.paths.out)
        print(f"Created directory {cfg.paths.out}")

    for im_name in im_names:
        image = cv2.imread(f"{cfg.paths.base}/{im_name}", cv2.IMREAD_GRAYSCALE)    
        altitude = int(im_name.split("_")[-2])  # Extract the altitude from the image name

        blob_contours, bbs, pool, dog_image, mask = find_litter(image.copy(), im_name, 
                    sigma=cfg.params.sigma_a * altitude + cfg.params.sigma_b,
                    # sigma=cfg.params.sigma,
                    dog_threshold=cfg.params.dog_threshold, 
                    size_max_threshold_perc=cfg.params.size_max_threshold_perc, 
                    verb=verb)
        
        piles_contours, piles_rectangles, piles_rectangles_for_drawing = group_contours(blob_contours, 70, image[pool.y_b:pool.y_t, pool.x_l:pool.x_r].copy())

        piles_rectangles_abs = [[pool2abs_point(center, pool), dims, angle] for center, dims, angle in piles_rectangles]
        
        sizes = get_real_piles_size(image.shape[:2], altitude-POOL_HEIGHT, np.deg2rad(49.6), np.deg2rad(38.3), piles_rectangles_abs)



        # show
        im_detected = draw_litter(image.copy(), pool, blob_contours, bbs, dog_image, mask, f"{cfg.paths.out}/{im_name.split('/')[-1][:-4]}", color=(0,0,255))
        cv2.drawContours(im_detected, piles_contours, -1, (255,0,0), 2)
        cv2.drawContours(im_detected, piles_rectangles_for_drawing, -1, (255,0,0), 2)

        # add text
        for size, box, rect in zip(sizes, piles_rectangles_for_drawing, piles_rectangles):
            text_mask = np.zeros_like(im_detected)
            position = (int(box[0][0]), int(box[0][1]) - 5)

            cv2.putText(text_mask, f"{size[0]*100:.0f} x {size[1]*100:.0f}cm", position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            rotation_matrix = cv2.getRotationMatrix2D(position, 90-rect[2], 1)
            rotated_text = cv2.warpAffine(text_mask, rotation_matrix, (im_detected.shape[1], im_detected.shape[0]))
            im_detected = cv2.addWeighted(im_detected, 1, rotated_text, 1, 0)


        plt.imshow(im_detected)
        plt.show()


if __name__ == "__main__":
    main()
    