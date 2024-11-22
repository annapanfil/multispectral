import os

from matplotlib import pyplot as plt

from display import draw_litter
import hydra
import cv2

from detection import find_litter, group_contours
from shapes import Rectangle

@hydra.main(config_path="../conf", config_name="net_10", version_base=None)
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
        

        im_detected = draw_litter(image.copy(), pool, blob_contours, bbs, dog_image, mask, f"{cfg.paths.out}/{im_name.split('/')[-1][:-4]}", color=(0,0,255))
        groups_contours, groups_rectangles = group_contours(blob_contours, 70, image[pool.y_b:pool.y_t, pool.x_l:pool.x_r].copy())

        cv2.drawContours(im_detected, groups_contours, -1, (255,0,0), 2)
        cv2.drawContours(im_detected, groups_rectangles, -1, (255,0,0), 2)
        print(groups_rectangles )

        plt.imshow(im_detected)
        plt.show()







if __name__ == "__main__":
    main()
    