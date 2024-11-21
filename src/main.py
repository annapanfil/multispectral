import os
from display import draw_litter
import hydra
import cv2

from detection import find_litter

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
    
        draw_litter(image.copy(), pool, blob_contours, bbs, dog_image, mask, f"{cfg.paths.out}/{im_name.split('/')[-1][:-4]}")



if __name__ == "__main__":
    main()
    


