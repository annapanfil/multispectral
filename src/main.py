import cv2
from matplotlib import pyplot as plt
import numpy as np
import hydra

from detection import detect_blobs, find_pool
from display import draw_litter, show_images

def find_litter(base_path: str, im_name: str, out_path:str,  sigma: int, dog_threshold: float, size_max_threshold_perc: float, gamma: float = 2, verb=False) -> None:

    image = cv2.imread(f"{base_path}/{im_name}", cv2.IMREAD_GRAYSCALE)    
    pool = find_pool(image, int(im_name.split("_")[1]), verb=verb)

    if pool is None:
        print(f"Pool not found in {im_name}. Trying raising contrast")
        equalized_image = cv2.equalizeHist(image)
        pool = find_pool(equalized_image, int(im_name.split("_")[1]), verb=verb)
        if pool is None: 
            print("Pool not found")
            return

    cropped_image = image[pool.y_b:pool.y_t, pool.x_l:pool.x_r]
    
    blob_contours, dog_image, mask = detect_blobs(cropped_image, sigma, dog_threshold)

    # show
    im_pool = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(im_pool, (pool.x_l, pool.y_b), (pool.x_r, pool.y_t), (0,0,255), 5)

    im_litter = draw_litter(cropped_image, blob_contours, size_max_threshold_perc * (pool.x_r- pool.x_l), circles=False, rectangles=True, contours=True)

    figure = show_images([im_pool, dog_image, mask, im_litter], ["detected pool", "after DOG", "mask (DOG + threshold)", "litter found"], show=verb)
    figure.savefig(f"{out_path}/{im_name.split('/')[-1][:-4]}_detected.png")

    plt.close()


@hydra.main(config_path="../conf", config_name="net_10", version_base=None)
def main(cfg):
    # print(cfg)
    im_names = [f"{x}_meanRE.png" for x in cfg.paths.im_numbers]
    verb=False

    for im_name in im_names:
        find_litter(cfg.paths.base, im_name, cfg.paths.out, cfg.params.sigma, cfg.params.dog_threshold, cfg.params.size_max_threshold_perc, cfg.params.gamma, verb=verb)


if __name__ == "__main__":
    main()
    


