import os
import time
import exiftool
import hydra
from pathlib import Path
from omegaconf import ListConfig
import numpy as np

from load import align_from_saved_matrices, load_image_set, get_irradiance
from visualise import get_components_view, get_custom_index, get_index_view, save_image, CHANNELS

def threshold_percentiles(image):
    image[image > np.percentile(image, 99.99)] = np.percentile(image, 99.99)
    image[image < np.percentile(image, 0.01)] = np.percentile(image, 0.01)

    return image


@hydra.main(config_path="conf", config_name="config_net_10", version_base=None)
def main(cfg):
    # setup environment
    image_numbers = [f"{x:04}" for x in cfg.params.image_numbers]

    if not os.path.exists(cfg.paths.output):
        print(f"creating directory {cfg.paths.output}")
        os.makedirs(cfg.paths.output)

    # process images
    for i, image_nr in enumerate(image_numbers):

        img_capt, panel_capt = load_image_set(
            cfg.paths.images, image_nr, cfg.paths.panel_image_nr
        )

        print(f"Aligning {len(img_capt.images)} images...")

        # get altitude
        with exiftool.ExifToolHelper() as et:
            altitude = et.get_tags(Path(cfg.paths.images, f"IMG_{image_nr}_1.tif"), ["Composite:GPSAltitude"])[0]["Composite:GPSAltitude"]

        if  isinstance(cfg.params.altitude_change, ListConfig): 
            altitude = int(altitude - cfg.params.altitude_change[i])
        else:
            altitude = int(altitude - cfg.params.altitude_change)
        
        # align the image
        img_type, irradiance_list = get_irradiance(img_capt, panel_capt, display=False)
        im_aligned = align_from_saved_matrices(img_capt, img_type, cfg.paths.warp_matrices, altitude, allow_closest=True)

        # save views
        filename = Path(cfg.paths.output, f"{image_nr}_{altitude}")

        NDWI_image = get_index_view(im_aligned, CHANNELS["NIR"], CHANNELS["G"], normalize=False)
        NDWI_image = threshold_percentiles(NDWI_image)
        save_image(NDWI_image, f"{filename}_RNDWI_thresh.png")

        mulRE_image = get_custom_index("1-((RE-G)/(RE+G) * (RE-B)/(RE+B))", im_aligned, normalize=False)
        mulRE_image = threshold_percentiles(mulRE_image)
        save_image(mulRE_image, f"{filename}_mulRE_thresh.png")

        meanRE_image = get_custom_index("0.5 * (RE-G)/(RE+G) + 0.5 * (RE-B)/(RE+B)", im_aligned, normalize=False)
        meanRE_image = threshold_percentiles(meanRE_image)
        save_image(meanRE_image, f"{filename}_meanRE_thresh.png")


        RGB_image = get_components_view(im_aligned, (2,1,0))
        save_image(RGB_image, f"{filename}_RGB.png", bgr=True)

        NDWI_image = get_index_view(im_aligned, CHANNELS["NIR"], CHANNELS["G"])
        save_image(NDWI_image, f"{filename}_RNDWI_norm.png")

        mulRE_image = get_custom_index("1-((RE-G)/(RE+G) * (RE-B)/(RE+B))", im_aligned)
        save_image(mulRE_image, f"{filename}_mulRE_norm.png")

        meanRE_image = get_custom_index("0.5 * (RE-G)/(RE+G) + 0.5 * (RE-B)/(RE+B)", im_aligned)
        save_image(meanRE_image, f"{filename}_meanRE_norm.png")



        # CIR_image = get_components_view(im_aligned, (3,2,1))
        # save_image(CIR_image, f"{filename}_CIR.png", bgr=True)

if __name__ == "__main__":
    main()