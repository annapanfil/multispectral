import os
import hydra
from pathlib import Path
import numpy as np

from processing.load import align_from_saved_matrices, get_altitude, load_image_set, get_irradiance
from processing.visualise import get_components_view, save_image, CHANNELS
from processing.evaluate_index import get_custom_index

def threshold_percentiles(image):
    image[image > np.percentile(image, 99.99)] = np.percentile(image, 99.99)
    image[image < np.percentile(image, 0.01)] = np.percentile(image, 0.01)

    return image

# you can run with multiple configs at once with python3 -m dataset_creation.export_photo_type --multirun processing="mandrac2025_5m,..."
# the configs should be in the 'conf/processing' folder
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

        img_capt, panel_capt = load_image_set(
            cfg.paths.images, image_nr, cfg.paths.panel_image_nr
        )

        print(f"Aligning {len(img_capt.images)} images...")

       
        # align the image
        altitude = get_altitude(cfg, image_nr, i)
        img_type = get_irradiance(img_capt, panel_capt, display=False)

        im_aligned = align_from_saved_matrices(img_capt, img_type, cfg.paths.warp_matrices, altitude, allow_closest=True)

        # save views
        filename = Path(cfg.paths.output, f"{image_nr}_{altitude}")

        RGB_image = get_components_view(im_aligned, (2,1,0))
        save_image(RGB_image, f"{filename}_RGB.png", bgr=True)

        other_channels_image = get_components_view(im_aligned, (2,3,4))
        save_image(other_channels_image, f"{filename}_234.png")

        # NDWI_image = get_index_view(im_aligned, CHANNELS["NIR"], CHANNELS["G"])
        # save_image(NDWI_image, f"{filename}_RNDWI.png")

        # mulRE_image = get_custom_index("1-((RE-G)/(RE+G) * (RE-B)/(RE+B))", im_aligned)
        # save_image(mulRE_image, f"{filename}_mulRE.png")

        # meanRE_image = get_custom_index("0.5 * (RE-G)/(RE+G) + 0.5 * (RE-B)/(RE+B)", im_aligned)
        # save_image(meanRE_image, f"{filename}_meanRE.png")

        # CIR_image = get_components_view(im_aligned, (3,2,1))
        # save_image(CIR_image, f"{filename}_CIR.png", bgr=True)

        

if __name__ == "__main__":
    main()