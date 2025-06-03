"""Export aligned channels, RGB and meanRE images for anotation from a config

You can run with multiple configs at once with python3 -m dataset_creation.export_aligned_channels --multirun processing="mandrac2025_5m,..."
The configs should be in the 'conf/processing' folder
"""

import os
import hydra
from pathlib import Path

from src.processing.load import align_from_saved_matrices, get_altitude, load_image_set, get_irradiance
from dev.visualise import get_components_view, save_image, CHANNELS
from src.processing.evaluate_index import get_custom_index
import micasense.imageutils as imageutils

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    cfg = cfg.processing
    # setup environment
    image_numbers = [f"{x:04}" for x in cfg.params.image_numbers]

    # if not os.path.exists(cfg.paths.channels_output):
    #     print(f"creating directory {cfg.paths.channels_output}")
    #     os.makedirs(cfg.paths.channels_output)

    # process images
    for i, image_nr in enumerate(image_numbers):

        img_capt, panel_capt = load_image_set(
            cfg.paths.images, image_nr, cfg.paths.panel_image_nr
        )

        print(f"Aligning {len(img_capt.images)} images...")

        # align the image
        altitude = get_altitude(cfg, image_nr, i)
        img_type = get_irradiance(img_capt, panel_capt, display=False)

        im_aligned = align_from_saved_matrices(img_capt, img_type, cfg.paths.warp_matrices, altitude, allow_closest=True, reference_band=0)

        # save channels and indexes
        path = Path(cfg.paths.channels_output, f"{image_nr}_{altitude}")

        if not os.path.exists(path):
            print(f"creating directory {path}")
            os.makedirs(path)

        for i in range(0, im_aligned.shape[2]):
            save_image(imageutils.normalize(im_aligned[:,:,i]), f"{path}/{image_nr}_{altitude}_ch{i}.tif")

        RGB_image = get_components_view(im_aligned, (2,1,0))
        save_image(RGB_image, f"{path}/{image_nr}_{altitude}_RGB.png", bgr=True)

        other_channels_image = get_components_view(im_aligned, (2,3,4))
        save_image(other_channels_image, f"{path}/{image_nr}_{altitude}_234.png")

        meanRE_image = get_custom_index("0.5 * (E-G)/(E+G) + 0.5 * (E-B)/(E+B)", im_aligned)
        save_image(meanRE_image, f"{path}/{image_nr}_{altitude}_meanRE.png")
        

if __name__ == "__main__":
    main()