import os
import hydra
from pathlib import Path

from load import align_from_saved_matrices, get_altitude, load_image_set, get_irradiance
from visualise import get_components_view, get_custom_index, save_image, CHANNELS
import micasense.imageutils as imageutils

@hydra.main(config_path="../conf/processing", config_name="config_net_10", version_base=None)
def main(cfg):
    # setup environment
    image_numbers = [f"{x:04}" for x in cfg.params.image_numbers]

    if not os.path.exists(cfg.paths.channels_output):
        print(f"creating directory {cfg.paths.channels_output}")
        os.makedirs(cfg.paths.channels_output)

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

        # save channels and indexes
        path = Path(cfg.paths.channels_output, f"{image_nr}_{altitude}")

        if not os.path.exists(path):
            print(f"creating directory {path}")
            os.makedirs(path)

        for i in range(0, im_aligned.shape[2]):
            save_image(imageutils.normalize(im_aligned[:,:,i]), f"{path}/{image_nr}_{altitude}_ch{i}.tif")

        RGB_image = get_components_view(im_aligned, (2,1,0))
        save_image(RGB_image, f"{path}/{image_nr}_{altitude}_RGB.png", bgr=True)

        meanRE_image = get_custom_index("0.5 * (RE-G)/(RE+G) + 0.5 * (RE-B)/(RE+B)", im_aligned)
        save_image(meanRE_image, f"{path}/{image_nr}_{altitude}_meanRE.png")
        

if __name__ == "__main__":
    main()