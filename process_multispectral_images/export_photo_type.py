#!/home/anna/miniconda3/envs/micasense/bin/python3

import time
import exiftool

from load import load_image_set, get_irradiance, align
from visualise import get_components_view, get_index_view, get_PI_image, show_image, save_image, plot_all_channels, plot_one_channel, CHANNEL_NAMES

if __name__ == "__main__":
    image_dir = "/home/anna/Pictures/pool/bags/0036SET/000"

    image_numbers = [f"{x:04}" for x in (4,6,10,12,18,19)]
    panel_image_nr = "0001"

    for image_nr in image_numbers:

        img_capt, panel_capt = load_image_set(
            image_dir, image_nr, panel_image_nr
        )

        print(f"Loaded {len(img_capt.images)} images.\nAligning images...")
        with exiftool.ExifToolHelper() as et:
            altitude = et.get_tags(f"{image_dir}/IMG_{image_nr}_1.tif", ["Composite:GPSAltitude"])[0]["Composite:GPSAltitude"]

        img_type, irradiance_list = get_irradiance(img_capt, panel_capt, display=False)

        start = time.time()
        sharpened_stack, im_aligned = align(img_capt, img_type, irradiance_list, matrices_fn=f"./out/warp_matrices_{altitude}.npy")
        duration = time.time() - start
        print(f"Alignment took {duration:.2f} s.")

        filename = f"./out/9:00_sunny_panel/{image_nr}_{altitude-50}"

        RGB_image = get_components_view(im_aligned, (2,1,0))
        save_image(RGB_image, f"{filename}_RGB.png", bgr=True)

        NDWI_image = get_index_view(im_aligned, CHANNEL_NAMES.index("NIR"), CHANNEL_NAMES.index("G"))
        save_image(NDWI_image, f"{filename}_NDWI.png")

        NDVI_image = get_index_view(im_aligned, CHANNEL_NAMES.index("NIR"), CHANNEL_NAMES.index("R"))
        save_image(NDVI_image, f"{filename}_NDVI.png")

        PI = get_PI_image(im_aligned)
        save_image(PI, f"{filename}_PI.png")


