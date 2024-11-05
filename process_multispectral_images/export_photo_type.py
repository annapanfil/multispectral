import time
import exiftool

from load import align_from_saved_matrices, load_image_set, get_irradiance, align_SIFT, align_rig_relatives, align_iterative, save_warp_matrices
from visualise import get_components_view, get_index_view, get_PI_image, show_image, save_image, plot_all_channels, plot_one_channel, CHANNEL_NAMES

if __name__ == "__main__":
    
    # hour set altitude_change images
    # 9:00 36 50 4,6,10,12,18,19
    # 12:00 41 35 1, 2, 3, 5, 8, 9
    # 15:00 35 45 4, 7, 9, 14, 17, 19

    # alignment 45 59 4,6,7,8,9,10
    # alignment 46 55 4,5,6,7,8,9,10

    image_dir = "/home/anna/Pictures/pool/bags/0035SET/000"
    # image_dir = "/home/anna/Pictures/grid_for_alignment/0046"
    name = "15:00_reference"
    altitude_change = 46

    image_numbers = [f"{x:04}" for x in (19,)]
    panel_image_nr = "0000"

    durations = []

    for image_nr in image_numbers:

        img_capt, panel_capt = load_image_set(
            image_dir, image_nr, panel_image_nr
        )

        print(f"Loaded {len(img_capt.images)} images.\nAligning images...")
        with exiftool.ExifToolHelper() as et:
            altitude = et.get_tags(f"{image_dir}/IMG_{image_nr}_1.tif", ["Composite:GPSAltitude"])[0]["Composite:GPSAltitude"]

        img_type, irradiance_list = get_irradiance(img_capt, panel_capt, display=False)

        filename = f"./out/{name}/{image_nr}_{altitude-altitude_change:.0f}"
        
        # ALIGN THE IMAGE
        start = time.time()
        im_aligned = align_from_saved_matrices(img_capt, img_type, f"./out/warp_matrices_reference/warp_matrices_{altitude-altitude_change}.npy")
        # im_aligned, warp_matrices = align_iterative(img_capt, img_type)
        # im_aligned = align_rig_relatives(img_capt, img_type)
        # sharpened_stack, im_aligned = align_SIFT(img_capt, img_type, irradiance_list, matrices_fn=f"./out/{name}/warp_matrices_{altitude-altitude_change}.npy")
        # save_warp_matrices(warp_matrices,f"./out/{name}/warp_matrices_{altitude-altitude_change}.npy")

        duration = time.time() - start
        print(f"Alignment took {duration:.2f} s.")
        durations.append(duration)
        

        # SAVE VIEWS
        RGB_image = get_components_view(im_aligned, (2,1,0))
        save_image(RGB_image, f"{filename}_RGB.png", bgr=True)

        RGB_image = get_components_view(im_aligned, (2,4,3))
        save_image(RGB_image, f"{filename}_243.png", bgr=True)

        NDWI_image = get_index_view(im_aligned, CHANNEL_NAMES.index("NIR"), CHANNEL_NAMES.index("G"))
        save_image(NDWI_image, f"{filename}_NDWI.png")

        # NDVI_image = get_index_view(im_aligned, CHANNEL_NAMES.index("NIR"), CHANNEL_NAMES.index("R"))
        # save_image(NDVI_image, f"{filename}_NDVI.png")

        # PI = get_PI_image(im_aligned)
        # save_image(PI, f"{filename}_PI.png")

    with open(f"./out/{name}/alignment_times.txt", mode="w") as file:
        for duration in durations:
            file.write(f"{duration}\n")

