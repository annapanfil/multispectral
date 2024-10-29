#!/home/anna/miniconda3/envs/micasense/bin/python3

import argparse
import os

import exiftool

from load import load_image_set, get_irradiance, align
from visualise import get_components_view, show_image, save_image, plot_all_channels, plot_one_channel
from gui import show_components_interactive

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Align photos from multispectral camera")
    parser.add_argument('--image_dir', '-d', type=str, required=False, help='Path to the folder with images', default="/home/anna/Obrazy/multispectral")
    parser.add_argument('--set_nr', '-s', type=int, required=True, help='Set number')
    parser.add_argument('--image_nr', '-i', type=int, required=True, help='Photo number')
    parser.add_argument('--panel_image_nr', '-p', type=int, required=False, help='Panel (QR code) image number')
    parser.add_argument('--output_dir', '-o', type=str, required=False, help='Output folder, if not specified, only displaying')
    parser.add_argument('--verbose', '-v', type=int, required=False, help="If to show photos in the middle of the process (0-dont show anything, 1-show only composites, 2-show everything)", default=1)

    args = parser.parse_args()
    if args.panel_image_nr is not None:
        args.panel_image_nr = f"{args.panel_image_nr:04}"

    # Create output directory if specified
    output = args.output_dir

    if output and not os.path.exists(output):
        print(f"creating directory {output}")
        os.makedirs(output)

    if not os.path.exists("./out"):
        print("creating directory ./out")
        os.makedirs("./out")

    out_fn = f"{output}/{args.image_nr:04}" if output else None

    return args, out_fn


if __name__ == "__main__":
    RESOLUTION = "full"
    args, out_fn = get_args()

    img_capt, panel_capt = load_image_set(
        f"{args.image_dir}/{args.set_nr:04}SET/000/",
        f"{args.image_nr:04}", 
        args.panel_image_nr
    )

    print(f"Loaded {len(img_capt.images)} images.\nAligning images...")
    with exiftool.ExifToolHelper() as et:
        altitude = et.get_tags(f"{args.image_dir}/{args.set_nr:04}SET/000/IMG_{args.image_nr:04}_1.tif", ["Composite:GPSAltitude"])[0]["Composite:GPSAltitude"]

    img_type, irradiance_list = get_irradiance(img_capt, panel_capt, display=True if args.verbose > 1 else False)
    sharpened_stack, im_aligned = align(img_capt, img_type, irradiance_list, matrices_fn=f"./out/warp_matrices_{altitude}.npy")

    # visualise
    figsize=(30,23) if RESOLUTION=="full" else (16,13)
    

    show_components_interactive(sharpened_stack, img_type, img_no=f"{args.image_nr:04}")

    # plot_all_channels(im_aligned,
    #                   out_fn=f"{out_fn}_all_channels.jpg" if out_fn else None,
    #                   show=(args.verbose > 1)
    #                   )

    rgb_image = get_components_view(im_aligned, img_type, band_indices=(2,1,0))
    if args.verbose > 1: show_image(rgb_image, "RGB composite", figsize)
    if out_fn: save_image(rgb_image, f"{out_fn}_RGB.jpg", bgr=True)

    # cir_image =  show_components(im_aligned, img_type, band_indices=(3,2,1))
