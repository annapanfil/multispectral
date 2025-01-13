#!/home/anna/miniconda3/envs/micasense/bin/python3

import argparse
import glob
import os

import exiftool

from load import load_image_set, get_irradiance, align_from_saved_matrices
from gui import show_components_interactive

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Align photos from multispectral camera")
    parser.add_argument('--image_dir', '-d', type=str, required=True, help='Path to the folder with images')
    parser.add_argument('--set_nr', '-s', type=int, required=False, help='Set number (only for specifying altitude difference)')
    parser.add_argument('--image_nr', '-i', type=int, required=True, help='Photo number')
    parser.add_argument('--panel_image_nr', '-p', type=int, required=False, help='Panel (QR code) image number')
    parser.add_argument('--output_dir', '-o', type=str, required=False, help='Output folder, if not specified, only displaying')
    parser.add_argument('--verbose', '-v', type=int, required=False, help="If to show photos in the middle of the process (0-dont show anything, 1-show only composites, 2-show everything)", default=1)
    parser.add_argument('--altitude', '-a', type=int, required=False, help="Altitude from which the photo was taken")

    args = parser.parse_args()
    if args.panel_image_nr is not None:
        args.panel_image_nr = f"{args.panel_image_nr:04}"

    # Check if file exists
    if (image_path := f"{args.image_dir}/IMG_{args.image_nr:04}_1.tif") and os.path.exists(image_path):
        pass 
    elif (image_path := glob.glob(f"{args.image_dir}/*_{args.image_nr:04}_*_ch1.tif*")):
        image_path = image_path[0]

    if not image_path:
        raise FileNotFoundError(f"Image not found in path {args.image_dir}/IMG_{args.image_nr:04}_1.tif or {args.image_dir}/*_{args.image_nr:04}_*_ch1.tif*")

    # Wyświetlamy, gdzie znaleziono plik
    print(f"Found image at: {image_path}")


    # get the altitude from the known images if not specified
    if args.altitude is None and args.set_nr is None:
        raise LookupError("You must specify the altitude or at least the set number.")

    if args.altitude is None:
        with exiftool.ExifToolHelper() as et:
            try:
                altitude = et.get_tags(image_path, ["Composite:GPSAltitude"])[0]["Composite:GPSAltitude"]
            except (exiftool.exceptions.ExifToolExecuteError, KeyError):
                raise LookupError(f"Cannot read altitude from the file {image_path} You must specify the altitude.")
        
        differences = {33: 55, 34: 54, 35: 45, 36: 50, 41: 35, 45: 59, 46: 55, 49: 47, 50: 47, 51: 46}
        if args.set_nr is None:
            print(Warning(f"No set number specified. Using the altitude from the file ({altitude} m), without any change."))

        if args.set_nr not in differences.keys():
            raise LookupError(f"Not known altitude difference for set {args.set_nr}. You must specify the altitude or set the difference in the code.")
    
        args.altitude = altitude - differences[args.set_nr]

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
        args.image_dir,
        f"{args.image_nr:04}", 
        args.panel_image_nr
    )

    print(f"Loaded {len(img_capt.images)} images with altitude {args.altitude}\nAligning images...")

    img_type = get_irradiance(img_capt, panel_capt)

    im_aligned = align_from_saved_matrices(img_capt, img_type, "/home/anna/code/process_multispectral_images/out/warp_matrices_reference/", args.altitude, True)
    show_components_interactive(im_aligned, img_no=f"{args.image_nr:04}")
