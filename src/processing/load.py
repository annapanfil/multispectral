import os
import sys
import exiftool
import numpy as np
from pathlib import Path
import cv2
from omegaconf import ListConfig
import skimage
from skimage.transform import ProjectiveTransform
sys.path.append('/home/anna/code/multispectral/libraries/imageprocessing')

import micasense.capture as capture
import micasense.imageutils as imageutils

def load_not_aligned(image_dir: str, image_nr: str, panel_image_nr: int, 
                    altitude: int, 
                    warp_matrices_path: str = "/home/anna/code/multispectral/out/warp_matrices_reference/"
                    ) -> np.ndarray:
        img_capt, panel_capt = load_image_set(
            image_dir,
            image_nr, 
            panel_image_nr
        )

        print(f"Loaded {len(img_capt.images)} images with altitude {altitude}\nAligning images...")

        img_type = get_irradiance(img_capt, panel_capt)

        im_aligned = align_from_saved_matrices(img_capt, img_type, warp_matrices_path , altitude, True)
        return im_aligned


def load_aligned(image_path, image_number) -> np.ndarray:
    img_names = find_images(Path(image_path), image_number)
    images = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in img_names]
    images = np.stack(images, axis=-1, dtype=np.float32)

    return images

def get_altitude(cfg, image_nr, i):
    """Read altitude based on the information from exif and config file"""
    if "altitude" in cfg.params:
        if isinstance(cfg.params.altitude, ListConfig):
            altitude = cfg.params.altitude[i]
        else:
            altitude = cfg.params.altitude
    else:
        try:
            with exiftool.ExifToolHelper() as et:
                altitude = et.get_tags(Path(cfg.paths.images, f"IMG_{image_nr}_1.tif"), ["Composite:GPSAltitude"])[0]["Composite:GPSAltitude"]
        except exiftool.exceptions.ExifToolExecuteError:
            print(f"ExifToolError: Could not read altitude from exif of {Path(cfg.paths.images, f'IMG_{image_nr}_1.tif')}. Please provide altitude in the config file.")
            sys.exit(1)
            
        if isinstance(cfg.params.altitude_change, ListConfig): 
            altitude = int(altitude - cfg.params.altitude_change[i])
        else:
            altitude = int(altitude - cfg.params.altitude_change)
    return altitude

def find_images(image_path:Path, image_number:str, panel=False, with_set=None):
    """Find images for a given capture number."""    

    image_names = list(image_path.glob('IMG_'+ image_number + '_*.tif*'))
    if image_names == []:
        image_names = sorted(list(image_path.glob(f"*_{image_number}_*_ch*.tif*")))

    image_names = [x.as_posix() for x in image_names]
   
    if len(image_names) == 0 and not with_set:
        image_names = find_images(image_path.joinpath(f"{int(image_number)//200:03}"), image_number, panel, with_set=True)
    
    if len(image_names) == 0:
        photo_type = "Images" if not panel else "Panel images"
        raise FileNotFoundError(f"{photo_type} not found in path {image_path}/IMG_{image_number}_*.tif* or {image_path}/*_{image_number}_*_ch*.tif*")

    return image_names

def load_image_set(
        image_path="/home/anna/Obrazy/multispectral/0001SET/000/",
        image_number="0000",
        panel_image_number=None
    ):
    """Load 6 images into one capture object
    @param image_path path to all of the photos. Assuming all images in the capture are in the same directory
    @param image_number number of capture to be displayed
    @param panel_image_number number of capture with a calibration QR code
    """

    img_names = find_images(Path(image_path), image_number)
    img_capt = capture.Capture.from_filelist(img_names)
    
    # QR code photos
    panel_names = find_images(Path(image_path), panel_image_number, panel=True) if panel_image_number is not None else None
    panel_capt = capture.Capture.from_filelist(panel_names) if panel_names is not None else None

    for img in img_capt.images:
        if img.rig_relatives is None:
            raise ValueError("Images must have RigRelatives tags set which requires updated firmware and calibration. See the links in text above")
    
    return img_capt, panel_capt

def get_irradiance(img_capt, panel_capt, display=False):
    """ get irradiance and image type and display
    @return reflectance or radiance
    """
    if panel_capt is not None:
        if panel_capt.panel_albedo() is not None:
            panel_reflectance_by_band = panel_capt.panel_albedo()
        else:
            panel_reflectance_by_band = [0.49, 0.49, 0.49, 0.49, 0.49] #RedEdge band_index order
        panel_irradiance = panel_capt.panel_irradiance(panel_reflectance_by_band)    
        irradiance_list = panel_capt.panel_irradiance(panel_reflectance_by_band) + [0] # add to account for uncalibrated LWIR band, if applicable
        img_type = "reflectance"
        to_plot = panel_irradiance
    else:
        if img_capt.dls_present():
            img_type='reflectance'
            irradiance_list = img_capt.dls_irradiance() + [0]
            to_plot = img_capt.dls_irradiance()
        else:
            img_type = "radiance"
            irradiance_list = None
    
    for img, irradiance in zip(img_capt.images, irradiance_list):
            img.reflectance(irradiance)

    if display:
        if img_type == "reflectance":
            img_capt.plot_undistorted_reflectance(to_plot)
        elif img_type == "radiance":
            img_capt.plot_undistorted_radiance()

    return img_type

def save_warp_matrices(warp_matrices, fn="./out/warp_matrices_SIFT.npy"):
    np.save(fn, np.array(warp_matrices), allow_pickle=True)
    print("Saved to", Path(fn).resolve())
   
def read_warp_matrices_for_SIFT(fn="./out/warp_matrices_SIFT.npy"):
    """Read warp matrices and transform them into ProjectiveTransform objects"""
    if Path(fn).is_file():
        load_warp_matrices = np.load(fn, allow_pickle=True)
        warp_matrices = []
        for matrix in load_warp_matrices: 
            transform = ProjectiveTransform(matrix=matrix.astype('float64'))
            warp_matrices.append(transform)
        print("Warp matrices successfully loaded.")
    else:
        print(f"No existing warp matrices found in path {fn}")
        warp_matrices = False

    return warp_matrices


def get_saved_matrices(warp_matrices_dir: str, altitude: int, allow_closest=False) -> np.array:
    """Get saved matrices from warp_matrices_dir with the given altitude
    @param allow_closest: if the matrices file is not found, get the closest one by the altitude.
    @return: warp_matrices"""
    fn = f"{warp_matrices_dir}/warp_matrices_{altitude}.npy"

    if Path(fn).is_file():
        warp_matrices = np.load(fn, allow_pickle=True).astype(np.float32)
        print(f"Warp matrices for altitude {altitude} successfully loaded.")
    else:
        if allow_closest:
            available_altitudes = [int(x.split(".")[0].split("_")[2]) for x in os.listdir(warp_matrices_dir) if x.endswith(".npy")]
            closest_altitude = min(available_altitudes, key=lambda x: abs(x - altitude))
            print(f"No existing warp matrices found for altitude {altitude}. Using closest altitude {closest_altitude}.")
            warp_matrices = get_saved_matrices(warp_matrices_dir, closest_altitude, allow_closest=False)
        else:
            raise FileNotFoundError(f"No existing warp matrices found in path {fn}")
    
    return warp_matrices

def align_rig_relatives(capt, img_type):
    """ align using rig relatives """
    reference_band = 5
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrices = capt.get_warp_matrices(ref_index=reference_band)

    cropped_dimensions, edges = imageutils.find_crop_bounds(capt,warp_matrices,reference_band=reference_band)
    im_aligned = imageutils.aligned_capture(
        capt, warp_matrices, warp_mode, cropped_dimensions, reference_band, img_type=img_type)

    return im_aligned

def align_SIFT(capture, img_type, irradiance_list, matrices_fn="./out/warp_matrices_SIFT.npy", verbose=0):
    """
    Align and sharpen multispectral images using SIFT algorithm.

    This function performs image alignment using the Scale-Invariant Feature Transform (SIFT) 
    algorithm and then applies radiometric pan-sharpening to the aligned images.

    @param capture (Capture): The Capture object containing the multispectral images to be aligned.
    @param img_type (str): The type of image data, e.g., 'reflectance' or 'radiance'.
    @param irradiance_list (list): List of irradiance values for each band.

    @return sharpened_stack (numpy.ndarray): The radiometrically pan-sharpened and aligned image stack.
    @return im_aligned (list): List of aligned images before pan-sharpening.
    """
    warp_matrices = read_warp_matrices_for_SIFT(matrices_fn)
    if warp_matrices is False:
        warp_matrices = capture.SIFT_align_capture(
            min_matches = 0,
            verbose=verbose,
            err_red=100.0, err_blue=100.0
            )
    
        # save the warp matrices for future use in the same flight
        save_warp_matrices(warp_matrices, matrices_fn)
    
    sharpened_stack, im_aligned = capture.radiometric_pan_sharpened_aligned_capture(
        warp_matrices=warp_matrices,
        irradiance_list=irradiance_list,
        img_type=img_type)
    
    return sharpened_stack, im_aligned


def align_iterative(capture, img_type):
    match_index = 5 # Index of the band 
    max_alignment_iterations = 20
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    pyramid_levels = 1 # for images with RigRelatives, setting this to 0 or 1 may improve alignment

    print("Aligning images. Depending on settings this can take from a few seconds to many minutes")
    # Can potentially increase max_iterations for better results, but longer runtimes
    warp_matrices, alignment_pairs = imageutils.align_capture(capture,
                                                            ref_index = match_index,
                                                            max_iterations = max_alignment_iterations,
                                                            warp_mode = warp_mode,
                                                            pyramid_levels = pyramid_levels)
    
    cropped_dimensions, edges = imageutils.find_crop_bounds(capture, warp_matrices, warp_mode=warp_mode, reference_band=match_index)
    print(cropped_dimensions)
    im_aligned = imageutils.aligned_capture(capture, warp_matrices, warp_mode, cropped_dimensions, match_index, img_type=img_type)

    return im_aligned, warp_matrices


def align_from_saved_matrices(capture, img_type: str, warp_matrices_dir: str, altitude: int, allow_closest=False):
    match_index = 5
    warp_mode = cv2.MOTION_HOMOGRAPHY
    
    warp_matrices = get_saved_matrices(warp_matrices_dir, altitude, allow_closest=allow_closest)
    cropped_dimensions, edges = imageutils.find_crop_bounds(capture, warp_matrices, warp_mode=warp_mode, reference_band=match_index)
    im_aligned = imageutils.aligned_capture(capture, warp_matrices, warp_mode, cropped_dimensions, match_index, img_type=img_type)

    return im_aligned