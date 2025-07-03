import os
import sys
import exiftool
import numpy as np
from pathlib import Path
import cv2
from omegaconf import ListConfig
from skimage.transform import ProjectiveTransform
import concurrent.futures
from ..timeit import timer

import time
def time_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"    {func.__name__} took {end - start:.6f} s")
        return result
    return wrapper

sys.path.append('/home/anna/code/multispectral/libraries/imageprocessing')

import micasense.capture as capture
import micasense.imageutils as imageutils

def load_not_aligned(image_dir: str, image_nr: str, panel_image_nr: int, 
                    altitude: int, 
                    warp_matrices_path: str = "/home/anna/code/multispectral/out/warp_matrices_reference/"
                    ) -> np.ndarray:
    """
    Loads a set of images that are not aligned and aligns them using precomputed warp matrices.
    Args:
        image_dir (str): Directory containing the images.
        image_nr (str): Identifier for the image set.
        panel_image_nr (int): Identifier for the panel image.
        altitude (int): Altitude at which the images were captured.
        warp_matrices_path (str, optional): Path to the directory containing the warp matrices. Defaults to "/home/anna/code/multispectral/out/warp_matrices_reference/".
    Returns:
        np.ndarray: Aligned images as a NumPy array.
    """
    
    img_capt, panel_capt = load_image_set(
        image_dir,
        image_nr, 
        panel_image_nr
    )

    print(f"Loaded {len(img_capt.images)} images with altitude {altitude}\nAligning images...")

    img_type = get_irradiance(img_capt, panel_capt)

    im_aligned = align_from_saved_matrices(img_capt, img_type, warp_matrices_path , altitude, True)
    return im_aligned


def load_aligned(image_path: str, image_number: str) -> np.ndarray:
    """
    Loads already aligned set of multispectral images from a specified directory.
    Args:
        image_path (str): Path to the folder containing the images.
        image_number (str): Number of the capture to be displayed in format like: 0024 or part of image name without the channel.
    Returns:
        np.ndarray: A numpy array containing the stacked images with shape (height, width, 6).
    Raises:
        ValueError: If the number of images found is not equal to 6.
        ValueError: If the shapes of the images differ by more than 10 pixels in any dimension.
    Warns:
        Warning: If the shapes of the images differ by less than 10 pixels, they will be resized to match the first image's shape.
    """

    img_names = find_images(Path(image_path), image_number)
    if len(img_names) not in (5, 6):
        raise ValueError(f"The image should have 5 or 6 channels, not {len(img_names)}. ({image_path}, {image_number})")
    images = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in img_names]
    for i in range(len(images)):
        if images[0].shape != images[i].shape:
            if abs(images[0].shape[0] - images[i].shape[0]) > 10 and abs(images[0].shape[1] - images[i].shape[1]) > 10:
                raise (f"Images have shapes different by more than 10 pixels ({images[0].shape}, {images[i].shape}). Please check them.")
            else:
                Warning("Images have different shapes. Resizing to the first image's shape")
                images[i] = cv2.resize(images[i], (images[0].shape[1], images[0].shape[0]))
    
    images = np.stack(images, axis=-1, dtype=np.float32)

    return images

def get_altitude(cfg, image_nr: str, i: int):
    """Read altitude based on the information from exif and config file.
    Args:
        cfg (hydra config): Configuration object containing parameters and paths.
        image_nr (str): Image number to identify the specific image file.
        i (int): Index to access altitude or altitude change from a list if applicable.
    Returns:
        int: The altitude value.
    Raises:
        SystemExit: If the altitude cannot be read from the exif data and is not provided in the config file.
    """
   
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

def find_images(image_path:Path, image_number:str, panel=False, with_set=None, no_panchromatic=False):
    """
    Find images for a given capture number.
    Args:
        image_path (Path): The path to the directory containing the images.
        image_number (str): The capture number of the images to find or part of image name without the channel.
        panel (bool, optional): True if we search for panel images. Defaults to False.
        with_set (bool, optional): Internal parameter to handle recursive search. Defaults to None.
        no_panchromatic (bool, optional): If True, exclude panchromatic images (ch6). Defaults to False.
    Returns:
        list: A list of image file paths as strings.
    Raises:
        FileNotFoundError: If no images are found for the given capture number.
    """   

    image_names = sorted(list(image_path.glob('IMG_'+ image_number + '_*.tif*')))
    if image_names == []:
        image_names = sorted(list(image_path.glob(f"*{image_number}*_ch*.tif*")))
    if no_panchromatic:
        image_names = [x for x in image_names if "ch6" not in x.name and "_6.tif" not in x.name]

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
        panel_image_number=None,
        no_panchromatic=True
    ):
    """
    Load 6 images into one capture object.

    Args:
        image_path (str): Path to the directory containing the images.
        image_number (str): Number of the capture to be displayed.
        panel_image_number (str, optional): Number of the capture with a calibration QR code.
        no_panchromatic (bool, optional): If True, exclude panchromatic images (ch6). Defaults to True.

    Returns:
        tuple: A tuple containing the image capture object and the panel capture object.
    """


    img_names = find_images(Path(image_path), image_number, no_panchromatic=no_panchromatic)
    img_capt = capture.Capture.from_filelist(img_names)
    
    # QR code photos
    panel_names = find_images(Path(image_path), panel_image_number, panel=True, no_panchromatic=no_panchromatic) if panel_image_number is not None else None
    panel_capt = capture.Capture.from_filelist(panel_names) if panel_names is not None else None

    for img in img_capt.images:
        if img.rig_relatives is None:
            raise ValueError("Images must have RigRelatives tags set which requires updated firmware and calibration. See the links in text above")
    
    return img_capt, panel_capt

def get_panel_irradiance(panel_capt):
    """ Get irradiance from the panel capture.

    Args:
        panel_capt (Capture): The panel capture object."""

    if (albedo := panel_capt.panel_albedo()) is not None:
        panel_reflectance_by_band = albedo
    else:
        panel_reflectance_by_band = [0.49, 0.49, 0.49, 0.49, 0.49] #RedEdge band_index order
    return panel_capt.panel_irradiance(panel_reflectance_by_band)

@timer
def get_irradiance(img_capt, panel_capt, panel_irradiance, display=False, vignetting=True):
    """
    Get irradiance and image type and display.

    Args:
        img_capt (Capture): The image capture.
        panel_capt (Capture): The panel capture.
        panel_irradiance (list): List of irradiance values for each band from the panel capture.
        display (bool, optional): Whether to display the images. Defaults to False.
        vignetting (bool, optional): Whether to apply vignetting correction for reflectance. Defaults to True.
    
    Returns:
        str: 'reflectance' or 'radiance'
    """
    def _no_vignette(self):
        shape = self.raw().shape
        ones = np.ones(shape, dtype=float).T
        x = np.zeros_like(ones)
        y = np.zeros_like(ones)
        return ones, x, y        

    if panel_capt is not None:
        if not vignetting:
            for img in panel_capt.images:
                img.vignette = _no_vignette.__get__(img)

        irradiance_list = panel_irradiance + [0] # add to account for uncalibrated LWIR band, if applicable
        img_type = "reflectance"
        to_plot = panel_irradiance
    else:
        if img_capt.dls_present():
            img_type='reflectance'
            irradiance_list = img_capt.dls_irradiance() + [0]
            if display:
                to_plot = img_capt.dls_irradiance()
        else:
            img_type = "radiance"
            irradiance_list = None

    def compute_radiance(img, irradiance):
        if not vignetting:
            img.vignette = _no_vignette.__get__(img)
        img.reflectance(irradiance)
        return img

    with concurrent.futures.ThreadPoolExecutor() as executor:
       executor.map(compute_radiance, img_capt.images, irradiance_list)

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

def load_all_warp_matrices(warp_matrices_dir: str) -> dict:
    """
    Load all warp matrices from the specified directory.

    Args:
        warp_matrices_dir (str): The directory containing the warp matrices.

    Returns:
        dict: A dictionary mapping altitudes to their corresponding warp matrices.
    """
    warp_matrices = {}
    for fn in os.listdir(warp_matrices_dir):
        if fn.endswith(".npy"):
            altitude = int(fn.split(".")[0].split("_")[2])
            warp_matrices[altitude] = np.load(os.path.join(warp_matrices_dir, fn), allow_pickle=True).astype(np.float32)
    
    return warp_matrices

def get_warp_mat_for_altitude(warp_matrices: dict, altitude: int, allow_closest=False, verb=False) -> np.array:
    """
    Get saved matrices from warp_matrices_dir with the given altitude.

    Args:
        warp_matrices (dict): Dictionary containing warp matrices for different altitudes.
        altitude (int): The altitude for which the matrices should be loaded.   
        allow_closest (bool): If the matrices file is not found, get the closest one by the altitude.
    Returns:
        np.array: The warp matrices.
    """

    try:
        warp_mat = warp_matrices[altitude]
    except KeyError:
        if allow_closest:
            available_altitudes = list(warp_matrices.keys())
            closest_altitude = min(available_altitudes, key=lambda x: abs(x - altitude))
            if verb: print(f"No existing warp matrices found for altitude {altitude}. Using closest altitude {closest_altitude}.")
            warp_mat = warp_matrices[closest_altitude]
        else:
            raise FileNotFoundError(f"No existing warp matrices found for altitude {altitude}.")
        
    return warp_mat


def get_saved_matrices(warp_matrices_dir: str, altitude: int, allow_closest=False, debug=False) -> np.array:
    # TODO: delete - kept for backward compatibility
    """
    Get saved matrices from warp_matrices_dir with the given altitude.

    Args:
        warp_matrices_dir (str): The directory containing the warp matrices.
        altitude (int): The altitude for which the matrices should be loaded.
        allow_closest (bool): If the matrices file is not found, get the closest one by the altitude.
    Returns:
        np.array: The warp matrices.
    """
    fn = f"{warp_matrices_dir}/warp_matrices_{altitude}.npy"

    if Path(fn).is_file():
        warp_matrices = np.load(fn, allow_pickle=True).astype(np.float32)
        if debug:
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

def align_rig_relatives(capt, img_type, reference_band=5):
    """ align using rig relatives """
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrices = capt.get_warp_matrices(ref_index=reference_band)

    cropped_dimensions, edges = imageutils.find_crop_bounds(capt,warp_matrices,reference_band=reference_band)
    im_aligned = imageutils.aligned_capture(
        capt, warp_matrices, warp_mode, cropped_dimensions, reference_band, img_type=img_type)

    return im_aligned

def align_SIFT(capture, img_type, irradiance_list, matrices_fn="./out/warp_matrices_SIFT.npy", verbose=0) -> tuple:
    """
    Align and sharpen multispectral images using SIFT algorithm.

    Args:
        capture (Capture): The Capture object containing the multispectral images to be aligned.
        img_type (str): The type of image data, e.g., 'reflectance' or 'radiance'.
        irradiance_list (list): List of irradiance values for each band.
        matrices_fn (str, optional): Path to the file containing the warp matrices. Defaults to "./out/warp_matrices_SIFT.npy".
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        numpy.ndarray: The radiometrically pan-sharpened and aligned image stack.
        list: List of aligned images before pan-sharpening.
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


def align_iterative(capture, img_type, reference_band = 5):
    """ align iteratively """
    max_alignment_iterations = 20
    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
    pyramid_levels = 1 # for images with RigRelatives, setting this to 0 or 1 may improve alignment

    print("Aligning images. Depending on settings this can take from a few seconds to many minutes")
    # Can potentially increase max_iterations for better results, but longer runtimes
    warp_matrices, alignment_pairs = imageutils.align_capture(capture,
                                                            ref_index = reference_band,
                                                            max_iterations = max_alignment_iterations,
                                                            warp_mode = warp_mode,
                                                            pyramid_levels = pyramid_levels)
    
    cropped_dimensions, edges = imageutils.find_crop_bounds(capture, warp_matrices, warp_mode=warp_mode, reference_band=reference_band)
    print(cropped_dimensions)
    im_aligned = imageutils.aligned_capture(capture, warp_matrices, warp_mode, cropped_dimensions, reference_band, img_type=img_type)

    return im_aligned, warp_matrices

@timer
def align_from_saved_matrices(capture, img_type: str, warp_matrices_dir: str, altitude: int, allow_closest=False, reference_band=5):
    #TODO: delete string possibility - kept for backward compatibility
    """
    Align images using precomputed warp matrices.
    
    Args:
        capture: The image capture object containing the images to be aligned.
        img_type (str): 'reflectance' or 'radiance'
        warp_matrices_dir (str): Directory path where the warp matrices are stored.
        altitude (int): The altitude at which the images were captured.
        allow_closest (bool, optional): If True, if the matrices file is not found, get the closest one by the altitude. Defaults to False.
        reference_band (int, optional): The index of the reference band for alignment. Defaults to 5.
    
    Returns:
        im_aligned: The aligned image capture object.
    """
    if isinstance(warp_matrices_dir, str):
        warp_matrices = get_saved_matrices(warp_matrices_dir, altitude, allow_closest=allow_closest)
    else: # dict
        warp_matrices = get_warp_mat_for_altitude(warp_matrices_dir, altitude, allow_closest=allow_closest)

    warp_mode = cv2.MOTION_HOMOGRAPHY
    cropped_dimensions, edges = imageutils.find_crop_bounds(capture, warp_matrices, warp_mode=warp_mode, reference_band=reference_band)
    im_aligned = imageutils.aligned_capture(capture, warp_matrices, warp_mode, cropped_dimensions, reference_band, img_type=img_type)

    return im_aligned
