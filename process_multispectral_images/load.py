import sys
import numpy as np
from pathlib import Path
import cv2
import skimage
from skimage.transform import ProjectiveTransform
sys.path.append('/home/anna/code/imageprocessing')

import micasense.capture as capture
import micasense.imageutils as imageutils


def find_images(image_path, image_number, panel=False):
    """Find images for a given capture number."""

    image_names = list(image_path.glob('IMG_'+ image_number + '_*.tif'))
    image_names = [x.as_posix() for x in image_names]
   
    if len(image_names) == 0:
        photo_type = "Images" if not panel else "Panel images"
        raise FileNotFoundError(f"{photo_type} not found in path {image_path}IMG_{image_number}_*.tif")

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
    panel_names =find_images(Path(image_path), panel_image_number, panel=True) if panel_image_number is not None else None
    panel_capt = capture.Capture.from_filelist(panel_names) if panel_names is not None else None

    for img in img_capt.images:
        if img.rig_relatives is None:
            raise ValueError("Images must have RigRelatives tags set which requires updated firmware and calibration. See the links in text above")
    
    return img_capt, panel_capt

def undistort(img_capt, panel_capt, display=False):
    """ correct distortion and display
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
    
    if display:
        if img_type == "reflectance":
            img_capt.plot_undistorted_reflectance(to_plot)
        elif img_type == "radiance":
            img_capt.plot_undistorted_radiance()

    return img_type, irradiance_list


def align_fast(capt, img_type):
    """ align using rig relatives """
    reference_band = 5
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrices = capt.get_warp_matrices(ref_index=reference_band)

    cropped_dimensions, edges = imageutils.find_crop_bounds(capt,warp_matrices,reference_band=reference_band)
    im_aligned = imageutils.aligned_capture(
        capt, warp_matrices, warp_mode, cropped_dimensions, reference_band, img_type=img_type)

    return im_aligned

def save_warp_matrices(warp_matrices, fn="./out/warp_matrices_SIFT.npy"):
    np.save(fn, np.array(warp_matrices, dtype=object), allow_pickle=True)
    print("Saved to", Path(fn).resolve())
   
def read_warp_matrices(fn="./out/warp_matrices_SIFT.npy"):
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

def align(capture, img_type, irradiance_list, matrices_fn="out/warp_matrices_SIFT.npy", verbose=0):
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
    warp_matrices = read_warp_matrices(matrices_fn)
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