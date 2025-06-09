import time
from typing import List, Tuple
import cv2

import numpy as np
from src.shapes import Rectangle
from src.processing.consts import CHANNELS, CAM_HFOV, CAM_VFOV
from src.processing.evaluate_index import apply_formula


def prepare_image(img_aligned: np.array, channels: List, is_complex: bool, new_size: Tuple[int, int]) -> np.array:
    """ Get the correct channels for prediction and resize the image.
    Args:
        im_aligned (np.array): The aligned image.
        channels (list): List of channels or formulas to use.
        is_complex (bool): Whether the image is complex or not.
        new_size (tuple): New size for the image.
    Returns:
        np.array: The prepared image.
    """
    # normalize to [0, 255]
    for i in range(0, img_aligned.shape[2]):
        img_aligned[:, :, i] = (img_aligned[:, :, i] - np.min(img_aligned[:, :, i])) / (np.max(img_aligned[:, :, i]) - np.min(img_aligned[:, :, i])) * 255

    height, width = img_aligned.shape[:2]

    
    image = np.zeros((height, width, len(channels)))

    for i, channel in enumerate(channels):
        if channel in CHANNELS:
            image[:, :, i] = img_aligned[:, :, CHANNELS[channel]]
        else:
            image[:, :, i] = apply_formula(img_aligned, channel, is_complex)

    image = cv2.resize(image, new_size)
    image = image.astype(np.uint8)

    return image

def greedy_grouping(rectangles: List[Rectangle], image_shape: Tuple, resize_factor=1.5, visualize=False, confidences: List[float] = None) -> Tuple[List, np.array]:
    """
    Merge intersecting rectangles.
    Args:
        rectangles (list): List of rectangles to merge.
        image_shape (tuple): Shape of the image (height, width).
        resize_factor (float): Factor to enlarge the rectangles for merging.
        visualize (bool): Whether to visualize the merging process.
        confidences(list): Confidences for the bbs
    Returns: tuple of
        merged_rectangles (list): List of merged rectangles.
        merged_rectangles_mask (np.array): Visualisation of the merged rectangles.
    """
    
    merged_mask = np.zeros(image_shape, dtype=np.uint8)

    for rectangle in rectangles:
        enlarged_rectangle = rectangle * resize_factor
        mask = np.zeros(image_shape, dtype=np.uint8)
        enlarged_rectangle.draw(mask, color=255, thickness=-1)
        merged_mask = cv2.bitwise_or(merged_mask, mask)

    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if visualize:
        merged_mask = cv2.cvtColor(merged_mask, cv2.COLOR_GRAY2RGB)

    merged_rectangles = []
    merged_confidences = []
    for contour in contours:
        # find original rectangles in the groups
        group_rectangles = [rectangle for rectangle in rectangles if cv2.pointPolygonTest(contour, (rectangle.x_l, rectangle.y_b), False) >= 0]

        if group_rectangles:

            merged_rectangles.append(Rectangle(
                x_l = min(r.x_l for r in group_rectangles),
                x_r = max(r.x_r for r in group_rectangles),
                y_b = min(r.y_b for r in group_rectangles),
                y_t = max(r.y_t for r in group_rectangles)
            ))

            if confidences is not None:
                merged_confidences.append(np.mean([confidences[i] for i, r in enumerate(rectangles) if r in group_rectangles]))
        
            if visualize:
                merged_rectangles[-1].draw(merged_mask, color=(255, 0, 0), thickness=5)

    return merged_rectangles, merged_mask, merged_confidences

def get_real_piles_size(
    im_shape: np.array, altitude: float, cam_hfov: float, cam_vfov: float, rectangles: list
):
    sizes = []
    for rect in rectangles:
        width_px, height_px = rect.width, rect.height
        im_height_px, im_width_px = im_shape

        image_width_m = 2 * altitude * np.tan(cam_hfov / 2)
        image_height_m = 2 * altitude * np.tan(cam_vfov / 2)

        width_m = image_width_m / im_width_px * width_px
        height_m = image_height_m / im_height_px * height_px

        sizes.append((width_m, height_m))
    return sizes

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.6f} s")
        return result
    return wrapper