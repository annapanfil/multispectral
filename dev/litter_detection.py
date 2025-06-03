import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from skimage.filters import difference_of_gaussians
from skimage import img_as_float

from src.shapes import Circle, Rectangle


def find_pool(image: np.array, altitude: int, threshold: int = 110, verb=False) -> Rectangle:
    """Get the dark, big rectangle from image – the pool"""

    if altitude < 7:
        return Rectangle(x_l=0, x_r=image.shape[1], y_b=0, y_t=image.shape[0])

    blurred = cv2.GaussianBlur(image, (25, 25), 0)

    if verb:
        plt.subplots(2, 3, figsize=(26, 16))
        plt.subplot(2, 3, 1)
        plt.imshow(blurred, cmap="gray")

    blurred[blurred < threshold] = 0
    blurred[blurred >= threshold] = 255

    blurred = cv2.copyMakeBorder(blurred, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

    if verb:
        plt.subplot(2, 3, 2)
        plt.imshow(blurred, cmap="gray")

    # Apply Sobel filter (gradient in x and y directions)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)
    edges = cv2.convertScaleAbs(edges)

    if verb:
        plt.subplot(2, 3, 3)
        plt.imshow(edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if verb:
        plt.subplot(2, 3, 4)
        plt.imshow(
            cv2.drawContours(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), contours, -1, (0, 255, 0), 5)
        )
        plt.subplot(2, 3, 5)
        contour_photo = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Find the largest quadrilateral with the correct ratio and big enough area (pool) in the image
    max_area_rectangle = None
    max_area = 0
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.015 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if verb:
            cv2.drawContours(contour_photo, approx, -1, (0, 255, 0), 5)

        # Check if the approximated contour has 4 vertices (indicating a quadrilateral)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Define acceptable aspect ratio range for rectangles
            if 1.5 < aspect_ratio < 2.5:

                # get the largest rectangle
                area = cv2.contourArea(approx)

                # Check if the area of the rectangle is within the acceptable range
                if area > 0.02 * image.shape[0] * image.shape[1]:
                    if area > max_area:
                        max_area = area
                        max_area_rectangle = Rectangle(x_l=x, y_b=y, x_r=x + w, y_t=y + h)

    if verb:
        if max_area_rectangle is not None:
            cv2.rectangle(
                contour_photo,
                (max_area_rectangle.x_l, max_area_rectangle.y_b),
                (max_area_rectangle.x_r, max_area_rectangle.y_t),
                (255, 0, 0),
                2,
            )
        plt.imshow(contour_photo)
        plt.show()

    return max_area_rectangle


def apply_dog(image: np.array, sigma: int, threshold=0.03) -> Tuple[np.array, np.array]:
    """
    Apply difference of gaussians
    """
    float_image = img_as_float(image)
    dog_image = difference_of_gaussians(float_image, sigma, 1.6 * sigma)

    mask = dog_image.copy()
    _, mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    return mask, dog_image


def pool2abs_rect(bboxes: List[Rectangle], pool: Rectangle) -> List[Rectangle]:
    abs_bboxes = []
    for bbox in bboxes:
        abs_bbox = Rectangle(
            x_l=bbox.x_l + pool.x_l,
            y_b=bbox.y_b + pool.y_b,
            x_r=bbox.x_r + pool.x_l,
            y_t=bbox.y_t + pool.y_b,
        )
        abs_bboxes.append(abs_bbox)

    return abs_bboxes


def pool2abs_point(point: Tuple[float, float], pool: Rectangle) -> Tuple[float, float]:
    return point[0] + pool.x_l, point[1] + pool.y_b


def get_figures_from_contours(contours) -> Tuple[List[Rectangle], List[Circle]]:
    """
    Extract bounding boxes as rectangles and circles from the contours
    """
    rectangles = []
    circles = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Get the circle around the ROI
        center = (x + w // 2, y + h // 2)
        radius = int(max(w, h) / 2)
        circle = Circle(x=center[0], y=center[1], r=radius)

        # filter by size
        if circle.r:
            circles.append(circle)
            rectangles.append(Rectangle(x_l=x, y_b=y, x_r=x + w, y_t=y + h))

    return rectangles, circles


def detect_blobs(
    image: np.array, sigma: int, threshold: float
) -> Tuple[Tuple[np.array], np.array, np.array]:
    """Get contours of blobs from the image"""
    mask, dog_image = apply_dog(image, sigma, threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, dog_image, mask


def find_litter(
    image: np.array,
    im_name,
    sigma: int,
    dog_threshold: float,
    size_max_threshold_perc: float,
    verb=False,
) -> Tuple[List[int], List[Rectangle], Rectangle, np.array, np.array]:
    pool = find_pool(image, int(im_name.split("_")[-2]), verb=verb)

    if pool is None:
        print(f"Pool not found in {im_name}. Trying raising contrast")
        equalized_image = cv2.equalizeHist(image)
        pool = find_pool(equalized_image, int(im_name.split("_")[-2]), verb=verb)
        if pool is None:
            print("Pool not found")
            return

    cropped_image = image[pool.y_b : pool.y_t, pool.x_l : pool.x_r]

    blob_contours, dog_image, mask = detect_blobs(cropped_image, sigma, dog_threshold)

    # filter out
    blob_contours = [
        contour
        for contour in blob_contours
        if cv2.minEnclosingCircle(contour)[1] < size_max_threshold_perc * (pool.x_r - pool.x_l)
    ]

    bb_rectangles, bb_circles = get_figures_from_contours(blob_contours)

    return blob_contours, bb_rectangles, pool, dog_image, mask


def merge_rectangles(rects: List[Rectangle], margin=0) -> List[Rectangle]:
    """ merge rectangles that overlap to bigger rectangles """
    merged = [rects[0]]

    for rect in rects[1:]:
        for m in merged:
            if m.intersection(rect, margin):
                merged.remove(m)
                merged.append(m | rect)
                break
        else:
            merged.append(rect)

    # check if we cant merge any more
    changed = True
    while changed:
        changed = False
        for i, rect in enumerate(merged):
            for j, other in enumerate(merged):
                if i != j and rect.intersection(other, margin):
                    merged.remove(rect)
                    merged.remove(other)
                    merged.append(rect | other)
                    changed = True
                    break

    return merged


def group_contours(contours: list, margin: int, image: np.array):
    # enlarge contours
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.drawContours(mask, contours, -1, 255, margin)
    cv2.drawContours(mask, contours, -1, 255, -1)

    # reduce joined contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))
    mask = cv2.erode(mask, kernel, iterations=1)

    # find rectangles
    joined_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects_for_drawing = []
    rects = []
    for contour in joined_contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rects.append(rect)
        rects_for_drawing.append(box)

    return joined_contours, rects, rects_for_drawing


def get_real_piles_size(
    im_shape: np.array, altitude: float, cam_hfov: float, cam_vfov: float, rectangles: list
):
    sizes = []
    for rect in rectangles:
        width_px, height_px = rect[1]
        im_height_px, im_width_px = im_shape

        image_width_m = 2 * altitude * np.tan(cam_hfov / 2)
        image_height_m = 2 * altitude * np.tan(cam_vfov / 2)

        width_m = image_width_m / im_width_px * width_px
        height_m = image_height_m / im_height_px * height_px

        sizes.append((width_m, height_m))
    return sizes
