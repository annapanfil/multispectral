import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from skimage.filters import difference_of_gaussians
from skimage import img_as_float

from shapes import Circle, Rectangle


def find_pool(image: np.array) -> Rectangle:

    image = cv2.GaussianBlur(image, (23,23), 0)

    image[image<130] = 0
    image[image>=130] = 255

    # Apply Sobel filter (gradient in x and y directions)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)
    edges = cv2.convertScaleAbs(edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest quadrilateral with the correct ratio (pool) in the image

    max_area_rectangle = None
    max_area = 0
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the approximated contour has 4 vertices (indicating a quadrilateral)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Define acceptable aspect ratio range for rectangles
            if 1.5 < aspect_ratio < 2.5:  # Adjust this range as needed

                # get the largest rectangle
                area = cv2.contourArea(approx)
                if area > max_area:
                    max_area = area
                    max_area_rectangle = Rectangle(x_l=x, y_b=y, x_r=x+w, y_t=y+h)        

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


def get_figures_from_contours(contours, image_shape, threshold) -> Tuple[List[Rectangle], List[Circle]]:
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
      if circle.r < threshold * image_shape[0]:
         circles.append(circle)     
         rectangles.append(Rectangle(x_l=x, y_b=y, x_r=x+w, y_t=y+h))

   return rectangles, circles     


def detect_blobs(image: np.array, sigma: int, threshold: float) -> Tuple[Tuple[np.array], np.array, np.array]:
   """ Get contours of blobs from the image """
   mask, dog_image = apply_dog(image,  sigma, threshold)
   contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   return contours, dog_image, mask
