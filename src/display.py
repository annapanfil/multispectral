import matplotlib.pyplot as plt
import cv2
import numpy as np

from detection import get_figures_from_contours

def show_images(images, titles=None):
    num_images = len(images)
    
    if titles is None:
        titles = ['Image {}'.format(i + 1) for i in range(num_images)]

    figsize = (5 * num_images, 5)
    plt.subplots(1, num_images, figsize=figsize)
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        if len(images[i].shape) == 3:  # Color image
            img_display = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            plt.imshow(img_display)
        else:  # Grayscale image
            plt.imshow(images[i], cmap="gray")  
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_image(img, title="Image", cmap_type="gray", figsize=(8,4)):
    plt.figure(figsize=figsize)
    if len(img.shape) == 3:  # Color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    else:  # Grayscale image
        plt.imshow(img, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()



def draw_litter(image: np.array, blob_contours: list, size_max_threshold: float, circles=True, rectangles=False, contours=False):
   """
   Draw contours around litter blobs on the image
    :param image: Input grayscale image
    :param blob_contours: List of contours
    :param size_max_threshold: Maximum size threshold for blob detection (circles and rectangles) as a percentage of image width
    :param circles: if to draw circles around detected blobs
    :param rectangles: if to dra rectangles around detected blobs
    :param contours: if to dra contours around detected blobs
    :return: Image with detected blobs, Dog image, and mask
    """
   bb_rectangles, bb_circles = get_figures_from_contours(blob_contours, image.shape, size_max_threshold)

   # Draw the contours
   im_detected = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   if contours:
      cv2.drawContours(im_detected, blob_contours, -1, (100,200,0), 2)
   if circles:
      for circle in bb_circles:
         cv2.circle(im_detected, (circle.x, circle.y), circle.r, (255,0,0), 2)
   if rectangles:
      for rectangle in bb_rectangles:
         cv2.rectangle(im_detected, (rectangle.x_l, rectangle.y_b), (rectangle.x_r, rectangle.y_t), (255,0,0), 2)

   return im_detected
