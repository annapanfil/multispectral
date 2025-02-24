import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List, Tuple

from detection.shapes import Circle, Rectangle

def show_images(images, titles=None, show=True):
    num_images = len(images)
    
    if titles is None:
        titles = ['Image {}'.format(i + 1) for i in range(num_images)]

    figsize = (5 * num_images, 5)
    figure, _ = plt.subplots(1, num_images, figsize=figsize)
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
    if show:
        plt.show()

    return figure
    

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


def draw_rectangles(image: np.array, rectangles: List[Rectangle], color: Tuple[int] = (255, 0, 0), strength: int = 2) -> np.array:
    for rectangle in rectangles:
        cv2.rectangle(image, (rectangle.x_l, rectangle.y_b), (rectangle.x_r, rectangle.y_t), color, strength)

    return image


def draw_litter(image: np.array, pool: Rectangle, blob_contours, bbs: List[Rectangle] = None, dog_image: np.array = None, mask: np.array = None, out_path: str = None, show: bool=False, color=(100,200,0)) -> np.ndarray:
    im_pool = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
    cv2.rectangle(im_pool, (pool.x_l, pool.y_b), (pool.x_r, pool.y_t), (0,0,255), 5)
    
    cropped_image = image[pool.y_b:pool.y_t, pool.x_l:pool.x_r]

    # Draw the contours
    im_detected = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB) if len(cropped_image.shape) == 2 else cropped_image
    
    cv2.drawContours(im_detected, blob_contours, -1,color, 2)
    
    if bbs is not None:
        draw_rectangles(im_detected, bbs, color)

    if dog_image is not None and mask is not None:
        figure = show_images([im_pool, dog_image, mask, im_detected], ["detected pool", "after DOG", "mask (DOG + threshold)", "litter found"], show=show)
    else:
        figure = show_images([im_pool, im_detected], ["detected pool", "litter found"], show=show)
    if out_path != "":
        figure.savefig(f"{out_path}_detected_verb.png")
        cv2.imwrite(f"{out_path}_detected.png", im_detected)

    plt.close()

    return im_detected
