import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List, Tuple

from detection.shapes import Rectangle



def show_images(images, title):
    """
    Display a grid of images.
    """

    aspect_ratio = 4 / 3
    cols = int(np.ceil(np.sqrt(len(images) * aspect_ratio))) 
    rows = int(np.ceil(len(images) / cols))

    figure, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, image in enumerate(images):
        ax = axes[i]
        ax.imshow(image)
        ax.axis('off')

    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

    return figure


def show_image(img, title="Image", cmap_type="gray", figsize=(8, 4)):
    plt.figure(figsize=figsize)
    if len(img.shape) == 3:  # Color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    else:  # Grayscale image
        plt.imshow(img, cmap=cmap_type)
    plt.title(title)
    plt.axis("off")
    plt.show()


def draw_rectangles(
    image: np.array, rectangles: List[Rectangle], color: Tuple[int] = (255, 0, 0), strength: int = 2
) -> np.array:
    for rectangle in rectangles:
        cv2.rectangle(
            image, (rectangle.x_l, rectangle.y_b), (rectangle.x_r, rectangle.y_t), color, strength
        )

    return image
