import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List, Tuple

from src.shapes import Rectangle

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

def draw_yolo_boxes(image_path, label_path, class_names, filter_class=-1, palette=[tuple(int(c * 255) for c in color) for color in plt.cm.tab20.colors], display=True):
    """ Draw bounding boxes in yolo format on image"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]

    annots = []
    # Open and parse YOLO label file
    with open(label_path, "r") as f:
        for line in f:
            class_id, center_x, center_y, width, height = map(float, line.split())
            if filter_class != -1 and class_id != filter_class:
                continue

            class_id = int(class_id)

            # Convert YOLO format to pixel values
            x1 = int((center_x - (width / 2)) * image_width)
            x2 = int((center_x + (width / 2)) * image_width)
            y1 = int((center_y - (height / 2)) * image_height)
            y2 = int((center_y + (height / 2)) * image_height)
                    
            annots.append(Rectangle(x1, y1, x2, y2, class_names[class_id]))
            # Draw bounding box and label
            color = palette[class_id%20] if type(palette) is list else palette

            cv2.rectangle(image, (x1, y1), (x2,y2), color, 5)
            # cv2.putText(image, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    if display:
        print("image:", image_path)
        plt.imshow(image)
        plt.show()

    return image, annots