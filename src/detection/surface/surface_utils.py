from collections import namedtuple
import json
import os
import pickle
import cv2
from matplotlib import pyplot as plt
import numpy as np

Circle = namedtuple("Circle", ["x", "y", "r"])
Rectangle = namedtuple("Rectangle", ["x_l", "y_b", "x_r", "y_t"])

def load_model(path):
    model = pickle.load(open(path, "rb"), encoding='latin1')
    print("Model loaded from", path)

    return model

def show_blobs(image, blobs, classes = None):
    """
    Visualize the detected blobs on the image.
    """
    if classes is None:
        classes = [1] * len(blobs)

    colors = [(255, 0, 0), (0, 255, 0)]

    for i, blob in enumerate(blobs):
        cv2.circle(image, 
                   (int(blob[0]), int(blob[1])), radius = int(blob[2]),
                   color=colors[classes[i]], thickness=3)
    plt.imshow(image)
    plt.show()

def show_all_detections(images, blobs):
    """
    Visualize the detected blobs on all images.
    """
    new_images = []
    for i, image in enumerate(images):
        new_image = image.copy()
        for blob in blobs[i]:
            cv2.circle(new_image, 
                       (int(blob[0]), int(blob[1])), radius = int(blob[2]),
                       color=(0, 0, 255), thickness=3)
        new_images.append(new_image)
    
    show_images(new_images, "Detected blobs on all images")


def show_images(images, title):
    """
    Display a grid of images.
    """

    aspect_ratio = 4 / 3
    cols = int(np.ceil(np.sqrt(len(images) * aspect_ratio))) 
    rows = int(np.ceil(len(images) / cols))

    _, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
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