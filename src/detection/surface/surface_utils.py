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

def load_data(data_path: str, split: str):
    """
    Load images, ground truth bounding boxes and altitudes from the specified directory.

    Args:
        data_path (str): Path to the directory containing images and ground truth labels. With directories images, jsons and labels.
        Images are RGB in JPG format. Labels are in YOLO format.

    Returns:
        tuple: A tuple containing three lists - images, ground truth labels and altitudes.
    """

    images = [cv2.cvtColor(cv2.imread(f"{data_path}/images/{split}/{img}"), cv2.COLOR_BGR2RGB) for img in os.listdir(f"{data_path}/images/{split}/")]
    gt_labels = []
    altitudes = []
    for img_name in os.listdir(f"{data_path}/images/{split}"):
        with open(f"{data_path}/labels/{split}/{img_name.replace('.JPG', '.txt')}") as f:
            gt_labels.append([tuple(map(float, line.split()[1:])) for line in f.readlines()])
        with open(f"{data_path}/jsons/{split}/{img_name.replace('.JPG', '.json')}") as f:
            data = json.load(f)
            altitude = int(data["imagePath"].split('.')[0].split('_')[-1])
            altitudes.append(altitude)

    return images, gt_labels, altitudes

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
    
    show_images(new_images)


def show_images(images):
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
    plt.show()