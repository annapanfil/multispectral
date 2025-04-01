import cv2
import os
import json

import numpy as np

class DataHandler():

    def load_and_resize(self, data_path: str, split: str, resolution: tuple):
        """
        Load and resize images, ground truth bounding boxes and altitudes from the specified directory.

        Args:
            data_path (str): Path to the directory containing images and ground truth labels. With directories images, jsons and labels.
            Images are RGB in JPG format. Labels are in YOLO format.
            resolution (tuple): Desired resolution for resizing images.

        Returns:
            tuple: A tuple containing three lists - resized images, ground truth labels and altitudes.
        """
        images, gt_labels, altitudes = self.load_data(data_path, split)
        images = self.resize_images(images, resolution)

        return images, gt_labels, altitudes

    
    def load_data(self, data_path: str, split: str):
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


    def resize_images(self, images, resolution):
        return [cv2.resize(im, resolution, interpolation=cv2.INTER_LINEAR) for im in images]    
    