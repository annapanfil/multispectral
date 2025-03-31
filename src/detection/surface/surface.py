import json
import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import surface_utils

class SURFACE():
    def __init__(self, model=None, resolution = "low"):
        self.layer_config = (-1, 0, 1, 2, 3, 4)
        self.sigma = 0.5
        self.channel_indices = (0,1,2)
        self.model = model
        self.thres = 0.25
        self.descriptor_scale = None
        self.n_kps = None

        if resolution == "low":
            self.descriptor_scale = 0.25
            self.n_kps = 1
        if resolution == "high":
            self.descriptor_scale = 0.666666  #input image resolution should be 1920x1080
            self.n_kps = 3

    def feature_extraction(self, img_bgr, height):
        """
        Performs blob detection and calculates histograms and SIFT descriptors for the blobs.
        Args:
            img_bgr (numpy.ndarray): Input image in BGR format.
            height (float): altitude of the image in meters.
        Returns:
            tuple:
                - X (numpy.ndarray): Array of blob descriptors containing histograms and SIFT features.
                - blobs (numpy.ndarray): Array of detected blobs with (x, y, radius) coordinates.
        Notes:
           The descriptors are computed based on the specified number of keypoints (`self.n_kps`),
            layer configuration (`self.layer_config`), and descriptor scale (`self.descriptor_scale`).
        """
        sigma_blobs, detection_scale, thresh = surface_utils.sigma_function(height)
        blob_detection_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blob_detection_img = cv2.resize(blob_detection_img, (int(blob_detection_img.shape[1] * detection_scale),
                                                             int(blob_detection_img.shape[0] * detection_scale)))
        descriptor_img = cv2.resize(img_bgr, (int(round(img_bgr.shape[1] * self.descriptor_scale)),
                                          int(round(img_bgr.shape[0] * self.descriptor_scale))))
        descriptor_img_gray = cv2.cvtColor(descriptor_img, cv2.COLOR_BGR2GRAY)

        blobs, _ = surface_utils.extract_blobs(blob_detection_img, min_sigma=sigma_blobs, max_sigma=sigma_blobs + 1,
                                  threshold=thresh)

        if len(blobs)< 1: return np.array([]), np.array([])

        blobs = np.array(blobs) / detection_scale
        #return blobs, []
        blobs_des = blobs * np.array([self.descriptor_scale, self.descriptor_scale, self.descriptor_scale])
        colorspaces = surface_utils.get_colorspaces(descriptor_img)
        keypoints = []
        for blob in blobs_des:
            kps = surface_utils.create_kps(blob, descriptor_img_gray, self.n_kps)
            keypoints += kps
        surface_utils.compute_octave(keypoints, self.layer_config, self.descriptor_scale)
        channels = surface_utils.get_channels(colorspaces, self.channel_indices)
        hists = surface_utils.compute_histogram(keypoints, channels)
        sifts = surface_utils.compute_SIFT(keypoints, channels, sigma=self.sigma)
        X = surface_utils.get_blob_descriptors(hists, sifts, self.n_kps)
    
        return X, blobs

    def forward_pass(self, img_bgr, height):
        """
        Predict the litter location in the image using the trained model.
        
        Args:
            img_bgr (numpy.ndarray): Input image in BGR format.
            height (float): altitude of the image in meters.
            
        Returns:
            tuple:
                - positive_blobs (numpy.ndarray): Detected blobs that are classified as litter.
                - confidence (numpy.ndarray): Confidence scores for the detected blobs.
                - blobs (numpy.ndarray): All detected blobs in the image.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Please train the model before using it.")
        
        clf, scaler, pca = self.model
        
        X, blobs = self.feature_extraction(img_bgr, height)
        if len(blobs)< 1: return np.array([]), np.array([]), np.array([])
        X = scaler.transform(X)
        X = pca.transform(X)

        scores = clf.predict_proba(X)
        distances = clf.decision_function(X)

        positive = np.where(distances > 0)[0]
        confidence = scores[positive][:, 1]
        positive_blobs = blobs[positive]
        keep_indices = surface_utils.filter_blobs_indices(positive_blobs, img_bgr)
        if len(keep_indices) > 0:
            positive_blobs = positive_blobs[keep_indices]
            confidence = confidence[keep_indices]
        else:
            return np.array([]), np.array([]), np.array([])
        keep = np.where(confidence >= self.thres)[0]
        return positive_blobs[keep] * np.array([1,1,1.5]), confidence[keep], blobs


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

    def get_labels(self, blobs, gt_labels):
        """
        Get labels for the detected blobs based on the ground truth labels.
        Treats the blobs as positive if they are within the bounding box of the ground truth labels.

        Args:
            blobs (numpy.ndarray): Detected blobs with (x, y, radius) coordinates.
            gt_labels (list): Ground truth labels for the image in YOLO format.

        Returns:
            np.ndarray: Array of labels for the detected blobs.
        """

        labels = []
        for blob in blobs:
            x, y, r = blob
            positive = False
            for gt in gt_labels:
                x_gt, y_gt, w_gt, h_gt = gt
                if (x >= x_gt - w_gt / 2) and (x <= x_gt + w_gt / 2) and (y >= y_gt - h_gt / 2) and (y <= y_gt + h_gt / 2):
                    labels.append(1)
                    break
            if not positive:
                labels.append(0)
                
        return np.array(labels).reshape(-1, 1) # make vertical

    def train_model(self, params: dict, data_path: str):
        """
        Train scaler, PCA and SVC model on the provided data.

        Args:
        params (dict): Parameters for scaler, pca and scv. Each set of parameters is a dict.
        data_path (str): Path to the training data.
        """
        images, gt_labels, altitudes = self.load_data(data_path, "train")

        X = []
        y = np.empty((0,1), dtype=np.uint8)

        for image, gt, altitude in zip(images, gt_labels, altitudes):
            X_temp, blobs = self.feature_extraction(image, altitude)
            if len(X_temp) == 0: continue
            y_temp= self.get_labels(blobs, gt)
            X.append(X_temp)
            y = np.vstack((y, y_temp))

        X = np.vstack(X)
        breakpoint()

        scaler = StandardScaler(params["scaler"])
        pca = PCA(params["pca"])
        clf = SVC(params["svc"])

        X = scaler.fit_transform(X)
        X = pca.fit_transform(X)

        clf.fit(X, y)

        self.model = (clf, scaler, pca)
        


