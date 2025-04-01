from typing import Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import surface_utils
from feature_extractor import FeatureExtractor

class SURFACE():
    def __init__(self, model=None, resolution = "low", debug=False):
        self.featureExtractor = FeatureExtractor(resolution=resolution)
        self.detection_confidence_thres = 0.25
        self.model = model
        self.debug = debug

    def get_detections(self, images, altitudes):
        """
        Visualize the detections on the images in the specified data path and split.
        
        Args:
           images (list): List of images to visualize detections on.
           altitudes (list): List of altitudes corresponding to each image.
        Returns:
            tuple: A tuple containing three lists - detected blobs, confidence scores, and all blobs.
        """
        all_detections, all_scores, all_proposals = [], [], []
        for image, alt in zip(images, altitudes):
            detections, scores, proposals = self.forward_pass(image, alt)
            all_detections.append(detections)
            all_scores.append(scores)
            all_proposals.append(proposals)
        
        if self.debug:
            surface_utils.show_all_detections(images, all_detections)

        return all_detections, all_scores, all_proposals


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
        
        X, blobs = self.featureExtractor.extract_features(img_bgr, height)
        if len(blobs)< 1: return np.array([]), np.array([]), np.array([])
        X = scaler.transform(X)
        X = pca.transform(X)

        scores = clf.predict_proba(X)
        distances = clf.decision_function(X)

        positive = np.where(distances > 0)[0]
        confidence = scores[positive][:, 1]
        positive_blobs = blobs[positive]
        keep_indices = SURFACE.filter_blobs_indices(positive_blobs, img_bgr)
        if len(keep_indices) > 0:
            positive_blobs = positive_blobs[keep_indices]
            confidence = confidence[keep_indices]
        else:
            return np.array([]), np.array([]), np.array([])
        keep = np.where(confidence >= self.detection_confidence_thres)[0]
        return positive_blobs[keep] * np.array([1,1,1.5]), confidence[keep], blobs
    

    def filter_blobs_indices(blobs, img):
        """
        Filters the indices of blobs that are within the bounds of the image.

        Args:
            blobs (list): A list of blob objects to be filtered.
            img (numpy.ndarray): The image associated with the blobs.

        Returns:
            numpy.ndarray: An array of indices corresponding to the blobs that satisfy the validation condition.
        """

        keep_indices = []
        for i in range(len(blobs)):
            blob = blobs[i]
            if SURFACE.is_blob_inside_img(blob, img):
                keep_indices += [i]
        return np.array(keep_indices)
    
    
    @staticmethod
    def is_blob_inside_img(blob, img):
        """
        Checks if a circular blob is within the bounds of a given image.

        Args:
            blob (tuple): A tuple (x, y, r) representing the blob's center coordinates (x, y) 
                        and its radius r.
            img (numpy.ndarray): The image (height, width, channels).

        Returns:
            bool: True if the blob is entirely within the image boundaries, False otherwise.
        """
        h,w,c = img.shape
        x = blob[0]
        y = blob[1]
        r = blob[2]
        return  not(x - r < 0 or x + r >= w or y - r < 0 or y + r >= h)


    def train_model(self, params: dict, data: Tuple):
        """
        Train scaler, PCA and SVC model on the provided data.

        Args:
        params (dict): Parameters for scaler, pca and scv. Each set of parameters is a dict.
        data (tuple): images, ground truth bounding boxes in YOLO format and altitudes.
        """
        
        X, y = self.featureExtractor.get_X_y(data)

        scaler = StandardScaler(**params["scaler"])
        pca = PCA(**params["pca"])
        clf = SVC(**params["svc"])

        X = scaler.fit_transform(X)
        X = pca.fit_transform(X)

        clf.fit(X, y.T)

        self.model = (clf, scaler, pca)
        


