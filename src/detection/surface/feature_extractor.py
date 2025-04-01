import itertools
import json
import os
from typing import Tuple
import cv2
import numpy as np
from skimage.feature import blob_dog
from math import sqrt
from scipy import ndimage
import itertools
from scipy.stats import multivariate_normal

import surface_utils

class FeatureExtractor():

    def __init__(self, resolution = "low"):
        self.layer_config = (-1, 0, 1, 2, 3, 4)
        self.channel_indices = (0, 1, 2)
        self.sigma = 0.5
        self.descriptor_scale = None
        self.n_kps = None

        if resolution == "low":
            self.descriptor_scale = 0.25
            self.n_kps = 1
        if resolution == "high":
            self.descriptor_scale = 0.666666  #input image resolution should be 1920x1080
            self.n_kps = 3


    def get_X_y(self, data: Tuple):
        """ 
        Load the data and extract features and labels for training.
        Args:
            data (tuple): images, ground truth bounding boxes in YOLO format and altitudes.
        Returns:
            tuple: A tuple containing:  
                - X (numpy.ndarray): Array of blob descriptors containing histograms and SIFT features.
                - y (numpy.ndarray): Array of labels for the detected blobs.   
        """

        images, gt_labels, altitudes = data

        X = []
        y = []

        for image, gt, altitude in zip(images, gt_labels, altitudes):
            X_temp, blobs = self.extract_features(image, altitude)
            if len(X_temp) == 0: continue
            y_temp = self.get_blob_labels(blobs, gt, image.shape[:2])

            X.append(X_temp)
            y.extend(y_temp)

        X = np.vstack(X)
        y = np.array(y)

        return X, y

    def extract_features(self, img_bgr, height):
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
        sigma_blobs, detection_scale, thresh = FeatureExtractor.sigma_function(height)
        blob_detection_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blob_detection_img = cv2.resize(blob_detection_img, (int(blob_detection_img.shape[1] * detection_scale),
                                                             int(blob_detection_img.shape[0] * detection_scale)))
        descriptor_img = cv2.resize(img_bgr, (int(round(img_bgr.shape[1] * self.descriptor_scale)),
                                          int(round(img_bgr.shape[0] * self.descriptor_scale))))
        descriptor_img_gray = cv2.cvtColor(descriptor_img, cv2.COLOR_BGR2GRAY)

        blobs, _ = FeatureExtractor.extract_blobs(blob_detection_img, min_sigma=sigma_blobs, max_sigma=sigma_blobs + 1,
                                  threshold=thresh)

        if len(blobs)< 1: return np.array([]), np.array([])

        blobs = np.array(blobs) / detection_scale
        #return blobs, []
        blobs_des = blobs * np.array([self.descriptor_scale, self.descriptor_scale, self.descriptor_scale])
        colorspaces = FeatureExtractor.get_colorspaces(descriptor_img)
        keypoints = []
        for blob in blobs_des:
            kps = FeatureExtractor.create_kps(blob, descriptor_img_gray, self.n_kps)
            keypoints += kps
        FeatureExtractor.compute_octave(keypoints, self.layer_config, self.descriptor_scale)
        channels = FeatureExtractor.get_channels(colorspaces, self.channel_indices)
        hists = FeatureExtractor.compute_histogram(keypoints, channels)
        sifts = FeatureExtractor.compute_SIFT(keypoints, channels, sigma=self.sigma)
        X = FeatureExtractor.get_blob_descriptors(hists, sifts, self.n_kps)
    
        return X, blobs
    

    @staticmethod
    def get_blob_labels(blobs, gt_labels, im_shape):
        """
        Get labels for the detected blobs based on the ground truth labels.
        Treats the blobs as positive if they are within the bounding box of the ground truth labels.

        Args:
            blobs (numpy.ndarray): Detected blobs with (x, y, radius) coordinates.
            gt_labels (list): Ground truth labels for the image in YOLO format.

        Returns:
            list: labels for the detected blobs.
        """
        h, w = im_shape

        labels = []
        for blob in blobs:
            x, y, r = blob
            positive = False
            for gt in gt_labels:
                x_gt = int(gt[0] * w)
                y_gt = int(gt[1] * h)
                w_gt = int(gt[2] * w)
                h_gt = int(gt[3] * h)

                if (x >= x_gt - w_gt / 2) and (x <= x_gt + w_gt / 2) and (y >= y_gt - h_gt / 2) and (y <= y_gt + h_gt / 2):
                    labels.append(1)
                    positive = True
                    break
            if not positive:
                labels.append(0)
                
        return labels
    
    @staticmethod
    def sigma_function(height):
        """
        Determines the sigma, scale, and threshold values based on the given height.
        Parameters:
            height (float): altitude from which the image was taken (in meters).
        Returns:
            tuple: A tuple containing:
                - sigma (int): The sigma value based on the height.
                - scale (float): A constant scale value (default is 0.15).
                - thresh (float): The threshold value based on the height.
        """
        scale = 0.15
        if height<=5:
            sigma = 6
            thresh = 0.15
        elif 5<height<=10:
            sigma = 4
            thresh = 0.1
        elif 10<height<=20:
            sigma=3
            thresh = 0.1
        # elif 20<height<=50:
        #     sigma = 2
        #     thresh = 0.05
        else:
            sigma = 2
            thresh = 0.05

        return sigma, scale, thresh
    
   

    @staticmethod
    def extract_blobs(img_gray, min_sigma=1, max_sigma=10, threshold=0.3, overlap=0.99):
        blobs_dog = blob_dog(img_gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold, overlap=overlap)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
        keypoints = []
        for blob in blobs_dog:
            y, x, r = tuple(blob)
            keypoints += [(x, y, r)]

        return keypoints, blobs_dog

   

    @staticmethod
    def get_colorspaces(img):
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        l,a,b = cv2.split(lab_img)
        # r,g,bb = cv2.split(rgb_img)
        # h,s,v = cv2.split(hsv_img)
        # y,cr,cb = cv2.split(ycrcb_img)
        # yy,u,vv = cv2.split(yuv_img)
        #return [l,a,b,r,g,bb,h,s,v,y,cr,cb,yy,u,vv]
        return [l, a, b]

    @staticmethod
    def gradient_m(i, j, picture):
        ft = (float(picture[i, j + 1]) - picture[i, j - 1]) ** 2
        st = (float(picture[i + 1, j]) - picture[i - 1, j]) ** 2

        return np.sqrt(ft + st)

    @staticmethod
    def gradient_theta(i, j, picture):
        # To avoid error division by zero
        eps = 1e-5

        ft = float(picture[i + 1, j]) - picture[i - 1, j]
        st = (float(picture[i, j + 1]) - picture[i, j - 1]) + eps

        ret = 180 + np.arctan2(ft, st) * 180 / np.pi
        return ret

    @staticmethod
    def create_histogram(i, j, picture, std, k_size):
        truncate = 4.0
        #kernel_size = 2 * int(std * truncate + .5) + 1
        kernel_size = k_size
        window = list(range(-kernel_size, kernel_size))
        # plt.imshow(picture[i-kernel_size:i+kernel_size, j-kernel_size:j+kernel_size])
        # plt.show()
        diag = set(itertools.permutations(window, 2))
        rooti, rootj = i, j
        theta_list = []

        gaussian = multivariate_normal(mean=[i, j], cov=1.5 * std)

        orient_hist = np.zeros([36, 1])

        for ii, jj in diag:
            x = rooti + ii
            y = rootj + jj
            if x - 1 < 0 or y - 1 < 0 or x + 1 > picture.shape[0] - 1 \
                    or y + 1 > picture.shape[1] - 1:
                continue

            # TODO: Warning the magnitude are really small
            magnitude = FeatureExtractor.gradient_m(x, y, picture)
            weight = magnitude * gaussian.pdf([x, y])

            orientation = FeatureExtractor.gradient_theta(x, y, picture)
            bins_orientation = np.clip(orientation // 10, 0, 35)
            orient_hist[int(bins_orientation)] += weight

        return orient_hist
    
    @staticmethod
    def transform_angle(alpha):
        if 0<alpha<180: return alpha + 180
        else: return alpha - 180

    @staticmethod
    def compute_angle(img_gray, keypoint, scale=0.5, std=sqrt(2)):
        h, w = img_gray.shape
        #img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur_shape = (int(h * scale), int(w * scale))
        blurred = ndimage.filters.gaussian_filter(cv2.resize(img_gray, (blur_shape[1], blur_shape[0])), std)
        kp_blur = cv2.KeyPoint(keypoint.pt[0] * scale, keypoint.pt[1] * scale, _size=keypoint.size * scale)
        orient_hist = FeatureExtractor.create_histogram(int(kp_blur.pt[1]), int(kp_blur.pt[0]), blurred, std, int(kp_blur.size / 2))
        sorted_hist = np.argsort(orient_hist, axis=0)[::-1]
        angle = FeatureExtractor.transform_angle(sorted_hist[0][0] * 10)
        keypoint.angle = angle

    @staticmethod
    def create_kps(blob, img_gray, n_kps):
        if n_kps==3:
            kpmid = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 0.5)
            FeatureExtractor.compute_angle(img_gray, kpmid)
            kplarge = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 1.0, _angle=kpmid.angle)
            kpsmall = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 0.25, _angle=kpmid.angle)
            kps = [kpsmall, kpmid, kplarge]
        elif n_kps==2:
            kpmid = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 0.65)
            FeatureExtractor.compute_angle(img_gray, kpmid)
            kpsmall = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 0.30, _angle=kpmid.angle)
            kps = [kpsmall, kpmid]
        else:
            kpmid = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 0.65)
            FeatureExtractor.compute_angle(img_gray, kpmid)
            kp = cv2.KeyPoint(blob[0], blob[1], _size=blob[2] * 2 * 1.0, _angle=kpmid.angle)
            kps = [kp]
        return kps

    @staticmethod
    def unpackSIFTOctave(kpt):
        """unpackSIFTOctave(kpt)->(octave,layer,scale)
        @created by Silencer at 2018.01.23 11:12:30 CST
        @brief Unpack Sift Keypoint by Silencer
        @param kpt: cv2.KeyPoint (of SIFT)
        """
        _octave = kpt.octave
        octave = _octave&0xFF
        layer  = (_octave>>8)&0xFF
        if octave>=128:
            octave |= -128
        if octave>=0:
            scale = float(1./(1<<octave))
        else:
            scale = float(1<<-octave)
        return (octave, layer, scale)

    @staticmethod
    def packSIFTOctave(octave, layer):
        if octave == -1:
            third_byte = int(255)
        else:
            third_byte = int(octave)
        second_byte = int(layer) << 8
        return second_byte | third_byte

    @staticmethod
    def compute_octave(kps, layers = (-1,0,1,2,3,4), scale=1.0):
        b1 = 20 * scale; b2 = 40*scale; b3 = 60*scale; b4 = 80*scale; b5=100*scale
        for kp in kps:
            if kp.size <= b1: kp.octave = FeatureExtractor.packSIFTOctave(layers[0],1)
            elif b1<kp.size<= b2: kp.octave = FeatureExtractor.packSIFTOctave(layers[1],1)
            elif b2 < kp.size <= b3: kp.octave = FeatureExtractor.packSIFTOctave(layers[2],1)
            elif b3 < kp.size <= b4: kp.octave = FeatureExtractor.packSIFTOctave(layers[3],1)
            elif b4 < kp.size <= b5: kp.octave = FeatureExtractor.packSIFTOctave(layers[4],1)
            else : kp.octave = FeatureExtractor.packSIFTOctave(layers[5],1)

    @staticmethod
    def get_channels(colorspace, indices):
        channels = []
        for i in indices:
            channels += [colorspace[i]]
        return channels

    @staticmethod
    def compute_histogram(kps, channels, scale=1.0, std=sqrt(2)):
        n_channels = len(channels)
        for i in range(n_channels):
            channels[i] = ndimage.filters.gaussian_filter(channels[i], std)
        histograms = []
        l = len(kps)
        for i in range(l):
            kp = kps[i]
            x = int(kp.pt[0] * scale)
            y = int(kp.pt[1] * scale)
            kernel = int(kp.size * scale / 2.)
            hist = []
            for channel in channels:
                c_patch = channel[y - kernel:y + kernel, x - kernel:x + kernel]
                c_hist, _ = np.histogram(c_patch.flatten(), bins=50, range=(0, 255))
                hist += c_hist.tolist()
            histograms += [hist]
        return histograms

    @staticmethod
    def compute_SIFT(kps, channels, sigma = 1.6):
        sift = cv2.xfeatures2d.SIFT_create(sigma=sigma)
        sifts = []
        l = len(kps)
        descriptors = []
        for c in channels:
            _, des = sift.compute(c, kps)
            descriptors += [des]
        for i in range(l):
            kp_descriptor = []
            for des in descriptors:
                kp_descriptor += des[i].tolist()
            sifts += [kp_descriptor]

        return sifts

    @staticmethod
    def get_blob_descriptors(histograms, sift_des, n_kps):
        size = len(histograms)
        descriptors = []
        for i in range(0, size, n_kps):
            hdes = []
            sdes = []
            for j in range(n_kps):
                hdes += histograms[i+j]
                sdes += sift_des[i+j]
            descriptors += [hdes + sdes]
        return np.array(descriptors)
