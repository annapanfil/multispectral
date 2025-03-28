import cv2
import numpy as np
import surface_utils
class SURFACE():

    def __init__(self, model, resolution = "low"):
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

    def forward_pass(self, img_bgr, height):
        clf, scaler, pca = self.model
        sigma_blobs, detection_scale, thresh = surface_utils.sigma_function1(height)
        blob_detection_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blob_detection_img = cv2.resize(blob_detection_img, (int(blob_detection_img.shape[1] * detection_scale),
                                                             int(blob_detection_img.shape[0] * detection_scale)))
        descriptor_img = cv2.resize(img_bgr, (int(round(img_bgr.shape[1] * self.descriptor_scale)),
                                          int(round(img_bgr.shape[0] * self.descriptor_scale))))
        descriptor_img_gray = cv2.cvtColor(descriptor_img, cv2.COLOR_BGR2GRAY)
        blobs, _ = surface_utils.extract_blobs3(blob_detection_img, min_sigma=sigma_blobs, max_sigma=sigma_blobs + 1,
                                  threshold=thresh)

        if len(blobs)< 1: return np.array([]), np.array([]), np.array([])

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

