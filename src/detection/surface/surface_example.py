import cv2
import numpy as np
import pickle
import surface_utils
from surface import SURFACE
import os
import matplotlib.pyplot as plt

path = "../../../models/sift_100%480.pickle"
model = surface_utils.load_model(path)
surface_model = SURFACE(model, "low")
surface_model.thres = 0.5
rootdir = "/home/anna/Datasets/SURFACE/bistrina_imgs/"

for filename in os.listdir(rootdir):
	if filename.endswith("json"): continue
	img_path = rootdir + filename
	image_bgr = cv2.imread(rootdir + filename)
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	height = int(filename.split(".")[0].split("_")[-1])

	detections, scores, proposals = surface_model.forward_pass(image_bgr, height)
	
	for i in range(len(detections)):
		det = detections[i]
		cv2.circle(image_rgb, (int(det[0]), int(det[1])), radius = int(det[2]), color=(0,255,0), thickness=3)
	plt.imshow(image_rgb)
	plt.show()