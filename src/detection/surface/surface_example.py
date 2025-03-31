from Surface import SURFACE
from surface_utils import load_model

path = "../../../models/sift_100%1920.pickle"
model = load_model(path)
surface_model = SURFACE(model, "high")
surface_model.detection_confidence_thres = 0.5
rootdir = "/home/anna/Datasets/SURFACE/full_ds"

surface_model.show_detections(rootdir, split="val")