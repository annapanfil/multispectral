from Surface import SURFACE

path = "../../../models/sift_100%480.pickle"
surface_model = SURFACE("low")
surface_model.load_model(path)
surface_model.detection_confidence_thres = 0.5
rootdir = "/home/anna/Datasets/SURFACE/full_ds"

surface_model.show_detections(rootdir, split="val")