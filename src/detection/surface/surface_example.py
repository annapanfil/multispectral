from evaluator import Evaluator
from surface import SURFACE
from surface_utils import load_model

path = "../../../models/sift_100%1920.pickle"
model = load_model(path)
surface_model = SURFACE(model, "high")
evaluator = Evaluator(debug=True)

surface_model.detection_confidence_thres = 0.5
ds_path = "/home/anna/Datasets/SURFACE/full_ds"

detections, scores, _ = surface_model.show_detections(ds_path, split="val")
scores = evaluator.print_metrics(ds_path, "val", detections, scores)