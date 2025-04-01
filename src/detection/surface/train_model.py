import pickle
import json

from surface import SURFACE
from evaluator import Evaluator

models_path = "../../../models"
ds_path = "/home/anna/Datasets/SURFACE/full_ds"

# Load the parameters from the JSON file
with open(f'{models_path}/sift_100%1920_params.json', 'r') as f:
    params = json.load(f)

params["pca"]["n_components"] = None

surface = SURFACE(resolution="low")
evaluator = Evaluator(debug=True)

surface.train_model(params, ds_path)

# pickle.dump(surface.model, open(f"{models_path}/my_model1920.pickle", "wb"))

detections, scores, proposals = surface.show_detections(ds_path, split="val")

scores = evaluator.print_metrics(ds_path, "val", detections, scores)