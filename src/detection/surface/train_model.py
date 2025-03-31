import pickle
import json

from Surface import SURFACE

models_path = "../../../models"
ds_path = "/home/anna/Datasets/SURFACE/full_ds"

# Load the parameters from the JSON file
with open(f'{models_path}/sift_100%480_params.json', 'r') as f:
    params = json.load(f)

params["pca"]["n_components"] = None

surface = SURFACE(resolution="low")
surface.train_model(params, ds_path)

pickle.dump(surface.model, open(f"{models_path}/my_model480.pickle", "wb"))

surface.show_detections(ds_path, split="val")