import pickle
import json

from surface import SURFACE

models_path = "../../../models"

# Load the parameters from the JSON file
with open(f'{models_path}/sift_100%480_params.json', 'r') as f:
    params = json.load(f)

surface = SURFACE(resolution="low")
surface.train_model(params, "/home/anna/Datasets/SURFACE/full_ds")

pickle.dump(surface.model, open("{models_path}/my_model480.pickle", "wb"))

