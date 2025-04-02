import pickle
import json

from matplotlib import pyplot as plt
import numpy as np

from data_handler import DataHandler
from evaluator import Evaluator
from surface import SURFACE
from surface_utils import load_model

####################################################
models_path = "../../../models"
ds_path = "/home/anna/Datasets/SURFACE/full_ds"

model = "sift_100%1920.pickle"
# model = None
resolution = (1920, 1080) #(1920, 1080) # #(480, 270) #(1280, 720)  # # 
debug = False

# resolution_str = "high" if resolution == (1920, 1080) else "low"
resolution_str = "high"
####################################################


dataHandler = DataHandler()
evaluator = Evaluator(debug=debug)

if model is not None:
    # load the model
    model = load_model(f"{models_path}/{model}")
    surface = SURFACE(model, resolution_str, debug)
else:
    # Load the parameters from the JSON file
    with open(f'{models_path}/sift_100%1920_params.json', 'r') as f: # TODO: replace with the correct moel
        params = json.load(f)

    params["pca"]["n_components"] = None # to little n_samples without augmentation to have 200

    # Train the model
    surface = SURFACE(resolution=resolution_str, debug=debug)
    train_data = dataHandler.load_and_resize(ds_path, "train", resolution)
    surface.train_model(params, train_data)

    # Save the model
    # pickle.dump(surface.model, open(f"{models_path}/my_model1920.pickle", "wb"))


# Evaluate the model
val_data = dataHandler.load_and_resize(ds_path, "val", resolution)

# detections, scores, proposals = surface.get_detections(val_data[0], val_data[2],  0.5)

all_detections, all_scores, all_blobs = surface.get_detections(val_data[0], val_data[2],  0)

for i in range(len(all_detections)):
    print(f"{len(all_detections[i])} litter out of {len(all_blobs[i])} blobs. Min confidence: {min(all_scores[i]).round(2) if all_scores[i].size > 0 else '-'}")

scores = evaluator.print_metrics(val_data[0], val_data[1], all_detections, all_scores)

acc = surface.get_accuracy(val_data)
print("Accuracy:", acc.round(4))