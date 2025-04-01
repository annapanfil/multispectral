import surface_utils
import json

model_path = "../../../models/"

path = f"{model_path}sift_100%1920.pickle"
model = surface_utils.load_model(path)
print(model)

with open(f'{model_path}sift_100%1920_params.json', 'w') as f:
    json.dump({
        "svc": model[0].get_params(),
        "scaler": model[1].get_params(),
        "pca": model[2].get_params(),
    }, f)

print("Model parameters saved to JSON.")