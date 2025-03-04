"""Create datasets for all the developed formulas and send them to the remote host"""

import sys
sys.path.append("/home/anna/code/multispectral/src/detection")

import subprocess
import mlflow
import pandas as pd

def run_job(job):
    print(" ".join(job))
    subprocess.run(job, check=True)

if __name__ == "__main__":
    # Fetch the experiments
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_id = "859627527365844361"
    df = mlflow.search_runs(experiment_ids=[experiment_id])

    # GET FORMULAS
    field_names = ["tags.mlflow.runName", "params.dataset", "params.final_formula"]
    selected_columns = df[field_names]

    baseline_rows = pd.DataFrame([["rndwi", None, "N # G"],
                    ["meanre", None, "(4#1) + (4#0)"],
                    ["RGB", None, None]], columns=field_names)

    selected_columns = pd.concat([selected_columns, baseline_rows], ignore_index=True)

    print(selected_columns)

    # CREATE BASE DATASETS
    for _, row in  selected_columns.iterrows():
        name, ds, formula = row
        if formula is None:
            print(name, "is not finished")
            continue

        # train and test on whole
        run_job(["python3", "-m", "detection.create_dataset", 
            "-n", f"{ds}{'-' if ds else ''}{name.replace(' ', '-')}_whole_random", 
            "-f", formula])
        
        # train on whole, test on pool
        run_job(["python3", "-m", "detection.create_dataset", 
            "-e", "mandrac",
            "-n", f"{ds}{'-' if ds else ''}{name.replace(' ', '-')}_whole_random", 
            "-f", formula])

    # MERGE INDEX AND RGB DATASETS
    # train on whole idx + RGB, test on whole
    run_job(["python3", "-m", "detection.merge_index_and_RGB_dataset", 
             "-d", "whole_random", 
             "-i", "ghost-net-"])
    
    # train on pool idx + RGB, test on pool
    run_job(["python3", "-m", "detection.merge_index_and_RGB_dataset", 
             "-d", "pool_random", 
             "-i", "pool-"])
    
    # SEND TO REMOTE
    print("Create zip and send it to remote host")
    dataset_dir = "/home/anna/Datasets/created"
    remote_host = "lariat@10.2.116.180"
    remote_path = f"/home/lariat/code/anna/datasets"
    
    subprocess.run(["zip", "-r", "datasets.zip", "*"], cwd=dataset_dir, check=True)
    subprocess.run(["scp", "datasets.zip", f"{remote_host}:{remote_path}"], cwd=dataset_dir, check=True)