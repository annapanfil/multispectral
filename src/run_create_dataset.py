"""Create datasets for all the developed formulas and send them to the remote host"""

import sys
sys.path.append("/home/anna/code/multispectral/src/detection")

import subprocess
import mlflow
import pandas as pd

def run_job(job):
    if None not in job:
        print(" ".join(job))
    else: print(job)
    subprocess.run(job, check=True)

if __name__ == "__main__":
    # Fetch the experiments
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_id = "859627527365844361"
    df = mlflow.search_runs(experiment_ids=[experiment_id])

    # GET FORMULAS
    field_names = ["tags.mlflow.runName", "params.dataset", "params.final_formula", "tags.discard"]
    # selected_columns = df[field_names]

    baseline_rows = pd.DataFrame([
            ["rndwi", None, "N # G", "false"],
            ["meanre", None, "(4#1) + (4#0)", "false"],
            # ["RGB", None, None, "false"]
        ],columns=field_names)


    selected_columns = pd.DataFrame([
            ["form1", "pool", "4 # 1", "false"],
            ["form2", "pool", "(4 - ((1 * 0) # (4 * 0)))", "false"],
            ["form3", "pool", "(((4 + (2 * 3)) + 4) # 1)", "false"],
            ["form4", "all", "(4 # (4 + 1))", "false"],
            ["form5", "all", "(((0 # 4) - 4) - (4 # 1))", "false"]
        ], columns=field_names)

    selected_columns = pd.concat([selected_columns, baseline_rows], ignore_index=True)

    pd.set_option('display.max_rows', None)  # Show all rows
    print(selected_columns)

    print(selected_columns[selected_columns["tags.discard"] != "true"])

    # CREATE BASE DATASETS
    for _, row in  selected_columns.iterrows():
        name, ds, formula, discard = row
        if discard == "true":
            print(name, "is discarded")
            continue

        if name == "RGB":
            run_job(["python3", "-m", "detection.create_RGB_dataset", "-n", "RGB_whole_random"])
            run_job(["python3", "-m", "detection.create_RGB_dataset", "-e", "mandrac", "-n", "RGB_pool_random"])
            continue

        if formula is None:
            print(name, "is not finished")
            continue

        # train and test on whole
        # run_job(["python3", "-m", "detection.create_dataset", 
        #     "-n", f"{ds if ds else ''}{'-' if ds else ''}{name.replace(' ', '-')}_whole_random", 
        #     "-c", formula])
        
        # train on whole, test on pool
        run_job(["python3", "-m", "detection.create_dataset", 
            "-e", "mandrac",
            "-n", f"{ds if ds else ''}{'-' if ds else ''}{name.replace(' ', '-')}_pool-3-channels_random", 
            "-c", "E", "-c", "G", "-c", formula])


    # MERGE INDEX AND RGB DATASETS
    # train on whole idx + RGB, test on whole
    # run_job(["python3", "-m", "detection.merge_index_and_RGB_dataset", 
    #          "-d", "whole_random", 
    #          "-i", "ghost-net-"])
    
    # train on pool idx + RGB, test on pool
    # run_job(["python3", "-m", "detection.merge_index_and_RGB_dataset", 
    #          "-d", "pool_random", 
    #          "-i", "pool-"])
    
    # SEND TO REMOTE
    print("Create zip and send it to remote host")
    dataset_dir = "/home/anna/Datasets/created"
    remote_host = "lariat@10.2.116.180"
    remote_path = f"/home/lariat/code/anna/datasets"
    
    subprocess.run(["zip", "-r", "datasets.zip", "."], cwd=dataset_dir, check=True)
    subprocess.run(["scp", "datasets.zip", f"{remote_host}:{remote_path}"], cwd=dataset_dir, check=True)