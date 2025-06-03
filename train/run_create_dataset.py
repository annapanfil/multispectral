"""Create datasets for all the developed formulas and send them to the remote host"""

import subprocess
import mlflow
import pandas as pd

def run_job(job):
    if None not in job:
        print(" ".join(job))
    else: print(job)
    subprocess.run(job, check=True)

if __name__ == "__main__":
    # # Fetch the experiments
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # experiment_id = "859627527365844361"
    # df = mlflow.search_runs(experiment_ids=[experiment_id])

    # GET FORMULAS
    field_names = ["tags.mlflow.runName", "params.dataset", "params.final_formula", "tags.discard"]
    # selected_columns = df[field_names]

    POOL_DS = ["ghost-net", "bags", "black-bed", "green-net", "pile"]
    SEA_DS = ["mandrac", "mandrac2", "mandrac3"]

    baseline_rows = pd.DataFrame([
            ["rndwi", None, "N # G", "false"],
            ["meanre", None, "(4#1) + (4#0)", "false"],
            ["RGB", None, None, "false"]
        ], columns=field_names)

    selected_columns = pd.DataFrame([
            ["form1", "pool", "4 # 1", "false"],
            ["form2", "pool", "(4 - ((1 * 0) # (4 * 0)))", "false"],
            ["form3", "pool", "(((4 + (2 * 3)) + 4) # 1)", "false"],
            ["form4", "all", "(4 # (4 + 1))", "false"],
            ["form5", "all", "(((0 # 4) - 4) - (4 # 1))", "false"],
            ["form6", "sea", "((((0 + 1) + 0) + 2) # (1 / 3))", "false"],
            ["form7", "sea", "((0 * 3) # 0)", "false"],
            ["form8", "sea", "(3 - (4 - 3))", "false"]
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
            # run_job(["python3", "-m", "dataset_creation.create_dataset", "-n", "RGB_whole_random", 
            #          "-c", "R", "-c", "G", "-c", "B"])
            # run_job(["python3", "-m", "dataset_creation.create_dataset", "-e", "mandrac", "-n", "RGB_pool_random", 
            #          "-c", "R", "-c", "G", "-c", "B"])
            run_job(["python3", "-m", "dataset_creation.create_dataset", "-n", "RGB_tr-whole-test-val-sea_random", 
                     "-c", "R", "-c", "G", "-c", "B", "-m", "13"]
                      + sum([["-t", d] for d in SEA_DS], []))
            run_job(["python3", "-m", "dataset_creation.create_dataset", "-n", "RGB_sea_random",
                        "-c", "R", "-c", "G", "-c", "B", "-m", "13"]
                        + sum([["-e", d] for d in POOL_DS], []))
            

            continue

        if formula is None:
            print(name, "is not finished")
            continue

        # train and test on whole
        # run_job(["python3", "-m", "dataset_creation.create_dataset", 
        #     "-n", f"{ds if ds else ''}{'-' if ds else ''}{name.replace(' ', '-')}_whole_random", 
        #     "-c", formula])
    
    
        # train on whole, test on pool
        # run_job(["python3", "-m", "dataset_creation.create_dataset", 
        #     "-e", "mandrac",
        #     "-n", f"{ds if ds else ''}{'-' if ds else ''}{name.replace(' ', '-')}_pool-3-channels_random", 
        #     "-c", "E", "-c", "G", "-c", formula])

        # train on whole, val and test on sea
        run_job(["python3", "-m", "dataset_creation.create_dataset",
            "-n", f"{ds if ds else ''}{'-' if ds else ''}{name.replace(' ', '-')}_tr-whole-test-val-sea_random",
            "-m", "13", "-c", "N", "-c", "G", "-c", formula] +
            sum([["-t", d] for d in SEA_DS], [])
        )

        # train and test on sea only
        run_job(["python3", "-m", "dataset_creation.create_dataset",
            "-n", f"{ds if ds else ''}{'-' if ds else ''}{name.replace(' ', '-')}_sea_random",
            "-m", "13", "-c", "N", "-c", "G", "-c", formula] +
            sum([["-e", d] for d in POOL_DS], [])
            )

    # train on whole, val and test on sea, only channel 3 (NIR) on all the inputs
    run_job(["python3", "-m", "dataset_creation.create_dataset",
                "-n", f"{ds if ds else ''}{'-' if ds else ''}NIR_tr-whole-test-val-sea_random",
                "-m", "13", "-c", "N", "-c", "N", "-c", "N"] +
                sum([["-t", d] for d in SEA_DS], [])
                )
    

    # train and test on sea only, only channel 3 (NIR) on all the inputs
    run_job(["python3", "-m", "dataset_creation.create_dataset",
            "-n", f"{ds if ds else ''}{'-' if ds else ''}NIR_sea_random",
            "-m", "13", "-c", "N", "-c", "N", "-c", "N"] +
            sum([["-e", d] for d in POOL_DS], [])
            )
    
    # train on whole, val and test on sea, only channel 3 (NIR) on all the inputs

    # MERGE INDEX AND RGB DATASETS
    # train on whole idx + RGB, test on whole
    # run_job(["python3", "-m", "dataset_creation.merge_index_and_RGB_dataset", 
    #          "-d", "whole_random", 
    #          "-i", "ghost-net-"])
    
    # train on pool idx + RGB, test on pool
    # run_job(["python3", "-m", "dataset_creation.merge_index_and_RGB_dataset", 
    #          "-d", "pool_random", 
    #          "-i", "pool-"])
    
    # SEND TO REMOTE
    # print("Create zip and send it to remote host")
    # dataset_dir = f"{DATASET_BASE_PATH}/created"
    # remote_host = "lariat@10.2.116.180"
    # remote_path = f"/home/lariat/code/anna/datasets"
    
    # subprocess.run(["zip", "-r", "datasets.zip", "."], cwd=dataset_dir, check=True)
    # subprocess.run(["scp", "datasets.zip", f"{remote_host}:{remote_path}"], cwd=dataset_dir, check=True)