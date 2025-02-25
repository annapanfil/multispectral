#!/home/anna/miniconda3/envs/micasense/bin/python3

import os
import sys
"""Get the bounding boxes from all channels and merge them into one file, assigned to each channel."""

def merge_bbs_from_all_channels(path:str, common_fn:str, out_path:str):
    files = [file for file in os.listdir(path) if file.startswith(common_fn)]
    bbs = set()
    for file in files:
        with open(os.path.join(path.replace("images", "labels"), file.split(".")[0]+".txt"), "r") as f:
            for line in f:
                bbs.add(line.replace("\n", ""))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for file in files:
        with open(os.path.join(out_path, file.split(".")[0]+".txt"), "w") as f:
            for line in bbs:
                f.write(line)
                f.write("\n")

    print(f"Merged {len(bbs)} bboxes in {len(files)} files and saved in {out_path}/{common_fn}_x.txt")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_labels.py [dataset_name]")
        sys.exit(1)

    path = f"/home/anna/Datasets/annotated/{sys.argv[1]}"
    common_fns = set("_".join(file.split("_")[:-1]) for file in os.listdir(f"{path}/images/train/"))
    for common_fn in common_fns:
        merge_bbs_from_all_channels(f"{path}/images/train/", common_fn, f"{path}/labels/train/")