"""Get the multiclass dataset annotations and filter them to only one class, saving to new directory."""

import os
from src.processing.consts import DATASET_BASE_PATH

source_dir = f"{DATASET_BASE_PATH}/created/hamburg_mapping-form8-val/two_classes"
output_dir = f"{DATASET_BASE_PATH}/created/hamburg_mapping-form8-val/pile_only"
class_no = 1

os.makedirs(output_dir, exist_ok=False)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=False)
    source = os.path.join(source_dir, split)
    output = os.path.join(output_dir, split)
    for fname in os.listdir(source):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(source, fname), "r") as f:
            with open(os.path.join(output, fname), "w") as out_f:
                for line in f:
                    if line.startswith(str(class_no)):
                        out_f.write("0"+ line[1:])
            