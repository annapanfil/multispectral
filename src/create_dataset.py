import cv2
import os
import pandas as pd
import shutil
import yaml

from detection import merge_rectangles
from shapes import Rectangle


def read_all_datasets(dataset_path = "/home/anna/Datasets/annotated", 
                      excluded_litter = ["grass_bio_brown", "flake_PE_black", "flake_PE_transparent"],
                      channel="RGB.png"):
    subdatasets = next(os.walk(dataset_path))[1]
    subdatasets.remove("warp_matrices")

    data = []
    columns = ["subdataset", "image_path", "annot_path", "annots"] 

    for subdataset in subdatasets:
        subdataset_path = os.path.join(dataset_path, subdataset)
        for image in os.listdir(os.path.join(subdataset_path, "images", "train")):
            if image.endswith(channel): # only one image per group
                image_path = os.path.join(subdataset_path, "images", "train", image)
                annot_path = os.path.join(subdataset_path, "labels", "train", image.replace(".png", ".txt").replace(".tiff", "txt"))
                
                with open(os.path.join(subdataset_path, "data_config.yaml"), "r") as file:
                    config = yaml.safe_load(file)
                    class_names = config["names"]

                image = cv2.imread(image_path)
                image_height, image_width = image.shape[:2]

                annots = []
                with open(annot_path, "r") as f:
                    for line in f:
                        class_id, center_x, center_y, width, height = map(float, line.split())
                        class_name = class_names[int(class_id)]
                        
                        if class_name not in excluded_litter:
                            annots.append(Rectangle(
                                int((center_x - (width / 2)) * image_width),
                                int((center_y - (height / 2)) * image_height),
                                int((center_x + (width / 2)) * image_width),
                                int((center_y + (height / 2)) * image_height),
                                class_name
                            ))

                data.append([subdataset, image_path, annot_path, annots])
            
    df = pd.DataFrame(columns=columns, data=data)
    return df


def create_files(dataset_path: str, df: pd.DataFrame, split: str):
    print("Creating files for", split, "set ...")
    df["new_image_path"] = df["image_path"].apply(lambda x: os.path.join(dataset_path, "images", split, x.split("/")[-1]))
    df["new_annot_path"] = df["new_image_path"].apply(lambda x: x.replace("images", "labels").replace(".png", ".txt").replace(".tiff", ".txt"))

    for i, row in df.iterrows():
        shutil.copy(row["image_path"], row["new_image_path"])

        img = cv2.imread(row["image_path"])
        height, width = img.shape[:2]

        with open(row["new_annot_path"], "w") as f:
            for pile in row["piles"]:
                f.write(f"0 {pile.center[0]/width} {pile.center[1]/height} {pile.width/width} {pile.height/height}\n")

        # print("created files for", row["new_image_path"].split("/")[-1])


if __name__ == "__main__":
    pile_margin = 13
    new_dataset_path = "/home/anna/Datasets/created/piles"
    train_sets = ["ghost-net", "bags_9", "bags_12", "black-bed_15", "mandrac-green-sea", "mandrac-transparent-marina"]

    ################################################
    print("Reading images from datasets ...")
    df = read_all_datasets()

    print("Merging rectangles ...")
    df["piles"] = df["annots"].apply(lambda x: merge_rectangles(x, pile_margin))

    print("Total images:", len(df), "Total piles:", df["piles"].apply(len).sum())
    print("Splitting data...")

    train = df[df["image_path"].str.contains("|".join(train_sets), regex=True)].index
    test = df[~df.index.isin(train)].index

    train_df = df.loc[train][["image_path", "piles"]]
    test_df = df.loc[test][["image_path", "piles"]]

    print("Train size:", len(train_df), "Test size:", len(test_df))

    if not os.path.exists(os.path.join(new_dataset_path)):
        os.makedirs(os.path.join(new_dataset_path, "images", "train"))
        os.makedirs(os.path.join(new_dataset_path, "images", "test"))
        os.makedirs(os.path.join(new_dataset_path, "labels", "train"))
        os.makedirs(os.path.join(new_dataset_path, "labels", "test"))

    with open(os.path.join(new_dataset_path, "data_config.yaml"), "w") as file:
        yaml.dump({"names": ["pile"], "nc": 1}, file)

    create_files(new_dataset_path, train_df, "train")
    create_files(new_dataset_path, test_df, "test")