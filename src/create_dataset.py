import random
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import yaml

from detection import merge_rectangles
from shapes import Rectangle


def read_all_datasets(dataset_path = "/home/anna/Datasets/annotated",
                      excluded_litter = ["grass_bio_brown", "flake_PE_black", "flake_PE_transparent"],
                      channel="RGB.png") -> pd.DataFrame:
    """ Read every annotation in the every image of certain path from subcatalogues of dataset_path
    @param excluded_litter – litter with this classes is not included in the result
    @param channel – taking images of this channel as a base
    @return dataframe with subdataset, image_path, "annot_path" – path to annotation file, annots – all the Rectangles corresponding to annotated litter 
    """

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


def create_files(dataset_path: str, df: pd.DataFrame, split: str, new_image_size: tuple) -> None:
    """ Copy images and annotations to the new locations 
    @param dataset_path – path to directory with new dataset
    @param df – containd image_path (of the originak image) and "piles" (Rectangles with annotations)
    @param split – train, test or val (or other)
    @param new_image_size – values to resize the image with
    """

    print("Creating files for", split, "set ...")
    df["new_image_path"] = df["image_path"].apply(lambda x: os.path.join(dataset_path, "images", split, x.split("/")[-1]))
    df["new_annot_path"] = df["new_image_path"].apply(lambda x: x.replace("images", "labels").replace(".png", ".txt").replace(".tiff", ".txt"))

    for i, row in df.iterrows():
        # shutil.copy(row["image_path"], row["new_image_path"])
        image = cv2.imread(row["image_path"])
        height, width = image.shape[:2]

        image = cv2.resize(image, new_image_size)
        cv2.imwrite(row["new_image_path"], image)


        with open(row["new_annot_path"], "w") as f:
            for pile in row["piles"]:
                f.write(f"0 {pile.center[0]/width} {pile.center[1]/height} {pile.width/width} {pile.height/height}\n")

        # print("created files for", row["new_image_path"].split("/")[-1])


def export_splits(train_df, val_df, test_df, new_dataset_path) -> None:
    """Create new dataset according to the splits """

    print(f"""
        Train:
            - Images: {len(train_df)} ({len(train_df) / len(df) * 100:.2f}%)
            - Piles: {train_df["piles"].apply(len).sum()} ({train_df["piles"].apply(len).sum() / df["piles"].apply(len).sum() * 100:.2f}%)
        Val:
            - Images: {len(val_df)} ({len(val_df) / len(df) * 100:.2f}%)
            - Piles: {val_df["piles"].apply(len).sum()} ({val_df["piles"].apply(len).sum() / df["piles"].apply(len).sum() * 100:.2f}%)
        Test:
            - Images: {len(test_df)} ({len(test_df) / len(df) * 100:.2f}%)
            - Piles: {test_df["piles"].apply(len).sum()} ({test_df["piles"].apply(len).sum() / df["piles"].apply(len).sum() * 100:.2f}%)
        """)
    

    if not os.path.exists(new_dataset_path):
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(new_dataset_path, "images", split))
            os.makedirs(os.path.join(new_dataset_path, "labels", split))

    dataset_name = new_dataset_path.split("/")[-1]

    with open(os.path.join(new_dataset_path, f"{dataset_name}.yaml"), "w") as file:
        yaml.dump({
            "train": f"{new_dataset_path}/images/train",
            "val": f"{new_dataset_path}/images/val",
            "test": f"{new_dataset_path}/images/test",
            "names": ["pile"], 
            "nc": 1
            }, file)

    create_files(new_dataset_path, train_df, "train", new_image_size)
    create_files(new_dataset_path, val_df, "val", new_image_size)
    create_files(new_dataset_path, test_df, "test", new_image_size)

if __name__ == "__main__":
    """Read the annotated images, merge piles and export them to a new daatset with train, test, val splits """

    pile_margin = 13 # litter distant by this number of pixels will be merged to pile
    new_image_size = (800, 608)
    
    ################################################
    print("Reading images from datasets ...")
    df = read_all_datasets(channel="RGB.png")

    print("Merging rectangles ...")
    df["piles"] = df["annots"].apply(lambda x: merge_rectangles(x, pile_margin))

    print("Total images:", len(df), "Total piles:", df["piles"].apply(len).sum())
    print("Splitting data...")

    ###############################################
    new_dataset_path = "/home/anna/Datasets/created/piles_m13"

    # random split
    # train_df, test_val = train_test_split(df, test_size=0.4, random_state=42)
    # test_df, val_df = train_test_split(test_val, test_size=0.5, random_state=42)
    
    # hand split
    train_sets = ["ghost-net", "bags_9","black-bed_15", "mandrac-green-sea", "mandrac-transparent-marina"]
    test_sets = ["green-net", "bags_12"] # and some of mandrac-green-sea with the shore
    from_train_to_test = [113, 123, 93, 110, 120, 119] #[108, 117, 109, 99, 100]

    train = df[df["image_path"].str.contains("|".join(train_sets), regex=True)].index
    test = df[df["image_path"].str.contains("|".join(test_sets), regex=True)].index
    val = df[~df.index.isin(train.union(test))].index

    train.drop(from_train_to_test),
    test.append(pd.Index(from_train_to_test))

    train_df = df.loc[train][["image_path", "piles"]]
    test_df = df.loc[test][["image_path", "piles"]]
    val_df = df.loc[val][["image_path", "piles"]]
    ################################################

    export_splits(train_df, val_df, test_df, new_dataset_path)

    