"""
Read the annotated images, merge piles and export them to a new daatset with train, test, val splits
"""

from typing import List
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import click

from src.processing.load import load_aligned
from src.processing.evaluate_index import apply_formula
from src.processing.consts import CHANNELS, DATASET_BASE_PATH
from src.shapes import Rectangle


def merge_rectangles(rects: List[Rectangle], margin=0) -> List[Rectangle]:
    """ merge rectangles that overlap to bigger rectangles """
    merged = [rects[0]]

    for rect in rects[1:]:
        for m in merged:
            if m.intersection(rect, margin):
                merged.remove(m)
                merged.append(m | rect)
                break
        else:
            merged.append(rect)

    # check if we cant merge any more
    changed = True
    while changed:
        changed = False
        for i, rect in enumerate(merged):
            for j, other in enumerate(merged):
                if i != j and rect.intersection(other, margin):
                    merged.remove(rect)
                    merged.remove(other)
                    merged.append(rect | other)
                    changed = True
                    break

    return merged

def read_one_dataset(
    dataset_path: str,
    excluded_litter: List[str] = ["grass_bio_brown", "flake_PE_black", "flake_PE_transparent"],
    channel: str = "RGB.png",
    has_aug: bool = False,
) -> List:
    """
    Read every annotation in every image from the dataset.

    Args:
        dataset_path (str, optional): Path to the dataset containing images and annotations.
        excluded_litter (list, optional): List of litter classes to exclude from the result.
        excluded_sets (list, optional): List of subdatasets to exclude from the result.
        channel (str, optional): Image channel to use as a base.

    Returns:
        list: list of entries with dataset, subdataset (first part of the image name), image_path, annot_path (path to annotation file),
        and annots (all Rectangles corresponding to annotated litter).
    """

    data = []

    image_names = os.listdir(os.path.join(dataset_path, "images", "train"))
    if has_aug:
        image_names += os.listdir(os.path.join(dataset_path, "images", "train_aug"))

    for image_name in image_names:
        if image_name.endswith(channel):  # only one image per group
            split = "train_aug" if has_aug and "_aug" in image_name else "train"
            image_path = os.path.join(dataset_path, "images", split, image_name)
            annot_path = os.path.join(
                dataset_path,
                "labels",
                split,
                image_name.rsplit(".", 1)[0] + ".txt"
            )    

            with open(os.path.join(dataset_path, "data_config.yaml"), "r") as file:
                config = yaml.safe_load(file)
                class_names = config["names"]

            image = cv2.imread(image_path)

            annots = []
            with open(annot_path, "r") as f:
                for line in f:
                    class_id, center_x, center_y, width, height = map(float, line.split())
                    class_name = class_names[int(class_id)]

                    if class_name not in excluded_litter:
                        annots.append(
                            Rectangle.from_yolo(
                                center_x, center_y, width, height, class_name, image.shape[:2]
                            )
                        )
            # if len(annots) == 0:
            #     print("No annotations in", image_path)
                # continue

            subdataset = image_name.split("_")[0]  # assuming subdataset is the first part of the image name

            data.append([dataset_path.rsplit("/", 1)[1], subdataset, image_path, annot_path, annots])

    return data
    

def read_all_datasets(
    datasets,
    dataset_path=f"{DATASET_BASE_PATH}/annotated",
    excluded_litter=["grass_bio_brown", "flake_PE_black", "flake_PE_transparent"],
    channel="RGB.png",
    has_aug = False,
) -> pd.DataFrame:
    """
    Read every annotation in every image from subdirectories of dataset_path.

    Args:
        datasets (list): List of subdatasets to include.
        dataset_path (str, optional): Path to the dataset containing subdirectories with images and annotations.
        excluded_litter (list, optional): List of litter classes to exclude from the result.
        channel (str, optional): Image channel to use as a base.

    Returns:
        pd.DataFrame: DataFrame containing dataset, subdataset, image_path, annot_path (path to annotation file),
        and annots (all Rectangles corresponding to annotated litter).
    """

    data = []
    columns = ["dataset", "subdataset", "image_path", "annot_path", "annots"]

    for dataset in datasets:
        new_data = read_one_dataset(
            dataset_path=os.path.join(dataset_path, dataset),
            excluded_litter=excluded_litter,
            channel=channel,
            has_aug=has_aug
        )
        data.extend(new_data)

    df = pd.DataFrame(columns=columns, data=data)
    return df


def create_files(
    dataset_path: str,
    df: pd.DataFrame,
    split: str,
    new_image_size: tuple,
    channels: list = ["R", "G", "B"],
    is_complex: bool = False,
    classes: list = ["pile"]
) -> None:
    """
    Copy images and annotations to the new locations.

    Args:
        dataset_path (str): Path to directory with new dataset.
        df (pd.DataFrame): Contains image_path (of the original image) and "piles" (Rectangles with annotations).
        split (str): Train, test, or val (or other).
        new_image_size (tuple): Values to resize the image with.
        channels (list, optional): Channel names or formulas to apply to the image. If not specified, the image is resized and saved as RGB.
        is_complex (bool, optional): True means the formula in channels is more complex and the image is normalized to [0, 255] before applying the formula.

    Returns:
        None
    """

    def get_new_name(row):
        path = os.path.join(dataset_path, "images", split, row.split("/")[-1])
        channels_str = "_".join(channels).replace("/", ":").replace(" ", "")
        path = "_".join(path.split("_")[:-1]) + "_" + channels_str + ".png"
        return path

    print("Creating files for", split, "set ...")
    df["new_image_path"] = df["image_path"].apply(
        lambda x: get_new_name(x)
    )
   
    df["new_annot_path"] = df["new_image_path"].apply(
        lambda x: x.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    )

    for i, row in df.iterrows():
        dir_path = "/".join(row["image_path"].split("/")[:-1])
        image_nr = "_".join(row["image_path"].split("/")[-1].split("_")[:-1])
        im_aligned = load_aligned(dir_path, image_nr)
        height, width = im_aligned.shape[:2]

        new_image = np.zeros((height, width, len(channels)))

        for i, channel in enumerate(channels):
            if channel in CHANNELS:
                new_image[:, :, i] = im_aligned[:, :, CHANNELS[channel]]
            else:
                new_image[:, :, i] = apply_formula(im_aligned, channel, is_complex)

        new_image = cv2.resize(new_image, new_image_size)

        new_image = new_image.astype(np.uint8)
        cv2.imwrite(row["new_image_path"], cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)) # imwrite assumes BGR

        with open(row["new_annot_path"], "w") as f:
            for pile in row["piles"]:
                if len(classes) == 1:
                    class_id = 0
                else:
                    class_id = classes.index(pile.label)
                f.write(
                    f"{class_id} {pile.center[0]/width} {pile.center[1]/height} {pile.width/width} {pile.height/height}\n"
                )

        # print("created files for", row["new_image_path"].split("/")[-1])


def export_splits(
    train_df,
    val_df,
    test_df,
    df,
    new_dataset_path: str,
    new_image_size: tuple,
    channels: list = ["R", "G", "B"],
    is_complex: bool = False,
    one_class: bool = True
) -> None:
    """Create new dataset according to the splits
    Args:
        train_df (pd.DataFrame): df with train split.
        val_df (pd.DataFrame): df with val split.
        test_df (pd.DataFrame): df with test split.
        df (pd.DataFrame): Contains image_path (of the original image) and "piles" (Rectangles with annotations).
        new_dataset_path (str): Path to directory with new dataset.
        new_image_size (tuple): Values to resize the image with.
        channels (list, optional): Channel names or formulas to apply to the image. If not specified, the image is resized and saved as RGB.
        is_complex (bool, optional): True means the formula is more complex and the image is normalized to [0, 255] before applying the formula.


    Returns:
        None
    """

    print(
        f"""
        Train:
            - Images: {len(train_df)} ({len(train_df) / len(df) * 100:.2f}%)
            - Piles: {train_df["piles"].apply(len).sum()} ({train_df["piles"].apply(len).sum() / df["piles"].apply(len).sum() * 100:.2f}%)
        Val:
            - Images: {len(val_df)} ({len(val_df) / len(df) * 100:.2f}%)
            - Piles: {val_df["piles"].apply(len).sum()} ({val_df["piles"].apply(len).sum() / df["piles"].apply(len).sum() * 100:.2f}%)
        Test:
            - Images: {len(test_df)} ({len(test_df) / len(df) * 100:.2f}%)
            - Piles: {test_df["piles"].apply(len).sum()} ({test_df["piles"].apply(len).sum() / df["piles"].apply(len).sum() * 100:.2f}%)
        """
    )

    if not os.path.exists(new_dataset_path):
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(new_dataset_path, "images", split))
            os.makedirs(os.path.join(new_dataset_path, "labels", split))

    dataset_name = new_dataset_path.split("/")[-1]

    if one_class:
        class_names = ["pile"]
        with open(os.path.join(new_dataset_path, f"{dataset_name}.yaml"), "w") as file:
            yaml.dump(
                {
                    "train": f"images/train",
                    "val": f"images/val",
                    "test": f"images/test",
                    "names": ["pile"],
                    "nc": 1,
                    "path": new_dataset_path
                },
                file,
            )
    else:
        class_names = df["piles"].explode().apply(lambda x: x.label if not pd.isna(x) else x).dropna().unique().tolist()

        with open(os.path.join(new_dataset_path, f"{dataset_name}.yaml"), "w") as file:
            yaml.dump(
                {
                    "train": f"images/train",
                    "val": f"images/val",
                    "test": f"images/test",
                    "names": class_names,
                    "nc": len(class_names),
                    "path": new_dataset_path
                },
                file,
            )

    create_files(new_dataset_path, train_df, "train", new_image_size, channels, is_complex, classes=class_names)
    create_files(new_dataset_path, val_df, "val", new_image_size, channels, is_complex, classes=class_names)
    create_files(new_dataset_path, test_df, "test", new_image_size, channels, is_complex, classes=class_names)


@click.command()
@click.option("--split", "-s", default="random", help="Split type (random-all (all data randomly), random-sub (each subdataset splitted randomly)  or val (all to validation))", show_default=True)
@click.option("--new_dataset_name", "-n", help="Name of the new dataset")
@click.option("--channels", "-c", multiple=True, default=["R", "G", "B"], help="channels or formulas to save", show_default=True)
@click.option("--datasets", "-ds", multiple=True, default = (), help="Datasets to include in the new dataset.")
@click.option("--test-ds", "-t", multiple=True, default =(), help="Datasets to include in testing dataset, taking all by default.")
@click.option("--aug", "-a", default=False, is_flag=True, help="Use augmented images from train_aug", show_default=True)
@click.option("--one_class", "-o", default=True, help="Create dataset with one class (pile) only", show_default=True)
@click.option("--pile_margin", "-m", default=None, help="Litter distant by this number of pixels will be merged to pile", show_default=True, type=int)
@click.option("--new_image_size", "-sz", nargs=2, default=(800, 608), help="New image size", show_default=True)

def main(split, new_dataset_name, channels, datasets, test_ds, aug, one_class, pile_margin, new_image_size):
    """
    Create a new dataset with train, test, val splits.
    """
    print(f"Creating a dataset {new_dataset_name}...")

    is_complex = False
    new_dataset_path = f"{DATASET_BASE_PATH}/created/{new_dataset_name}"

    if os.path.exists(new_dataset_path):
        print("Dataset already exists. Skipping...")
        return

    channels = list(channels)
    for i in range(len(channels)):
        channels[i] = channels[i].upper()
        for ch, n in CHANNELS.items():
            channels[i] = channels[i].replace(str(n), ch)
        print(f"Channel {i} will be: {channels[i]}")
        if len(channels[i]) > 40:
            print("Treating the formula as complex. (Will be applied to all the formulas)")
            is_complex = True


    ################################################
    print("Reading images from datasets ...")

    df = read_all_datasets(
        datasets=datasets,
        dataset_path=f"{DATASET_BASE_PATH}/annotated",
        excluded_litter=["grass_bio_brown", "flake_PE_black", "flake_PE_transparent"],
        channel="ch0.tiff",
        has_aug=aug
    )

    if pile_margin is not None and pile_margin != "None":
        print(f"Merging rectangles with margin {pile_margin}...")
        df["piles"] = df["annots"].apply(lambda x: merge_rectangles(x, pile_margin))
    else :
        df["piles"] = df["annots"]

    print("Total images:", len(df), "Total piles:", df["piles"].apply(len).sum())
    print(df.head())

    print("Splitting data...")

    ###############################################

    if split == "random-all":
        if test_ds == ():
            # random split
            train_df, test_val = train_test_split(df, test_size=0.4, random_state=42)
            test_df, val_df = train_test_split(test_val, test_size=0.5, random_state=42)
        else:
            # random split on test datasets
            train_df, test_val = train_test_split(df[df["dataset"].isin(test_ds)], test_size=0.4, random_state=42)
            test_df, val_df = train_test_split(test_val, test_size=0.5, random_state=42)

            # add the rest of the datasets to train
            train_df = pd.concat([train_df, df[~df["dataset"].isin(test_ds)]])

    elif split == "random-sub":
        unique_subdatasets = df["subdataset"].unique()
        print(f"Randomly splitting each of subdatasets: {', '.join(unique_subdatasets)}")

        # random split for each subdataset
        train_df = pd.DataFrame(columns=["image_path", "piles"])
        val_df = pd.DataFrame(columns=["image_path", "piles"])
        test_df = pd.DataFrame(columns=["image_path", "piles"])

        for subdataset in unique_subdatasets:
            sub_df = df[df["subdataset"] == subdataset]
            if len(test_ds) > 0:
                print("This split doesn't support providing test datasets.")
            else:
                # 60-20-20 split for each subdataset
                train, test_val = train_test_split(sub_df, test_size=0.4, random_state=42)
                test, val = train_test_split(test_val, test_size=0.5, random_state=42)

            train_df = pd.concat([train_df, train])
            val_df = pd.concat([val_df, val])
            test_df = pd.concat([test_df, test])
        
    elif split == "val":
        # all to validation
        val_df = df[["image_path", "piles"]].copy()

        train_df = pd.DataFrame(columns=["image_path", "piles"])
        train_df.astype(val_df.dtypes)
        test_df = train_df.copy()

    else:
        print("Split not recognised. Supports only 'random-all', 'random-sub' and 'val'.")
        exit(1)
    ################################################

    export_splits(train_df, val_df, test_df, df, new_dataset_path, new_image_size, channels, is_complex, one_class=one_class)


if __name__ == "__main__":
    main()
