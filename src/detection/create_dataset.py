"""
Read the annotated images, merge piles and export them to a new daatset with train, test, val splits
"""

import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import click

from detection.litter_detection import merge_rectangles
from processing.load import load_aligned
from processing.evaluate_index import get_custom_index
from processing.consts import CHANNELS
from detection.shapes import Rectangle


def read_all_datasets(
    dataset_path="/home/anna/Datasets/annotated",
    excluded_litter=["grass_bio_brown", "flake_PE_black", "flake_PE_transparent"],
    excluded_sets=[],
    channel="RGB.png",
) -> pd.DataFrame:
    """
    Read every annotation in every image from subdirectories of dataset_path.

    Args:
        dataset_path (str, optional): Path to the dataset containing subdirectories with images and annotations.
        excluded_litter (list, optional): List of litter classes to exclude from the result.
        excluded_sets (list, optional): List of subdatasets to exclude from the result.
        channel (str, optional): Image channel to use as a base.

    Returns:
        pd.DataFrame: DataFrame containing subdataset, image_path, annot_path (path to annotation file),
        and annots (all Rectangles corresponding to annotated litter).
    """

    subdatasets = next(os.walk(dataset_path))[1]
    subdatasets.remove("warp_matrices")
    for excluded_set in excluded_sets:
        subdatasets.remove(excluded_set)

    data = []
    columns = ["subdataset", "image_path", "annot_path", "annots"]

    for subdataset in subdatasets:
        subdataset_path = os.path.join(dataset_path, subdataset)
        for image in os.listdir(os.path.join(subdataset_path, "images", "train")):
            if image.endswith(channel):  # only one image per group
                image_path = os.path.join(subdataset_path, "images", "train", image)
                annot_path = os.path.join(
                    subdataset_path,
                    "labels",
                    "train",
                    image.replace(".png", ".txt").replace(".tiff", "txt"),
                )

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
                            annots.append(
                                Rectangle(
                                    int((center_x - (width / 2)) * image_width),
                                    int((center_y - (height / 2)) * image_height),
                                    int((center_x + (width / 2)) * image_width),
                                    int((center_y + (height / 2)) * image_height),
                                    class_name,
                                )
                            )

                data.append([subdataset, image_path, annot_path, annots])

    df = pd.DataFrame(columns=columns, data=data)
    return df


def apply_formula(image_path: str, formula: str, is_complex: bool = False) -> np.array:
    """
    Apply formula to the image.
    Args:
        image_path (str): Path to the image.
        formula (str): Formula to apply to the image.
        is_complex (bool, optional): True means the formula is more complex and the image is normalized to [0, 255] before applying the formula.

    Returns:
        np.array: Image with the formula applied.
    """

    dir_path = "/".join(image_path.split("/")[:-1])
    image_nr = "_".join(image_path.split("/")[-1].split("_")[:4])

    im_aligned = load_aligned(dir_path, image_nr)

    image = get_custom_index(formula, im_aligned, is_complex)
    image = (image * 255).astype(np.uint8)

    return image


def create_files(
    dataset_path: str,
    df: pd.DataFrame,
    split: str,
    new_image_size: tuple,
    formula: str = None,
    is_complex: bool = False,
) -> None:
    """
    Copy images and annotations to the new locations.

    Args:
        dataset_path (str): Path to directory with new dataset.
        df (pd.DataFrame): Contains image_path (of the original image) and "piles" (Rectangles with annotations).
        split (str): Train, test, or val (or other).
        new_image_size (tuple): Values to resize the image with.
        formula (str, optional): Formula to apply to the image. If not specified, the image is only resized.
        is_complex (bool, optional): True means the formula is more complex and the image is normalized to [0, 255] before applying the formula.

    Returns:
        None
    """

    print("Creating files for", split, "set ...")
    df["new_image_path"] = df["image_path"].apply(
        lambda x: os.path.join(dataset_path, "images", split, x.split("/")[-1])
    )
    df["new_annot_path"] = df["new_image_path"].apply(
        lambda x: x.replace("images", "labels").replace(".png", ".txt").replace(".tiff", ".txt")
    )

    for i, row in df.iterrows():
        image = cv2.imread(row["image_path"])
        height, width = image.shape[:2]

        if formula:
            image = apply_formula(row["image_path"], formula, is_complex)
            row["new_image_path"] = "_".join(row["new_image_path"].split("_")[:-1]) + "_formula.png"
            row["new_annot_path"] = "_".join(row["new_annot_path"].split("_")[:-1]) + "_formula.txt"

        image = cv2.resize(image, new_image_size)

        cv2.imwrite(row["new_image_path"], image)

        with open(row["new_annot_path"], "w") as f:
            for pile in row["piles"]:
                f.write(
                    f"0 {pile.center[0]/width} {pile.center[1]/height} {pile.width/width} {pile.height/height}\n"
                )

        # print("created files for", row["new_image_path"].split("/")[-1])


def export_splits(
    train_df,
    val_df,
    test_df,
    df,
    new_dataset_path: str,
    new_image_size: tuple,
    formula: str = None,
    is_complex: bool = False,
) -> None:
    """Create new dataset according to the splits
    Args:
        train_df (pd.DataFrame): df with train split.
        val_df (pd.DataFrame): df with val split.
        test_df (pd.DataFrame): df with test split.
        df (pd.DataFrame): Contains image_path (of the original image) and "piles" (Rectangles with annotations).
        new_dataset_path (str): Path to directory with new dataset.
        new_image_size (tuple): Values to resize the image with.
        formula (str, optional): Formula to apply to the image. If not specified, the image is only resized.
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

    with open(os.path.join(new_dataset_path, f"{dataset_name}.yaml"), "w") as file:
        yaml.dump(
            {
                "train": f"../../{dataset_name}/images/train",
                "val": f"../../{dataset_name}/images/val",
                "test": f"../../{dataset_name}/images/test",
                "names": ["pile"],
                "nc": 1,
            },
            file,
        )

    create_files(new_dataset_path, train_df, "train", new_image_size, formula, is_complex)
    create_files(new_dataset_path, val_df, "val", new_image_size, formula, is_complex)
    create_files(new_dataset_path, test_df, "test", new_image_size, formula, is_complex)


@click.command()
@click.option("--pile_margin", "-m", default=13, help="Litter distant by this number of pixels will be merged to pile", show_default=True)
@click.option("--new_image_size", "-sz", nargs=2, default=(800, 608), help="New image size", show_default=True)
@click.option("--split", "-s", default="random", help="Split type (random or hand)", show_default=True)
@click.option("--new_dataset_name", "-n", help="Name of the new dataset")
@click.option("--formula", "-f", help="Formula applied to image channels, can be None, then RGB images are saved")
@click.option("--exclude", "-e", default ="", help="Datasets to exclude from the new dataset.")
def main(pile_margin, new_image_size, split,new_dataset_name, formula, exclude):
    """
    Create a new dataset with train, test, val splits.
    """
    print(f"Creating a dataset {new_dataset_name}...")

    is_complex = False
    new_dataset_path = f"/home/anna/Datasets/created/{new_dataset_name}"
    if formula:
        for ch, n in CHANNELS.items():
            formula = formula.replace(str(n), ch)
        print("Formula: ", formula, " will be applied to the images.")
        if len(formula) > 40:
            print("Treating the formula as complex.")
            is_complex = True

    excluded_sets = [] if exclude == "" else [exclude]

    ################################################
    print("Reading images from datasets ...")

    df = read_all_datasets(
        dataset_path="/home/anna/Datasets/annotated",
        excluded_litter=["grass_bio_brown", "flake_PE_black", "flake_PE_transparent"],
        excluded_sets=excluded_sets,
        channel="RGB.png",
    )

    print("Merging rectangles ...")
    df["piles"] = df["annots"].apply(lambda x: merge_rectangles(x, pile_margin))

    print("Total images:", len(df), "Total piles:", df["piles"].apply(len).sum())
    print("Splitting data...")

    ###############################################

    if split == "random":
        # random split
        train_df, test_val = train_test_split(df, test_size=0.4, random_state=42)
        test_df, val_df = train_test_split(test_val, test_size=0.5, random_state=42)

    elif split == "hand":
        # hand split
        train_sets = [
            "ghost-net",
            "bags_9",
            "black-bed_15",
            "mandrac-green-sea",
            "mandrac-transparent-marina",
        ]
        test_sets = ["green-net", "bags_12"]  # and some of mandrac-green-sea with the shore
        from_train_to_test = [113, 123, 93, 110, 120, 119]  # [108, 117, 109, 99, 100]

        train = df[df["image_path"].str.contains("|".join(train_sets), regex=True)].index
        test = df[df["image_path"].str.contains("|".join(test_sets), regex=True)].index
        val = df[~df.index.isin(train.union(test))].index

        train.drop(from_train_to_test),
        test.append(pd.Index(from_train_to_test))

        train_df = df.loc[train][["image_path", "piles"]]
        test_df = df.loc[test][["image_path", "piles"]]
        val_df = df.loc[val][["image_path", "piles"]]

    else:
        print("Split not recognised")
    ################################################

    export_splits(train_df, val_df, test_df, df, new_dataset_path, new_image_size, formula, is_complex)


if __name__ == "__main__":
    main()
