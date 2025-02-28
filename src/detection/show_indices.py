"""Compare different indices on the same image."""

import sys
sys.path.append('/home/anna/code/multispectral/src')

from matplotlib import pyplot as plt

from processing.load import load_aligned
from processing.evaluate_index import get_custom_index
from processing.consts import CHANNEL_NAMES


if __name__ == "__main__":
    base_path = "/home/anna/Datasets/annotated/"

    images_paths = ["ghost-net/images/train/ghost-net_10_0016_10.png",
                    "black-bed/images/train/black-bed_15_0007_20.png",
                    "mandrac/images/train/mandrac-transparent-marina_12_0031_10.png",
                    "mandrac/images/train/mandrac-green-sea_11_0237_15.png",
                    "mandrac/images/train/mandrac-transparent-sea_10_0091_15.png"]

    formulas = [
        "N # G",
        "(4#1) + (4#0)",

        "(4 # (4 + 1))",
        "(4 # ((3 + 2) + 1))",
        "(2 # 1)",
        "((1 # 4) * (3 # (((1 / 4) + (2 # (1 / (2 * 0)))) + 1)))",
        "(((3 # ((((1 + 3) - (2 / 3)) + (0 / 4)) / 3)) * 1) / 1)",
        "((3 # 1) - (1 # (3 * 2)))",
        "(4 # (2 + 2))",

        "(2+0) - 4"
    ]

    titles = ["rndwi", "meanRE"] + [f"ghost-net {i+1}" for i in range(7)] + ["sea 1"] + ["pool"]

    for image_path in images_paths:
        image_path = base_path + image_path
        dir_path = "/".join(image_path.split("/")[:-1])
        image_nr = "_".join(image_path.split("/")[-1].split("_")[1:3])

        im_aligned = load_aligned(
            dir_path,
            image_nr
        )

        fig, ax = plt.subplots(4, 3, figsize=(24, 12))
        ax = ax.flatten()

        for i, formula in enumerate(formulas):
            for n, channel in enumerate(CHANNEL_NAMES):
                formula = formula.replace(str(n), channel)
            # print(formula)

            result = get_custom_index(formula, im_aligned, norm_to_255=len(formula) > 40)
            ax[i].imshow(result, cmap="gray")
            ax[i].set_title(titles[i])
            ax[i].axis("off")

        plt.show()
