#!/home/anna/miniconda3/envs/micasense/bin/python3

import sys
from typing import Counter
import cv2
import os

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import yaml

from detection.litter_detection import find_pool

"""Display gui showing all the items from each class on all the photos from the group. Create the summary for each photo with the number of items of each class. Save everything to out directory."""

def draw_boxes_one_class(pool_image, image_path, label_path, filter_class, color, image_shape, correction, scale):
    """ Draw bounding boxes in yolo format on image"""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_height, image_width = image_shape

    i=0
    # Open and parse YOLO label file
    with open(label_path, "r") as f:
        for line in f:
            class_id, center_x, center_y, width, height = map(float, line.split())
            if filter_class != -1 and class_id != filter_class:
                continue

            class_id = int(class_id)

            # Convert YOLO format to pixel values in image coordinates
            x1 = int((center_x - (width / 2)) * image_width)
            x2 = int((center_x + (width / 2)) * image_width)
            y1 = int((center_y - (height / 2)) * image_height)
            y2 = int((center_y + (height / 2)) * image_height)

            trash = image[y1:y2, x1:x2]

            # convert from image coordinates to pool coordinates and to common pool coordinates
            x1 = int((x1 - correction[0]) * scale[0])
            x2 = int((x2 - correction[0]) * scale[0])
            y1 = int((y1 - correction[1]) * scale[1])
            y2 = int((y2 - correction[1]) * scale[1])
                    
            # Draw bounding box and label
            cv2.rectangle(pool_image, (x1, y1), (x2,y2), color, 5)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(pool_image.shape[1], x2)
            y2 = min(pool_image.shape[0], y2)
            if trash.size > 0 and x2 - x1 > 0 and y2 - y1 > 0:
                pool_image[y1:y2, x1:x2] = cv2.resize(trash, (x2 - x1, y2 - y1))

            i += 1

    return pool_image, i


def get_wrong_on_each_photo(wrong, group, out_path="out"):
    with open(f"{out_path}/{group}_summary.txt", "w") as f:
        for photo in wrong:
            f.write(f"\nPhoto {photo}:\n")
            if len(wrong[photo]) == 0:
                f.write("OK\n")
            else:
                for name, n_obj, correct_n_obj in wrong[photo]:
                    f.write(f"{n_obj} {name} instead of {correct_n_obj} ({'too little' if n_obj < correct_n_obj else 'too many'})\n")
    print(f"Summary saved to {out_path}/{group}_summary.txt")


def show_class_in_normalised_pool(photo_names, n_objects, raw_palette, image_bbs, group, config, class_id, out_path="out"):
    plt.figure(figsize=(20,8))
    legend_elements = [
        Patch(facecolor=color, edgecolor='black', label=f'photo {name}, objects: {n_obj}') for color, name, n_obj in zip(raw_palette[:i+1], photo_names, n_objects)]
    plt.title(f"Group {group} {config['names'][class_id]}")
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.imshow(image_bbs)
    plt.axis('off')

    plt.savefig(f"{out_path}/{group}_{config['names'][class_id]}.png")
    print(f"Saved to {out_path}/{group}_{config['names'][class_id]}.png")
    # plt.show()


def get_correction_and_scale(path, image_fn, desired_width, desired_height):
    # find the pool
    image = cv2.imread(f"{path}/images/train/{image_fn}", cv2.COLOR_BGR2GRAY)
    altitude = int(image_fn.split("_")[-2]) 
    pool = find_pool(image, altitude)

    if pool is None:
        equalized_image = cv2.equalizeHist(image)
        pool = find_pool(equalized_image, altitude)  
    if pool is None:
        print(f"Pool not found in {image_fn}.")

    correction = (pool.x_l, pool.y_b) if pool is not None else (0, 0)

    if pool is not None:
        scale_x = desired_width/(pool.x_r - pool.x_l)
        scale_y = desired_height/(pool.y_t - pool.y_b)
    else: 
        scale_x = desired_width/image.shape[1]
        scale_y = desired_height/image.shape[0]

    return correction, (scale_x, scale_y)


if __name__ == "__main__":
    # get filenames
    if len(sys.argv) < 4:
        print("Usage: python3 check_annotations.py [dataset_name] [out_path] [groups]")
        sys.exit(1)

    path = f"/home/anna/Datasets/annotated/{sys.argv[1]}"
    out_path = sys.argv[2]
    groups = sys.argv[3:] #["9:00", "12:00", "15:00", "5_bags"]
    image_fns = [image_fn for image_fn in os.listdir(f"{path}/images/train") if image_fn.endswith("_meanRE.png")]
    image_fns.sort()

    with open(f"{path}/data_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    image_shape = cv2.imread(f"{path}/images/train/{image_fns[0]}", cv2.COLOR_BGR2GRAY).shape[:2] # assuming all the images have the same dimensions
    pool_width = 2450
    pool_height = int(2.5/5.4 * pool_width)

    raw_palette = plt.cm.tab20.colors + plt.cm.tab20b.colors
    palette = [tuple(int(c * 255) for c in color) for color in raw_palette]


    for group in groups:
        group_fns = [fn for fn in image_fns if group in fn]
        photo_names = [("_".join(image_fn.split("_")[-3:-1])) for image_fn in group_fns]

        if len(group_fns) == 0:
            print(f"No images found for group {group}. Check the group name.")
            continue
        
        print(f"Processing {group} group with {len(group_fns)} images")
        
        wrong = {photo: [] for photo in photo_names}
        objects = {}
        
        if len(group_fns) > len(palette): print("Not enough distinct colours. Reusing colours.")

        for class_id in range(len(config["names"])):
            image_bbs = np.zeros((pool_height, pool_width, 3), dtype=np.uint8) # representing the pool
            n_objects = []

            for i, image_fn in enumerate(group_fns):
                correction, scale = get_correction_and_scale(path, image_fn, pool_width, pool_height)

                # show bbs
                image_bbs, n_obj = draw_boxes_one_class(image_bbs,
                                    f"{path}/images/train/{image_fn.replace('_meanRE', '_RGB')}",
                                    f"{path}/labels/train/{image_fn.replace('.png', '.txt')}",
                                    class_id,
                                    palette[i%len(palette)],
                                    image_shape,
                                    correction,
                                    scale
                                    )
        
                n_objects.append(n_obj)


            # plot if the number of items is not the same in all the images
            most_common_n = Counter(n_objects).most_common(1)[0][0]
            objects[class_id] = (most_common_n, min(n_objects), max(n_objects))

            wrong_n_obj = [(photo, n_obj) for photo, n_obj in zip(photo_names, n_objects) if n_obj != most_common_n]

            for photo, n_obj in wrong_n_obj:
                wrong[photo].append((config["names"][class_id], n_obj, most_common_n))
 
            print(f"{config['names'][class_id]}: {most_common_n} objects ({min(n_objects)}-{max(n_objects)})")

            # if len(wrong_n_obj) > 0:
            show_class_in_normalised_pool(photo_names, n_objects, raw_palette, image_bbs, group, config, class_id, out_path)
            

        get_wrong_on_each_photo(wrong, group, out_path)

        # save the number of objects of each class for the group
        with open(f"{out_path}/{group}_objects.txt", "w") as f:
            for (class_id, n_obj) in objects.items():
                f.write(f"{config['names'][class_id]}: {n_obj[0]} objects ({n_obj[1]} - {n_obj[2]})\n")