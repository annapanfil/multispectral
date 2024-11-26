import cv2
import os

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import yaml

from detection import find_pool


def draw_boxes_one_class(image, label_path, class_names, filter_class, color, image_shape, correction, scale):
    """ Draw bounding boxes in yolo format on image"""
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

            # convert from image coordinates to pool coordinates and to common pool coordinates
            x1 = int((x1 - correction[0]) * scale[0])
            x2 = int((x2 - correction[0]) * scale[0])
            y1 = int((y1 - correction[1]) * scale[1])
            y2 = int((y2 - correction[1]) * scale[1])
                    
            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2,y2), color, 5)
            # cv2.putText(image, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

            i += 1

    return image, i

def get_correction_and_scale(image_fn, desired_width, desired_height):
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
    path = "/home/anna/Datasets/annotated/bags"
    groups = ["9:00", "12:00", "15:00", "5_bags"]
    image_fns = [image_fn for image_fn in os.listdir(f"{path}/images/train") if image_fn.endswith("_meanRE.png")]

    with open(f"{path}/data_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    image_shape = cv2.imread(f"{path}/images/train/{image_fns[0]}", cv2.COLOR_BGR2GRAY).shape[:2] # assuming all the images have the same dimensions
    pool_width = 2450
    pool_height = int(2.5/5.4 * pool_width)

    palette = [tuple(int(c * 255) for c in color) for color in plt.cm.tab20.colors]

    for group in groups:
        group_fns = [fn for fn in image_fns if group in fn]
        
        print(f"Processing {group} group with {len(group_fns)} images")
        if len(group_fns) > 20: print("Not enough distinct colours. Reusing colours.")

        for class_id in range(len(config["names"])):
            image_bbs = np.zeros((pool_height, pool_width, 3), dtype=np.uint8) # representing the pool
            altitudes = []
            n_objects = []

            for i, image_fn in enumerate(group_fns):
                correction, scale = get_correction_and_scale(image_fn, pool_width, pool_height)

                # show bbs
                image_bbs, n_obj = draw_boxes_one_class(image_bbs,
                                    f"{path}/labels_joined/train/{image_fn.replace('.png', '.txt')}",
                                    config["names"],
                                    class_id,
                                    palette[i%20],
                                    image_shape,
                                    correction,
                                    scale
                                    )
                
                altitudes.append(int(image_fn.split("_")[-2])) 
                n_objects.append(n_obj)


            # plot if the number of items is not the same in all the images
            wrong_n_obj = [altitude for altitude, n_obj in zip(altitudes, n_objects) if n_obj < max(n_objects)]

            if len(wrong_n_obj) > 0:
                plt.figure(figsize=(20,8))
                legend_elements = [
                    Patch(facecolor=color, edgecolor='black', label=f'altitude {altitude}, objects: {n_obj}') for color, altitude, n_obj in zip(plt.cm.tab20.colors[:i+1], altitudes, n_objects)]
                plt.title(f"Group {group} {config['names'][class_id]}")
                plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title=f"Less objects in images: {wrong_n_obj}")
                plt.imshow(image_bbs)
                plt.axis('off')
                plt.show()