import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import subprocess

if __name__ == "__main__":

    ds_path = "/home/anna/Datasets/annotated/"
    annots_prefix = "mandrac3" # for the COCO annot file
    desired_annots_prefix = "mandrac" # added to names of images and annotations in the resulting dataset
    experiment_name = "transparent_and_green"
    experiment_time = "12"
    ds_for_annot_path = f"/home/anna/Datasets/for_annotation/mandrac_2025_04_16/mandrac2025_04_16_{experiment_name}"
    show = False

    # ----------------------------------------------------------------------------------

    # Load COCO JSON
    with open(f"{ds_path}/{annots_prefix}_{experiment_name}_annots.json") as f:
        coco_data = json.load(f)

    # ----- PLOT COCO ANNOTATIONS ----
    # get annots for each image
    image_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        image_anns[ann["image_id"]].append(ann)

    image_id_to_fn = {img["id"]: img["file_name"] for img in coco_data["images"]}

    print(len(image_anns))

    if show:
        for img, anns in image_anns.items():
            # load image
            fn = image_id_to_fn[img]
            alt = fn.split("_")[1] 
            nr = fn.split("_")[0]

            image = cv2.imread(f"{ds_for_annot_path}/{alt}/{nr}_{alt}/{fn.replace('tiff', 'tif')}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # draw annotations
            for ann in anns:
                polygon = np.array(ann["segmentation"]).reshape((-1, 2)).astype(int)
                cv2.polylines(image, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
                bbox = ann["bbox"]  # [x,y,w,h]
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 2)

            plt.imshow(image)
            plt.show()


    # ---- CONVERT TO YOLO AND COPY TO A NEW DIRECTORY ----

    # Load COCO JSON
    with open(f"{ds_path}/{annots_prefix}_{experiment_name}_annots.json") as f:
        coco_data = json.load(f)

    # Create class_id to name mapping (adjust based on your data)
    class_map = {1: "pile"}  # Update with your actual classes

    # Process annotations
    image_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        image_anns[ann["image_id"]].append(ann)

    image_id_to_n = {img["id"]: i for i, img in enumerate(coco_data["images"])}

    print(f"Found {len(image_anns)} annotated images")

    for img_id, anns in image_anns.items():
        # Get image info
        img_info = coco_data["images"][image_id_to_n[img_id]]
        fn = img_info["file_name"]
        img_width, img_height = img_info["width"], img_info["height"]
        
        # Load image
        alt = fn.split("_")[1]
        nr = fn.split("_")[0]

        img_path = f"{ds_for_annot_path}/{alt}/{nr}_{alt}/{fn.replace('tiff', 'tif')}" # in new directory there are only channel images, no RGB or other
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare YOLO output file
        yolo_txt_base_path = f"{ds_path}/{annots_prefix}/{experiment_name}/labels"
        prefix = f"{desired_annots_prefix}-{experiment_name.replace('_', '-')}_{experiment_time}_"
        yolo_txt_paths = [f"{yolo_txt_base_path}/{prefix}{'_'.join(fn.split('_')[:-1])}_ch{i}.txt" for i in range(0, 5)]  

        ann_string = ""
        for ann in anns:
            # Convert COCO bbox [x,y,w,h] to YOLO format [x_center, y_center, w, h] (normalized)
            x, y, w, h = ann["bbox"]
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            class_id = ann.get("category_id", 0) - 1  # YOLO uses 0-based indices
                
            ann_string += f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
            
        for path in yolo_txt_paths:          
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            with open(path, 'w') as yolo_file:
                yolo_file.write(ann_string)

    print("YOLO conversion complete!")
    print(f"Saved the annotations to {ds_path}/{annots_prefix}/{experiment_name}/")


    # ---- COPY IMAGES TO THE NEW DIRECTORY AND RENAME THEM ----
    dst_dir = f"{ds_path}{annots_prefix}/{experiment_name}/images"

    subprocess.run(f"mkdir -p {dst_dir}", shell=True)
    subprocess.run(f"find {ds_for_annot_path} -mindepth 3 -maxdepth 3 -type f -regex '.*ch[0-9]\\.tif' -exec cp {{}} {dst_dir} \\;", shell=True)
    subprocess.run(f"cd {dst_dir} && for file in *; do mv \"$file\" \"{prefix}$file\"; done", shell=True)

    count = len(os.listdir(dst_dir))

    print(f"Copied {count/5:.0f} images (with 5 channels) to {dst_dir} and renamed them.")