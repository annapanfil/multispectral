from itertools import accumulate
import cv2
import os
from typing import List

from matplotlib import pyplot as plt
import numpy as np
from detection import convert_from_pool_to_abs_coords, find_litter
from display import draw_rectangles
import hydra
import optuna
from shapes import Rectangle

def iou(box1: Rectangle, box2: Rectangle) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    box1, box2: Rectangle(x_l, y_b, x_r, y_t)
    """
    # Compute the area of the intersection rectangle
    xl = max(box1.x_l, box2.x_l)
    yb = max(box1.y_b, box2.y_b)
    xr = min(box1.x_r, box2.x_r)
    yt = min(box1.y_t, box2.y_t)

    intersection = max(0, xr - xl) * max(0, yt - yb)

    # Compute the union area
    area1 = (box1.x_r - box1.x_l) * (box1.y_t - box1.y_b)
    area2 = (box2.x_r - box2.x_l) * (box2.y_t - box2.y_b)

    union = area1 + area2 - intersection

    # Compute IoU
    return intersection / union if union > 0 else 0.0


def calculate_ap(gt_boxes, pred_boxes, iou_threshold=0.5, verb=False):
    """
    Calculate Average Precision (AP) for a single IoU threshold for one class.
    Assumes that the predictions don't have confidences.
    """
    tp = [0] * len(pred_boxes)
    fp = [0] * len(pred_boxes)

    # Iterate over ground truth boxes
    for j, gt_box in enumerate(gt_boxes):
        best_iou = 0
        best_pred_idx = -1

        # Find the best matching predicted box for each ground truth box
        for i, pred_box in enumerate(pred_boxes):
            if tp[i] == 1 or fp[i] == 1:
                continue  # Skip already matched or marked boxes

            iou_value = iou(gt_box, pred_box)
            if iou_value > best_iou:
                best_iou = iou_value
                best_pred_idx = i

        # If the IoU exceeds the threshold, count it as a true positive
        if best_iou >= iou_threshold:
            tp[best_pred_idx] = 1
        else:
            # If no match, this predicted box is a false positive
            if best_pred_idx != -1:
                fp[best_pred_idx] = 1

    # Handle unmatched predicted boxes by marking them as false positives
    for i in range(len(pred_boxes)):
        if tp[i] == 0 and fp[i] == 0:  # If not already TP or FP, mark as FP
            fp[i] = 1

    # Calculate precision and recall
    tp_cumsum = list(accumulate(tp))
    fp_cumsum = list(accumulate(fp))
    precision = [tp_cumsum[i] / (tp_cumsum[i] + fp_cumsum[i]) for i in range(len(tp))]
    recall = [tp_cumsum[i] / len(gt_boxes) for i in range(len(tp))]

    if verb:
        # plot precision-recall curve
        plt.plot(recall, precision)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("Precision-Recall curve for ap with threshold " + str(iou_threshold))
        plt.show()

    # Compute Average Precision (AP)
    ap = 0
    for i in range(len(precision)):
        if i == 0 or recall[i] > recall[i - 1]: # it cant go down, but we dont want to plot the same recall value twice
            ap += precision[i] * (recall[i] - recall[i - 1])

    return ap


def mean_ap(gt_boxes_all_images: List[List[Rectangle]], pred_boxes_all_images: List[List[Rectangle]], iou_thresholds=[0.5], verb=False):
    """
    Calculate mean Average Precision (mAP) over multiple IoU thresholds.
    """
    aps = []
    for iou_threshold in iou_thresholds:
        image_aps = []
        for gt_boxes, pred_boxes in zip(gt_boxes_all_images, pred_boxes_all_images):
            ap = calculate_ap(gt_boxes, pred_boxes, iou_threshold, verb)
            image_aps.append(ap)
        aps.append(np.mean(image_aps))

    if verb:
        plt.plot(iou_thresholds, aps)
        plt.xlabel("IoU threshold")
        plt.ylabel("mAP")
        plt.title("Mean Average Precision (mAP) for different IoU thresholds")
        plt.show()

    return np.mean(aps)  # Mean AP over all IoU thresholds



def evaluate_detector(params: dict, images_paths: List[str], ground_truth_boxes: List[Rectangle], verb=False):
    """
    Evaluate the detector using mean IoU with ground truth bounding boxes.
    """

    all_gt_boxes = []  # To store ground truth boxes for each image
    all_pred_boxes = []  # To store predicted boxes for each image

    for img_path, gt_boxes in zip(images_paths, ground_truth_boxes):
        # Detect bounding boxes using the detector
        altitude = int(img_path.split("_")[-2])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)    

        _, detected_boxes, pool, _, _ = find_litter(
            image, img_path.split("/")[-1], params["sigma_a"] * altitude + params["sigma_b"], 
            params["dog_thresh"], params["max_litter_size_tresh_perc"]
        )

        detected_boxes = convert_from_pool_to_abs_coords(detected_boxes, pool)

        # print(f"Found {len(detected_boxes)} boxes")

        if verb:
            img_to_show = cv2.imread(img_path)

            draw_rectangles(img_to_show, gt_boxes, (0, 255, 0))
            draw_rectangles(img_to_show, detected_boxes, (255, 0, 0))

            plt.imshow(img_to_show)
            plt.show()

        all_gt_boxes.append(gt_boxes) 
        all_pred_boxes.append(detected_boxes) 

    # Compute mAP for the current set of images
    return mean_ap(all_gt_boxes, all_pred_boxes, np.arange(0.4, 1.01, 0.1), verb)


def read_bboxes(label_path: str, image_width: int, image_height: int) -> List[Rectangle]:
    bboxes = []
    with open(label_path, "r") as f:
        for line in f:
            class_id, center_x, center_y, width, height = map(float, line.split())
            # class_id = int(class_id)
            
            # Convert YOLO format to pixel values
            xl = int((center_x - (width / 2)) * image_width)
            xr = int((center_x + (width / 2)) * image_width)
            yt = int((center_y - (height / 2)) * image_height)
            yb = int((center_y + (height / 2)) * image_height)

            bboxes.append(Rectangle(xl, yt, xr, yb))
    
    return bboxes


@hydra.main(config_path="../conf", config_name="annotated", version_base=None)
def optimize_params_dog(cfg):
    # get images
    files = os.listdir(os.path.join(cfg.paths.base, "images", "train"))

    filtered_files = {
        '_'.join(file.split('_')[:4])  # Get the first 4 parts of the filename
        for file in files
    }

    images = [os.path.join(cfg.paths.base, "images", "train", f"{x}_{cfg.paths.img_type}.{cfg.paths.extension}") for x in filtered_files]

    # get image width and height
    image = cv2.imread(images[0])
    image_height, image_width = image.shape[:2]
    print(f"Image 0 width: {image_width} height: {image_height}. Assuming all the images are like it.")

    # get ground truth bounding boxes
    label_paths = [os.path.join(cfg.paths.base, "labels_joined", "train", f"{x}_{cfg.paths.img_type}.txt") for x in filtered_files]
    labels = [read_bboxes(path, image_width, image_height) for path in label_paths]

    def objective(trial):
        params = {
            "dog_thresh": trial.suggest_float('dog_thresh', 0, 0.5),
            "max_litter_size_tresh_perc": trial.suggest_float('max_litter_size_tresh_perc', 0, 0.5), # as percentage of the pool width
            "sigma_a": trial.suggest_float('sigma_a', -1/3, 0), # for the sigma = sigma_a * alt + sigma_b
            "sigma_b": trial.suggest_float('sigma_b', 0,10)
        }

        if params['sigma_a'] * 30 + params['sigma_b'] <= 0:
            return float("-inf")

        return evaluate_detector(params, images, labels, verb=False)

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna_db.sqlite3",
        study_name="dog_contours_AP@[0.4:1]",
    )
    study.optimize(objective, n_trials=100)

    print("best params: ", study.best_params)


if __name__ == "__main__":
    optimize_params_dog()

