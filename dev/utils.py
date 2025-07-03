import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from train.yolo import read_ground_truth
from typing import List
from src.shapes import Rectangle
import ultralytics
import cv2


def print_results_yolo(val_results, title="", filename=None):
    res_title = f"{title.upper()} VALIDATION RESULTS"
    res_string = f"""
{res_title}
{'=' * len(res_title)}

Results:
mAP50: {val_results.box.ap50[0]:.3f}
mAP50-95: {val_results.box.ap[0]:.3f}
Precision: {val_results.box.mp:.3f}
Recall: {val_results.box.mr:.3f}
F1: {val_results.box.f1[0]:.3f}
"""

    print(f"\n{res_string}")
    
    if filename:
        with open(filename, "a") as f:
            f.write(res_string)

    _, axs = plt.subplots(1, 5, figsize=(20, 4))

    for i, ax in enumerate(axs.ravel()[:-1]):
        
        x = val_results.curves_results[i][0]
        y = val_results.curves_results[i][1].T.flatten()
        
        ax.plot(x, y)
        ax.set_xlabel(val_results.curves_results[i][2])
        ax.set_ylabel(val_results.curves_results[i][3])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(val_results.curves[i])

        window_size = 50
        y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        x_smooth = x[window_size//2 : -window_size//2 + 1]

        ax.plot(x_smooth, y_smooth, color='blue', label='Smoothed')

    ax = axs[4]
    cm = val_results.confusion_matrix.matrix.astype(int)
    class_names = ["Litter", "Background"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')

    plt.tight_layout()
    plt.show()




def get_tp_idx(gt_bbs: List[Rectangle], pred_bbs: List[Rectangle], iou_threshold=0.5):
    """
    Parameters:
        - gts: list of ground truth boxes for one image
        - predictions: list of predicted boxes for one image
    Return:
        - idx_gt: indices of ground truth boxes that were matched with predicted boxes (correctly detected)
        - idx_pred: indices of predicted boxes that were matched with ground truth boxes (true positives)
    """

    if len(gt_bbs) == 0 or len(pred_bbs) == 0:
        return [], []
    
    else:
        pred_correct = [0] * len(pred_bbs)
        gt_detected = [0] * len(gt_bbs)

        # Iterate over ground truth boxes
        for i, gt_box in enumerate(gt_bbs):
            any_correct = False
            # Find matching predicted boxes for each ground truth box
            for j, pred_box in enumerate(pred_bbs):
                if pred_correct[j] == 1:
                    continue  # Skip already matched boxes

                if gt_box.iou(pred_box) > iou_threshold:
                    pred_correct[j] = 1
                    any_correct = True

            if any_correct:
                gt_detected[i] = 1

        idx_gt = ([i for i, detected in enumerate(gt_detected) if detected])
        idx_pred = ([i for i, correct in enumerate(pred_correct) if correct])

    return idx_gt, idx_pred

def get_fp_fn_images(pred_results: ultralytics.engine.results.Results, confidence: float):
    """ Get false positives and false negatives images from the predictions.
        Returns:
            - `fps_image`: Image with combined false positives from all images. One row represents one image.
            - `fns_image`: Image with combined false negatives from all images. One row represents one image.
    """

    # save gt and pred images
    gts, classes = read_ground_truth(pred_results)

    fps_list = []
    fns_list = []

    for preds, gt in zip(pred_results, gts):
        # for 1 image
        pred_bbs = [Rectangle(*box.xyxy[0]) for box in preds.boxes if box.conf > confidence]
        gt_bbs = [Rectangle(*box) for box in gt]

        img = preds.orig_img
        idx_gt, tps = get_tp_idx(gt_bbs, pred_bbs)

        fps = [pred_bbs[i] for i in range (len(pred_bbs)) if i not in tps]
        fns = [gt_bbs[i] for i in range(len(gt_bbs)) if i not in idx_gt]

        if fps:
            fps_ims = [img[fp.y_b:fp.y_t, fp.x_l:fp.x_r] for fp in fps]
            max_height = max(p.shape[0] for p in fps_ims)
            padded = [cv2.copyMakeBorder(p, 0, max_height - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for p in fps_ims]
            fps_list.append(cv2.hconcat(padded))
        
        if fns:
            fns_ims = [img[fn.y_b:fn.y_t, fn.x_l:fn.x_r] for fn in fns]
            max_height = max(p.shape[0] for p in fns_ims)
            padded = [cv2.copyMakeBorder(p, 0, max_height - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for p in fns_ims]
            fns_list.append(cv2.hconcat(padded))

    if not fps_list:
        print("No false positives found.")
        fps_image = None
    else:
        max_width = max(p.shape[1] for p in fps_list)
        padded = [cv2.copyMakeBorder(p, 0, 0, 0, max_width - p.shape[1], cv2.BORDER_CONSTANT, value=0) for p in fps_list]
        fps_image = cv2.vconcat(padded)
        
    if not fns_list:
        print("No false negatives found.")
        fns_image = None
    else:
        max_width = max(p.shape[1] for p in fns_list)
        padded = [cv2.copyMakeBorder(p, 0, 0, 0, max_width - p.shape[1], cv2.BORDER_CONSTANT, value=0) for p in fns_list]
        fns_image = cv2.vconcat(padded)
        
    return fps_image, fns_image


def save_img_and_log(image, path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    print(f"Saved image to {path}")