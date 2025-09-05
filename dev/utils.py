import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from train.yolo import read_ground_truth
from typing import List
from src.shapes import Rectangle
import ultralytics
import cv2
from src.utils import greedy_grouping


def print_results_yolo(val_results, title="", filename=None, confidence=None):
    res_title = f"{title.upper()} VALIDATION RESULTS"

    if not confidence:
        precision = val_results.box.mp
        recall = val_results.box.mr
        f1 = val_results.box.f1[0]
    else:
        # get index for conf
        confidence_thresholds = val_results.curves_results[1][0]
        confidence_idx = np.argmin(np.abs(confidence_thresholds - confidence))
        recall = val_results.curves_results[3][1].T[confidence_idx][0]
        precision = val_results.curves_results[2][1].T[confidence_idx][0]
        f1 = val_results.curves_results[1][1].T[confidence_idx][0]

    res_string = f"""
{res_title}
{'=' * len(res_title)}

Results {"for the confidence " + str(confidence) if confidence else ""}:
mAP50: {val_results.box.ap50[0]:.3f}
Precision: {precision:.3f}
Recall: {recall:.3f}
F1: {f1:.3f}
"""

    print(f"\n{res_string}")
    
    if filename:
        with open(filename, "a") as f:
            f.write(res_string)

    _, axs = plt.subplots(1, 4, figsize=(16, 4))

    for i, ax in enumerate(axs.ravel()):
        
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

    plt.tight_layout()
    plt.savefig(f"{filename}_curves.png")
    plt.show()

def get_matches(gt_bbs: List[Rectangle], pred_bbs: List[Rectangle], iou_threshold=0.5):
    """
    Parameters:
        - gts: list of ground truth boxes for one image
        - predictions: list of predicted boxes for one image
    Return:
        - pred_correct: list of booleans. True indicates that the predicted box was matched with any of ground truth box (1 - true positive, 0 - false positive)
        - gt_detected: list of booleans. True indicates that the ground truth box was matched with any of predicted boxes (0 - false negative)
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

        return pred_correct, gt_detected
  

def get_tp_idx(gt_bbs: List[Rectangle], pred_bbs: List[Rectangle], iou_threshold=0.5):
    """
    Parameters:
        - gts: list of ground truth boxes for one image
        - predictions: list of predicted boxes for one image
    Return:
        - idx_gt: indices of ground truth boxes that were matched with predicted boxes (correctly detected)
        - idx_pred: indices of predicted boxes that were matched with ground truth boxes (true positives)
    """

    pred_correct, gt_detected = get_matches(gt_bbs, pred_bbs, iou_threshold)
    if pred_correct == [] or gt_detected == []:
        return [], []

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
    tps_list = []

    fps_images = set()
    fns_images = set()
    tps_images = set()

    for i in range(len(pred_results)):
        preds = pred_results[i]
        gt = gts[i]
        # for 1 image
        pred_bbs = [Rectangle(*box.xyxy[0]) for box in preds.boxes if box.conf > confidence]
        gt_bbs = [Rectangle(*box) for box in gt]

        img = preds.orig_img
        idx_gt, tps = get_tp_idx(gt_bbs, pred_bbs)

        fps = [pred_bbs[i] for i in range (len(pred_bbs)) if i not in tps]
        fns = [gt_bbs[i] for i in range(len(gt_bbs)) if i not in idx_gt]
        tps = [pred_bbs[i] for i in tps]

        if fps:
            fps_ims = [img[fp.y_b:fp.y_t, fp.x_l:fp.x_r] for fp in fps]
            max_height = max(p.shape[0] for p in fps_ims)
            padded = [cv2.copyMakeBorder(p, 0, max_height - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for p in fps_ims]
            fps_list.append(cv2.hconcat(padded))
            fps_images.add(pred_results[i].path.rsplit("/")[-1])
        
        if fns:
            fns_ims = [img[fn.y_b:fn.y_t, fn.x_l:fn.x_r] for fn in fns]
            max_height = max(p.shape[0] for p in fns_ims)
            padded = [cv2.copyMakeBorder(p, 0, max_height - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for p in fns_ims]
            fns_list.append(cv2.hconcat(padded))
            fns_images.add(pred_results[i].path.rsplit("/")[-1])

        if tps:
            tps_ims = [img[tp.y_b:tp.y_t, tp.x_l:tp.x_r] for tp in tps]
            max_height = max(p.shape[0] for p in tps_ims)
            padded = [cv2.copyMakeBorder(p, 0, max_height - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for p in tps_ims]
            tps_list.append(cv2.hconcat(padded))
            tps_images.add(pred_results[i].path.rsplit("/")[-1])

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
    
    if not tps_list:
        print("No true positives found.")
    else:
        max_width = max(p.shape[1] for p in tps_list)
        padded = [cv2.copyMakeBorder(p, 0, 0, 0, max_width - p.shape[1], cv2.BORDER_CONSTANT, value=0) for p in tps_list]
        tps_image = cv2.vconcat(padded)
        
    print(f"Found {len(fps_images)} in images with false positives: {', '.join(fps_images)}")
    print(f"Found {len(fns_images)} in images with false negatives: {', '.join(fns_images)}")
    print(f"Found {len(tps_images)} in images with true positives: {', '.join(tps_images)}")
    
    return fps_image, fns_image, tps_image


def get_fp_fn_tp(gt_bbs: List[Rectangle], pred_bbs: List[Rectangle], iou_threshold=0.5):
    pred_correct, gt_detected = get_matches(gt_bbs, pred_bbs, iou_threshold)

    tp = sum(pred_correct)
    fp = len(pred_bbs) - tp
    fn = len(gt_bbs) - sum(gt_detected)

    return tp, fp, fn


def get_metrics(pred_results: ultralytics.engine.results.Results, confidence=0.5, iou_threshold=0.5):
    """
    Parameters:
       - pred_results: ultralytics engine results object
       - confidence: confidence threshold for predictions
       - iou_threshold: IoU threshold for true positive detection
    Return:
        - precision, recall, f1_score
    """
   
    gts, _ = read_ground_truth(pred_results)

    tps = 0
    fps = 0
    fns = 0
    for pred, gt  in zip(pred_results, gts):
        pred_bbs = [Rectangle(*box.xyxy[0]) for box in pred.boxes if box.conf > confidence] # why they are not NMSed completely?
        gt_bbs = [Rectangle(*box) for box in gt]

        tp, fp, fn = get_fp_fn_tp(gt_bbs, pred_bbs, iou_threshold)
        tps += tp
        fps += fp
        fns += fn

    precision = tps / (tps + fps) if tps + fps > 0 else 1.0
    recall = tps / (tps + fns) if tps + fns > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1_score


def get_curves(model, ds, confidences=np.linspace(0,1,100), iou_threshold=0.5):
    """
    Get precision-recall, and precision, recall and f1-score vs confidence curves image, together with ap50
    Parameters:
        pred_results: All predictions for confidence >= 0
        confidences: List of confidence thresholds to calculate metrics for
        iou_threshold: IoU threshold for true positive detection
    Returns:
        - plot: matplotlib figure with precision-recall and precision, recall and f1-score vs confidence curves
        - ap50: Average precision at IoU threshold 0.5
    """

    precisions = []
    recalls = []
    f1s = []
    i = 0
    for confidence in confidences:
        if i % 10 == 0:
            print(f"Calculating metrics for confidence {confidence:.2f} ({i+1}/{len(confidences)})")
        i += 1
        pred_results =  model.predict(source=ds, save=False, conf=confidence, verbose=False, imgsz=(800, 608), iou=0.5)
        precision, recall, f1 = get_metrics(pred_results, confidence=confidence, iou_threshold=iou_threshold)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    plot = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.plot(confidences, precisions, label='Precision-Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Precision')
    plt.title('Precision-Confidence')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.subplot(1, 4, 2)
    plt.plot(confidences, recalls, label='Recall-Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Recall')
    plt.title('Recall-Confidence')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.subplot(1, 4, 3)
    plt.plot(confidences, f1s, label='F1-Score-Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('F1 Score')
    plt.title('F1-Confidence')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.subplot(1, 4, 4)
    plt.plot(recalls, precisions, label='Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.xlim(0, 1)
    plt.ylim(0, 1)  

    ap50 = np.trapz(recalls, precisions)

    return plot, ap50


def save_img_and_log(image, path):
    if image is None or len(image) == 0:
        print(f"Image is empty, not saving to {path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    print(f"Saved image to {path}")