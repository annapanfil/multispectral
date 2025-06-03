from matplotlib import patches
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator #models/yolo/detect/val.py
from ultralytics.utils.metrics import box_iou

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from train.yolo import read_ground_truth, show_gt_and_pred


def is_inside(bb, bbs, idx_filter=None):
    """
    Check if bounding box is inside any of the given bounding boxes.

    Args:
        bb (int): Bounding box index.
        bbs (torch.Tensor): Tensor of shape (N, 4) representing bounding boxes where each bounding box is of the
            format: (x1, y1, x2, y2).
        idx_filter (torch.Tensor): Tensor of shape (M,) representing indices of bounding boxes to consider.

    Returns:
        int: index of the outside bounding box, None if not found.
    """
    for i, bbox in enumerate(bbs):
        if i in idx_filter:
            if bbox[0] <= bb[0] and bbox[1] <= bb[1] and bbox[2] >= bb[2] and bbox[3] >= bb[3]:
                return i
    return None
        

class MyValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks={}):
        """Initialize detection model with necessary variables and settings.""" 
        
        _callbacks["on_val_batch_end"].append(self._add_new_rows)
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

        self.additional_pred_bbs = []
        self.additional_confidences = []
        self.additional_pred_cls = []
        
    def _add_new_rows(self, _):
        print("\nAdding new rows\n")
        if self.additional_pred_bbs != []:
            self.additional_pred_bbs = torch.cat(self.additional_pred_bbs, dim=0)  # Stack along dim=0
            self.stats["tp"][-1] = torch.cat((self.stats["tp"][-1], self.additional_pred_bbs), dim=0)
            self.stats["conf"][-1] = torch.cat((self.stats["conf"][-1], torch.cat(self.additional_confidences, dim=0)), dim=0)
            self.stats["pred_cls"][-1] = torch.cat((self.stats["pred_cls"][-1], torch.cat(self.additional_pred_cls, dim=0)), dim=0)

            self.additional_pred_bbs = []
            self.additional_confidences = []
            self.additional_pred_cls = []

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        correct, matches = self.match_predictions(detections[:, 5], gt_cls, iou)

        detections_thr = detections[detections[:, 4] >= 0.5] #TODO: change to lowest thresh
        matched_gt = matches[matches[:, 1] < detections_thr.shape[0]][:, 0]
        not_matched_gt = np.setdiff1d(np.arange(gt_cls.shape[0]), matched_gt)        

        additional_matches = []
        for bb in not_matched_gt:
            outside_bb = is_inside(gt_bboxes[bb], gt_bboxes, matched_gt) 
            if outside_bb is not None:
                # match the prediction from the outside_bb to the bb
                prediction_idx = matches[matches[:, 0] == outside_bb, 1][0]
                additional_matches.append(bb)
                self.additional_pred_bbs.append(correct[prediction_idx].unsqueeze(0))
                self.additional_confidences.append(detections[prediction_idx, 4].unsqueeze(0))
                self.additional_pred_cls.append(detections[prediction_idx, 5].unsqueeze(0))

        global images
        images.append(self.plot_detections(detections_thr[:, :4], gt_bboxes, matched_gt, additional_matches))

        return correct
    
    
    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
                if i == 0:
                    max_matches = matches
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device), max_matches

    def plot_detections(self, detections, gt_bboxes, matched_gt, additional_matches):
        img = np.ones((650, 800, 3), dtype=np.uint8) * 255
        for bb in detections:
            x1, y1, x2, y2 = bb
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

        for bb in gt_bboxes:
            x1, y1, x2, y2 = bb
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        for bb in matched_gt:
            x1, y1, x2, y2 = gt_bboxes[bb]
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)


        for bb in additional_matches:
            x1, y1, x2, y2 = gt_bboxes[bb]
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 192, 203), 1)

        return img


if __name__ == "__main__":
    ds_path = "/home/anna/Datasets/created/pool-form1_pool-3-channels_random/"
    FULL_DS = False

    if FULL_DS:
        #test whole val ds
        yaml = f"{ds_path}/pool-form1_pool-3-channels_random.yaml"
        split = "val"
    else: 
        yaml =  "detection/data.yaml"
        split = "example"

    images = [] # for showing them in the final image

    # Load model
    model = YOLO("/home/anna/code/multispectral/src/detection/pool-form1_pool-3-channels_random_best.pt")

    # get metrics
    results = model.val(data=yaml, validator=DetectionValidator)
    ap50_yolo = results.results_dict["metrics/mAP50(B)"]
    ap5095_yolo = results.results_dict['metrics/mAP50-95(B)']

    our_results = model.val(data=yaml, validator=MyValidator)
    ap50_our = our_results.results_dict["metrics/mAP50(B)"]
    ap5095_our = our_results.results_dict['metrics/mAP50-95(B)']

    plt.plot(np.linspace(50, 95, len(results.box.all_ap[0])), results.box.all_ap[0], color="red", 
            label=f"original (AP50={ap50_yolo:.3f}, AP50-95={ap5095_yolo:.3f})")
    plt.plot(np.linspace(50, 95, len(our_results.box.all_ap[0])), our_results.box.all_ap[0], color="green", 
            label=f"improved (AP50={ap50_our:.3f}, AP50-95={ap5095_our:.3f})")
    plt.xlabel("IoU Threshold (%)")
    plt.ylabel("AP")
    plt.title("AP vs IoU Threshold (utils/metrics.py/ap_per_class)")
    plt.legend()
    plt.show()

    print( "         |  AP50  | AP50-95")
    print( "---------------------------")
    print(f"Original | {ap50_yolo:.4f} | {ap5095_yolo:.4f}")
    print(f"Improved | {ap50_our:.4f} | {ap5095_our:.4f}")

    # show image
    n_cols = 6 if FULL_DS else 1
    n_rows = (len(images) + n_cols - 1) // n_cols

    grid_img = None
    for i in range(n_rows):
        row_images = images[i * n_cols:(i + 1) * n_cols]
        if len(row_images) < n_cols:
            row_images += [np.zeros_like(row_images[0])] * (n_cols - len(row_images)) # empty images

        row_img = cv2.hconcat([cv2.resize(img, (800, 608)) for img in row_images])
        if grid_img is None:
            grid_img = row_img
        else:
            grid_img = cv2.vconcat([grid_img, row_img])

    plt.imshow(grid_img)
    plt.axis('off')
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    legend_patches = [
        patches.Patch(color='green', label='Ground truth'),
        patches.Patch(color='blue', label='Prediction'),
        patches.Patch(color='red', label='Matched GT'),
        patches.Patch(color='pink', label='New match')
    ]
    plt.legend(handles=legend_patches, loc='lower right')
    plt.show()