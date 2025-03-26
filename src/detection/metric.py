from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator #models/yolo/detect/val.py
from ultralytics.utils.metrics import box_iou

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from detection.yolo import read_ground_truth

ds_path = "/home/anna/Datasets/created/pool-form1_pool-3-channels_random/"
yaml =  f"{ds_path}/pool-form1_pool-3-channels_random.yaml" # "detection/data.yaml"
split = "val" #"example"

class MyValidator(DetectionValidator):
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
        return self.match_predictions(detections[:, 5], gt_cls, iou)

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
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

# Load model
model = YOLO("/home/anna/code/multispectral/src/detection/pool-form1_pool-3-channels_random_best.pt")

# # show image
# results = model.predict(source=f"{ds_path}images/{split}/", conf=0.3, save=False)
# gt = read_ground_truth(results)

# # Add annots to each image
# for result, gt in zip(results, gt):
#     result.names[0] = ""
#     img = result.plot(conf=True, line_width=1, font_size=5)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     for box in gt:
#         x1, y1, x2, y2 = box
#         img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

# plt.imshow(img)
# plt.show()

# get metrics
our_results = model.val(data=yaml, validator=MyValidator)
ap50_our = our_results.results_dict["metrics/mAP50(B)"]
ap5095_our = our_results.results_dict['metrics/mAP50-95(B)']


results = model.val(data=yaml, validator=DetectionValidator)
ap50_yolo = results.results_dict["metrics/mAP50(B)"]
ap5095_yolo = results.results_dict['metrics/mAP50-95(B)']


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



