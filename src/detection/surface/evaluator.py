from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
from surface_utils import Circle, Rectangle
import surface_utils
import cv2

class Evaluator():
    def __init__(self, debug=False, merging_conf_threshold=0.5):
        """
        Initialize the Evaluator class.
        Args:
            debug (bool): If True, shows debug photos.
            merging_conf_threshold (float): The images with confidence scores below this threshold will be merged via greedy grouping and finally shown. The rest is only used for calculating AP
        
        """
        self.debug = debug
        self.merging_conf_threshold = merging_conf_threshold

    def print_metrics(self, images, gt_labels, detections, confidences):
        """
        Print the evaluation metrics.
        """
        metrics = self.evaluate(images, gt_labels, detections, confidences)

        for metric in metrics.keys():
            print(f"{metric}: {metrics[metric]:.4f}")

    def evaluate(self, images, gt_labels, detections, confidences):
        """
        Evaluate the model using the provided dataset and split.
        Args:
            images (list): List of images.
            gt_labels (list): List of ground truth labels in yolo format.
            detections (list): List of detected objects.
            confidences (list): List of scores for each detection.
            proposals (list): List of proposals for each detection.
        Returns:
            dict: Evaluation results.
        """
        gt_labels = [self._yolo2rectangle(gt, image) for gt, image in zip(gt_labels, images)]

        metrics = {}

        all_pred_rectangles = []
        all_pred_confs = []
        if self.debug:
            merged_images = []

        for image, dets, confs in zip(images, detections, confidences):
            image_shape = image.shape[:2]
            circles = [Circle(int(d[0]), int(d[1]), int(d[2])) for d in dets]
            pred_rectangles, pred_confs, merged_img = self.greedy_grouping(circles, confs, image_shape, resize_factor=1.5, visualize=self.debug)
            
            if self.debug: merged_images.append(merged_img)
            
            # pred_rectangles = [Evaluator._circle2rectangle(circle) for circle in merged_circles]
            all_pred_rectangles.append(pred_rectangles)
            all_pred_confs.append(pred_confs)

      
            
        metrics["ap50"] = self.calculateAP(gt_labels, all_pred_rectangles, all_pred_confs)
        
        if self.debug:
            surface_utils.show_images(merged_images, "greedy grouping")

        # Show gt vs predictions
        for i in range(len(images)):
            for g in gt_labels[i]:
                cv2.rectangle(images[i], (int(g.x_l), int(g.y_b)), (int(g.x_r), int(g.y_t)), (0, 255, 0), 4)
            for j, p in enumerate(all_pred_rectangles[i]):
                if all_pred_confs[i][j] >= self.merging_conf_threshold:
                    cv2.rectangle(images[i], (int(p.x_l), int(p.y_b)), (int(p.x_r), int(p.y_t)), (255, 0, 0), 4)
                else:
                    cv2.rectangle(images[i], (int(p.x_l), int(p.y_b)), (int(p.x_r), int(p.y_t)), (0, 0, 255), 4)

        surface_utils.show_images(images, f"GT (green), predictions (red), predictions with confidence < {self.merging_conf_threshold} (blue)")

        return metrics

    def calculateAP(self, true_rects, pred_rects, confs=None, iou_threshold=0.5):
        """
        Calculate the precision for a given IoU threshold. Rectangles should be from the same class.
        Args:
            true_rectangles (list): List of true rectangles
            predicted_rectangles (list): List of predicted rectangles
            predicted_confidences (list): List of predicted confidences
            iou_threshold (float): IoU threshold
        Returns:
            float: AP result.
        """

        tp = self._get_tp(true_rects, pred_rects, iou_threshold)
        n_true_rects = sum(len(tr_img) for tr_img in true_rects)

        if confs is not None:
            # Sort by confidence score
            sorted_data = [sorted(zip(c, t), key=lambda x: -x[0]) for c, t in zip(confs, tp)]
            confs, tp = zip(*[zip(*d) if d else ([], []) for d in sorted_data])
            tp = list(map(list, tp))

        tp = [el for sublist in tp for el in sublist]    
        tp = np.array(tp)

        # Calculate precision and recall
        x, prec_values = np.linspace(0, 1, 1000), []
        ap, p_curve, r_curve = np.zeros(len(tp)), np.zeros(1000), np.zeros(1000)

        fpc = (1 - tp).cumsum(0)
        tpc = tp.cumsum(0)

        # Recall
        recall = tpc / (n_true_rects + 1e-16)  # recall curve
        # r_curve = np.interp(-x, -confs, recall, left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        # p_curve = np.interp(-x, -confs, precision, left=1)  # p at pr_score

        # AP from recall-precision curve
        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

        # prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

        return ap

    def _get_tp(self, true_rects, pred_rects, iou_threshold):
        """
        Calculate True Positives (TP) based on IoU threshold.
        Args:
            true_rects (list): List of lists of true rectangles for each image
            pred_rects (list): List of lists of predicted rectangles for each image
            iou_threshold (float): IoU threshold
        Returns:
            np.array: Binary array where TP is 1 and FP is 0.
        """
        tp = []

        for true_boxes, pred_boxes in zip(true_rects, pred_rects):
            iou_matrix = np.array([[Evaluator._iou_rectangles(p, g) for g in true_boxes] for p in pred_boxes])

            tp_image = np.zeros(len(pred_boxes))
            assigned_gt = set()  # track assigned ground truth boxes

            for i in range(len(pred_boxes)):
                matches = np.where(iou_matrix[i] >= iou_threshold)[0]
                if matches.size > 0:
                    best_match_idx = matches[np.argmax(iou_matrix[i, matches])]
                    if best_match_idx not in assigned_gt:
                        tp_image[i] = 1
                        assigned_gt.add(best_match_idx)
            
            tp.append(tp_image)

        if self.debug:
            print("TP for images: { ", end="")
            pos = 0
            for i, n in enumerate([len(p) for p in pred_rects]):
                tp_image = tp[pos:pos + n]
                pos += n
                print("{i}: {tp_s}".format(i=i, tp_s=int(sum(tp_image))), end=", ")
            print(" }")
        

        return tp

    @staticmethod
    def _iou_rectangles(rect1: Rectangle, rect2: Rectangle) -> float:
        """
        Calculate the Intersection over Union (IoU) for two rectangles.
        """
        x_left = max(rect1.x_l, rect2.x_l)
        x_right = min(rect1.x_r, rect2.x_r)
        y_bottom = max(rect1.y_b, rect2.y_b)
        y_top = min(rect1.y_t, rect2.y_t)

        if x_left >= x_right or y_bottom >= y_top:
            return 0.0

        
        intersection = max(0, x_right - x_left) * max(0, y_top - y_bottom)

        area1 = (rect1.x_r - rect1.x_l) * (rect1.y_t - rect1.y_b)
        area2 = (rect2.x_r - rect2.x_l) * (rect2.y_t - rect2.y_b)

        union = area1 + area2 - intersection
        return intersection / union


    def greedy_grouping(self, circles: List[Circle], confs: List, image_shape: Tuple, resize_factor=1.5, visualize=False) -> Tuple[List, np.array]:
        """
        Merge intersecting circles. Mean confidence of all circles in group is used as a group confidence.
        Args:
            circles (list): List of Circle objects.
            confs (list): List of confidence scores for each circle.
            image_shape (tuple): Shape of the image (height, width).
        """
        # filter circles to merge
        circles_to_merge = [circle for circle, conf in zip(circles, confs) if conf >= self.merging_conf_threshold]
     
        merged_circles_mask = np.zeros(image_shape, dtype=np.uint8)

        for circle in circles_to_merge:
            enlarged_circle = Circle(circle.x, circle.y, int(circle.r * resize_factor))
            current_circle_mask = Evaluator._create_circle_mask(enlarged_circle, image_shape)
            merged_circles_mask = cv2.bitwise_or(merged_circles_mask, current_circle_mask)

        _, contours, _ = cv2.findContours(merged_circles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if visualize:
            merged_circles_mask = cv2.cvtColor(merged_circles_mask, cv2.COLOR_GRAY2RGB)

        merged_rectangles = []
        merged_confs = []
        for contour in contours:
            # find original circles in the groups
            group_circles = [circle for circle in circles_to_merge if cv2.pointPolygonTest(contour, (circle.x, circle.y), False) >= 0]

            if group_circles:
                x_min = min(c.x - c.r for c in group_circles)
                y_min = min(c.y - c.r for c in group_circles)
                x_max = max(c.x + c.r for c in group_circles)
                y_max = max(c.y + c.r for c in group_circles)

                merged_rectangles.append(Rectangle(x_min, y_min, x_max, y_max))
                merged_confs.append(np.mean([confs[circles.index(c)] for c in group_circles]))

                if visualize:
                    cv2.rectangle(merged_circles_mask, (x_min, y_min), (x_max, y_max), (255, 0, 0), 5)

        # process other circles and merge
        for circle, conf in zip(circles, confs):
            if circle not in circles_to_merge:
                rectangle = Evaluator._circle2rectangle(circle)
                merged_rectangles.append(rectangle)
                merged_confs.append(conf)

                if visualize:
                    cv2.rectangle(merged_circles_mask, (rectangle.x_l, rectangle.y_b), (rectangle.x_r, rectangle.y_t), (0, 255, 0), 5)

        return merged_rectangles, merged_confs, merged_circles_mask

    @staticmethod
    def _create_circle_mask(circle: Circle, image_shape: Tuple[int, int]) -> np.array:
        """ Create a mask for one circle """
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.circle(mask, (circle.x, circle.y), circle.r, 255, -1)
        return mask

    @staticmethod
    def _circle2rectangle(circle: Circle) -> Tuple:
        """ Convert circle to rectangle """

        x, y, r = circle.x, circle.y, circle.r
        return Rectangle(max(x - r, 0), max(y - r, 0), x + r, y + r)
    
    @staticmethod
    def _yolo2rectangle(yolo_rects: List[Tuple], image: np.array) -> List[Rectangle]:
        """
        Convert YOLO rectangles to Rectangle objects.
        """
        h, w = image.shape[:2]
        return [Rectangle(int((rect[0] - rect[2] / 2) * w), int((rect[1] - rect[3] / 2) * h),
                          int((rect[0] + rect[2] / 2) * w), int((rect[1] + rect[3] / 2) * h)) for rect in yolo_rects]
