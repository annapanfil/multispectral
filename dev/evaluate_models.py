%matplotlib inline

from ultralytics import YOLO
import os
import yaml
import sys
from matplotlib import pyplot as plt
from math import ceil
import numpy as np
import seaborn as sns
import cv2

sys.path.append("..")
from dev.display import draw_yolo_boxes
from src.processing.consts import DATASET_BASE_PATH
from train.yolo import show_gt_and_pred, read_ground_truth
from dev.utils import print_results_yolo, get_fp_fn_images, save_img_and_log

MODELS = { 
    "form8_sea":  "../models/sea-form8_sea_aug-random_best.pt",
    "form8_sea_onnx": "../models/sea-form8_sea_aug-random_best.onnx",    
    "form8_mandrac": "../models/mandrac_form8_random-sub_best.pt",
    "form8_mandrac-hamburg": "../models/mandrac-hamburg_form8_random-sub_best.pt",
    "form2_mandrac": "../models/mandrac_form2_random-sub_best.pt",
    "form2_mandrac-hamburg": "../models/mandrac-hamburg_form2_random-sub_best.pt",
}

CONFIDENCE = 0.42
OUT_PATH = "../out/summary"

ONE_CLASS_DIR = "one_class"
TWO_CLASS_DIR = "two_classes"
PILE_DIR = "pile_only"
LITTER_DIR = "litter_only"

DATASET_FORM8 = {
    "name": "hamburg mapping N G form8",
    "dataset_path": f"{DATASET_BASE_PATH}/created/hamburg_mapping-form8-val",
    "config_name": "hamburg_mapping-form8-val.yaml",
    "two_class": True,
}

DATASET_FORM8_INV = {
    "name": "hamburg mapping form8 G N (inverse)",
    "dataset_path": f"{DATASET_BASE_PATH}/created/hamburg_mapping-inverse-form8-val",
    "config_name": "hamburg_mapping-inverse-form8-val.yaml",
    "two_class": True,
}

DATASET_FORM2 = {
    "name": "hamburg mapping N G form2",
    "dataset_path": f"{DATASET_BASE_PATH}/created/hamburg_mapping-form2-val",
    "config_name": "hamburg_mapping-form2-val.yaml",
    "two_class": True,
}

DATASET_DUBROVNIK = {
    "name": "dubrovnik N G form8",
    "dataset_path": f"{DATASET_BASE_PATH}/created/mandrac_form8_random-sub",
    "config_name": "mandrac_form8_random-sub.yaml",
    "two_class": False
}

DATASETS = [DATASET_FORM8, DATASET_FORM8_INV, DATASET_FORM2]

def change_labels(dataset_path, dir):
    if os.path.exists(f"{dataset_path}/{dir}"):
        !rm -rf {dataset_path}/labels
        !ln -s {dataset_path}/{dir} {dataset_path}/labels
        print(f"labels changed to {dir}")
    else:
        print(f"labels directory {dataset_path}/{dir} does not exist, keeping old labels")

def show_annotations(dataset_path, config_name, output_dir, filename, all_channels=False):
    for split in ("train", "val", "test"):
        if not os.path.exists(f"{dataset_path}/images/{split}"):
            print(f"Images directory for {split} does not exist in {dataset_path}/images/")
            continue
        files = os.listdir(f"{dataset_path}/images/{split}")
        if len(files) == 0:
            print(f"No images found in {dataset_path}/images/{split}")
            continue

        if not all_channels:
            # RGB or channel 0 only
            rgb_files = [f for f in files if f.endswith(('_RGB.png'))]
            files = rgb_files if rgb_files else [f for f in files if f.endswith(('_ch0.tif', '_ch0.tiff'))]

            if len(files) == 0:
                print(f"No image with RGB or channel 0 found in {dataset_path}/images/{split}")
                continue

        files.sort()

        n_cols = 7 if len(files) > 7 else len(files)
        n_rows = ceil(len(files) / n_cols)

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*4, n_rows*3))

        with open(f"{dataset_path}/{config_name}", "r") as file:
            config = yaml.safe_load(file)

        for fn, ax in zip(files, axs.flatten()):
            ax.set_title(fn)
            ax.axis("off")
            image, annots = draw_yolo_boxes(f"{dataset_path}/images/{split}/{fn}", 
                            f"{dataset_path}/labels/{split}/{os.path.splitext(fn)[0] + '.txt'}",
                            config["names"], display=False)
            ax.imshow(image)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}_annots-{split}.jpg")
        plt.close()

        print(f"Saved annotations for {split} to {output_dir}/{filename}_annots-{split}.jpg")


# # Evaluate model

# %%
def save_images(pred_results, dataset, output_dir, title="", confidence=CONFIDENCE, show_empty=False):
    # save gt and pred images
    fp_image, fn_image = get_fp_fn_images(pred_results, confidence)

    change_labels(dataset['dataset_path'], TWO_CLASS_DIR)
    gt, classes = read_ground_truth(pred_results)
    image = show_gt_and_pred(pred_results, gt, n_cols=3, gt_classes=classes, show_empty=show_empty, verb=False) #6

    save_img_and_log(image, f"{output_dir}/preds-gt_{title.replace(' ', '-')}_conf-{confidence:.2f}.jpg")
    save_img_and_log(fp_image, f"{output_dir}/fps_{title.replace(' ', '-')}_conf-{confidence:.2f}.jpg")
    save_img_and_log(fn_image, f"{output_dir}/fns_{title.replace(' ', '-')}_conf-{confidence:.2f}.jpg")

def evaluate_model(model_name, dataset, names=[], two_class=True, confidence=None, postfix="", our_litter_only=False, onnx=False, split="val"):
    model = YOLO(MODELS[model_name])
    ds_path = dataset['dataset_path']
    output_dir = f"{OUT_PATH}/{model_name}_model{postfix}_results"
    os.mkdir(output_dir)

    label_dirs = [ONE_CLASS_DIR, PILE_DIR, LITTER_DIR] if two_class else ["-"]
    if our_litter_only:
        label_dirs = [PILE_DIR]
        names = ["Our litter only"]
    for label_dir, name in zip(label_dirs, names):
        if two_class:
            change_labels(ds_path, label_dir)

        val_results = model.val(data=f"{ds_path}/{dataset['config_name']}", plots=True, verbose=False, imgsz=(800, 608), split=split, iou_thres=0.5)

        if not confidence:
            # get optimal confidence threshold for the highest F1 score
            opt_confidence = True
            confidence_thresholds = val_results.curves_results[1][0]
            f1_scores = val_results.curves_results[1][1]
            optimal_idx = np.argmax(f1_scores)
            confidence = confidence_thresholds[optimal_idx]
            print(f"Optimal confidence threshold for {name} is {confidence:.2f}")
        else:
            opt_confidence = False

        pred_results = model.predict(source=f"{ds_path}/images/{split}", save=False, conf=confidence, verbose=False, imgsz=(800, 608))
        print_results_yolo(val_results, name, filename=f"{output_dir}/metrics_conf-{confidence:.2f}.txt", confidence=None if opt_confidence else confidence)
        save_images(pred_results, dataset, output_dir, name, confidence=confidence, show_empty=True)

#################################################################

def evaluate_onnx_model(model_path, dataset, names, two_class=False, confidence=CONFIDENCE):
    import onnxruntime as ort
    from src.shapes import Rectangle
    from src.utils import greedy_grouping
    from dev.utils import get_tp_idx

    ds_path = dataset['dataset_path']
    label_dirs = [ONE_CLASS_DIR, PILE_DIR, LITTER_DIR] if two_class else ["-"]
    for label_dir, name in zip(label_dirs, names):
        if two_class:
            change_labels(ds_path, label_dir)

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # TPs, FPs, FNs = 0, 0, 0

    gt_boxes = []
    pred_boxes = []
    for img_path in os.listdir(f"{ds_path}/images/val"):
        image = cv2.imread(f"{ds_path}/images/val/{img_path}")
        img_height, img_width, img_channels = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Add batch dimension: (1, C, H, W)
        input = np.transpose(image, (2, 1, 0))
        input = np.expand_dims(input, axis=0)
        # Normalize input
        input = input.astype(np.float32) / 255.0


        # Detection
        results = session.run(None, {input_name: input})

        bbs_img = []
        for pred_bbs in results[0]:
            for bb in pred_bbs:
                x1, y1, x2, y2, conf, cls = bb
                # filter if needed: e.g., if conf > 0.5
                if conf > CONFIDENCE:
                    rect = Rectangle(y1, x1, y2, x2, cls)
                    bbs_img.append(rect)        

        # Merging
        merged_bbs, _, _ = greedy_grouping(bbs_img, image.shape[:2], resize_factor=1.5, visualize=False)

        gt_boxes_img = []

        img_nr = img_path.split('.')[0]
        with open(f"{ds_path}/labels/val/{img_nr}.txt", 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_name, center_x, center_y, width, height = map(float, parts[:5])

                rectangle = Rectangle.from_yolo(center_x, center_y, width, height, class_name, (img_height, img_width))
                gt_boxes_img.append(rectangle)

        gt_boxes.append(gt_boxes_img)
        pred_boxes.append(merged_bbs)
        # idx_gt, tps = get_tp_idx(gt_boxes_img, merged_bbs)

        # fps = [pred_bbs[i] for i in range (len(pred_bbs)) if i not in tps]
        # fns = [gt_boxes_img[i] for i in range(len(gt_boxes_img)) if i not in idx_gt]
        
        # TPs += len(tps)
        # FPs += len(fps)
        # FNs += len(fns)

    def calculate_iou(box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        Format: [x1, y1, x2, y2] (top-left and bottom-right coordinates)
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area
        return iou

    def calculate_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
        """
        Calculate precision, recall, f1, and AP50 for object detection
        
        Args:
            gt_boxes: List of ground truth boxes for all images [[x1,y1,x2,y2], ...]
            pred_boxes: List of predicted boxes for all images [[x1,y1,x2,y2,confidence], ...]
            iou_threshold: Threshold for considering a detection correct (0.5 for AP50)
        
        Returns:
            Dictionary containing precision, recall, f1, and AP50
        """
        # Sort predictions by confidence (descending)
        pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
        
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        precision_recall = []
        
        # For AP calculation
        used_gt_indices = set()
        
        for pred_box in pred_boxes_sorted:
            # Extract coordinates (ignore confidence for IoU calculation)
            pred_coords = pred_box[:4]
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in used_gt_indices:
                    continue
                iou = calculate_iou(pred_coords, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Determine if prediction is true positive or false positive
            if best_iou >= iou_threshold and best_gt_idx not in used_gt_indices:
                tp += 1
                fn -= 1  # Since we've matched a ground truth
                used_gt_indices.add(best_gt_idx)
                precision_recall.append((tp / (tp + fp), tp / (tp + fn)))
            else:
                fp += 1
                precision_recall.append((tp / (tp + fp), tp / (tp + fn)))
        
        # Calculate final metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AP50 (area under precision-recall curve)
        ap = 0
        if precision_recall:
            # Sort by recall
            precision_recall_sorted = sorted(precision_recall, key=lambda x: x[1])
            recalls = [x[1] for x in precision_recall_sorted]
            precisions = [x[0] for x in precision_recall_sorted]
            
            # Make precision monotonically decreasing
            for i in range(len(precisions)-2, -1, -1):
                precisions[i] = max(precisions[i], precisions[i+1])
            
            # Calculate AP
            ap = 0
            for i in range(1, len(recalls)):
                if recalls[i] != recalls[i-1]:
                    ap += (recalls[i] - recalls[i-1]) * precisions[i]
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'AP50': ap
        }

    metrics = calculate_metrics(gt_boxes, pred_boxes)

    return metrics

    # precision = TPs / (TPs + FPs) if (TPs + FPs) > 0 else 0
    # recall = TPs / (TPs + FNs) if (TPs + FNs) > 0 else 0
    # f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # return {"precision": precision, "recall": recall, "f1": f1, "TPs": TPs, "FPs": FPs, "FNs": FNs}

######################################################################################
if __name__ == "__main__":
    
    # show annotations for all datasets
    # for dataset in DATASETS:
    #     print(dataset['name'])
    #     change_labels(dataset["dataset_path"], TWO_CLASS_DIR)
    #     show_annotations(dataset["dataset_path"], dataset["config_name"], f"{OUT_PATH}/annots", dataset["name"].replace(' ', '-'), all_channels=True)

    ## EVALUATE OLD MODEL (USED IN HAMBURG)

    # model_name = "form8_sea"
    # print("EVALUATING MODEL ON N G form8 DATASET")
    # evaluate_model(model_name, DATASET_FORM8, ["all present litter", "only our litter", "only present litter"], postfix="-conf", our_litter_only=True)

    # print("EVALUATING MODEL ON form8 G N DATASET")
    # evaluate_model(model_name, DATASET_FORM8_INV, ["all present litter (inverse)", "only our litter (inverse)", "only present litter (inverse)"], postfix="-inv")

    # Manually move images to 167 to different directory
    # print("EVALUATING MODEL ON N G form8 DATASET WITHOUT THE SHORE")
    # evaluate_model(model_name, DATASET_FORM8, ["all present litter (no shore)", "only our litter (no shore)", "only present litter (no shore)"], postfix="-no-shore")

    ### ONNX EXPORTED
    # evaluate_model("form8_sea_onnx", DATASET_FORM8, ["all present litter", "only our litter", "only present litter"], onnx=True)
    # evaluate_onnx_model("../models/model.onnx", DATASET_FORM8, ["all present litter", "only our litter", "only present litter"], two_class=False, confidence=CONFIDENCE)
    # evaluate_onnx_model("../models/sea-form8_sea_aug-random_best.onnx", DATASET_FORM8, ["all present litter", "only our litter", "only present litter"], two_class=False, confidence=CONFIDENCE)

    ### EVALUATE NEW MODELS

    # for model_name in ("form8_mandrac", "form8_mandrac-hamburg"):
    #     print(f"EVALUATING {model_name}")
    #     evaluate_model(model_name, DATASET_FORM8, ["all present litter", "only our litter", "only present litter"], postfix="-test")   

    # for model_name in ("form2_mandrac", "form2_mandrac-hamburg"):
    #     print(f"EVALUATING {model_name}")
    #     evaluate_model(model_name, DATASET_FORM2, ["all present litter", "only our litter", "only present litter"])   

    ### EVALUATE MODELS FOR THE PAPER
    print(f"EVALUATING form8_mandrac on all mandrač datasets")
    evaluate_model("form8_mandrac", DATASET_DUBROVNIK, two_class=False, names=["all present litter"], postfix="-dubrovnik-conf-0.42-iou0.5-own", split="test")

    # print(f"EVALUATING form8_mandrac on Hamburg mapping dataset")
    # evaluate_model("form8_mandrac", DATASET_FORM8, our_litter_only=True, postfix="-our-litter-conf-0.42", confidence=0.42)

