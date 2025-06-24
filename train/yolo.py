#  ls datasets/ | grep -v -e piles* | xargs -I {} python3 yolo.py -d {}

from matplotlib import pyplot as plt
from clearml import Task
from ultralytics import YOLO
import click
import cv2
import numpy as np

from src.processing.consts import DATASET_BASE_PATH

def read_ground_truth(results):
    # read ground truth boxes
     gt_boxes = []
     gt_classes = []
     for res in results:
          label_path = res.path.replace("images", "labels").replace('.png', '.txt')
          with open(label_path, 'r') as f:
               boxes = []
               classes = []
               for line in f:
                    cl, cx, cy, w, h = map(float, line.split())
                    x1 = int((cx - w / 2) * 800)
                    y1 = int((cy - h / 2) * 608)
                    x2 = int((cx + w / 2) * 800)
                    y2 = int((cy + h / 2) * 608)
                    boxes.append((x1, y1, x2, y2))
                    classes.append(int(cl))
               gt_boxes.append(boxes)
               gt_classes.append(classes)
     return gt_boxes, gt_classes

def show_gt_and_pred(results, gt_boxes, additional_boxes = [], n_cols=4, show_empty=True, gt_classes=None):
     if additional_boxes == []: additional_boxes = [[] for _ in range(len(results))]
     temp_images = []

     if gt_classes: 
          palette=[tuple(int(c * 255) for c in color) for color in plt.cm.tab20.colors]
     else: 
          gt_classes = [None] * len(gt_boxes)

     # Add annots to each image
     for result, gt, abb, gtc in zip(results, gt_boxes, additional_boxes, gt_classes):
          if show_empty == False and len(result.boxes) == 0 and len(gt) == 0 and len(abb) == 0:
               print(f"No boxes present for {result.path}. Skipping...")
               continue
          result.names[0] = ""
          img = result.plot(conf=True, line_width=1, font_size=5)

          for j, box in enumerate(gt):
               if gt_classes is not None:
                    class_n = gtc[j]
                    color = palette[class_n%20]
               else: 
                    color = (0, 255, 0)
               x1, y1, x2, y2 = box
               img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

          for box in abb:
               x1, y1, x2, y2 = box
               img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

          temp_images.append(img)

     # Combine them into grid
     n_rows = (len(results) + n_cols - 1) // n_cols
     grid_img = None
     for i in range(n_rows):
          row_images = temp_images[i * n_cols:(i + 1) * n_cols]
          if not row_images:
               continue
          if len(row_images) < n_cols:
               row_images += [np.zeros_like(row_images[0])] * (n_cols - len(row_images)) # empty images

          row_img = cv2.hconcat([cv2.resize(img, (800, 608)) for img in row_images])
          if grid_img is None:
               grid_img = row_img
          else:
               grid_img = cv2.vconcat([grid_img, row_img])

     return cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)


@click.command()
@click.option("--dataset", "-d", help="Name of the dataset in the {DATASET_BASE_PATH}/created directory")
@click.option("--experiment_name", "-n", default="indices", help="Project name in clear ml")
@click.option("--task_name", "-t", default="", help="Addition to the task name (default is dataset name)")
@click.option("--tag", multiple=True, help="Tag for the Clear ML experiment")
@click.option("--version", "-v", default="yolov10n", help="YOLO version")
def main(dataset, experiment_name, task_name, tag, version):
     print("Running task for", dataset)
     Task.set_credentials(
          api_host="https://api.clear.ml",
          web_host="https://app.clear.ml",
          files_host="https://files.clear.ml",
          key='R3KF8584CYHXTMATKOBMGBTKOR0GL8',
          secret='iJTV32010g3PEL3Euv0b1cZaClWTuu3dlCx4oQOyLCoYo4rqk2RsSIaVePr-AX-yR7A'
     )

     task = Task.init(project_name=experiment_name, task_name=f"{dataset} {task_name}")
     task.add_tags(list(tag))

     model = YOLO(f"{version}.pt")  # load a pretrained model
     epochs = 62
     batch = 4
     lr = 0.0004291291727623719
     momentum = 0.0004291291727623719
     decay = 0.0009051275502118082
     optimizer = "AdamW"
     yaml_path = f"{DATASET_BASE_PATH}/created/{dataset}" 
     results = model.train(data=f"{yaml_path}/{dataset}.yaml", epochs=epochs, imgsz=[800, 608], batch=8)

     results = model.val(split="train", save=False) 
     print(results.results_dict)
     task.logger.report_scalar("train metrics", "train/mAP50", results.box.map50, epochs)
     task.logger.report_scalar("train metrics", "train/mAP50-95", results.box.map, epochs)
     task.logger.report_scalar("train metrics", "train/precision", results.box.mp, epochs)
     task.logger.report_scalar("train metrics", "train/recall", results.box.mr, epochs)

     for split in ("train", "val"):

          results = model.predict(source=f"{DATASET_BASE_PATH}/created/{dataset}/images/{split}", conf=0.3, save=False)
          gt, _ = read_ground_truth(results)
          image = show_gt_and_pred(results, gt)
          task.logger.report_image(title="Predictions Grid", series=split, image = image)

     task.close()


if __name__ == "__main__":
     main()