#  ls datasets/ | grep -v -e piles* | xargs -I {} python3 yolo.py -d {}

from clearml import Task
from ultralytics import YOLO
import click
import cv2
import numpy as np

def read_ground_truth(results):
    # read ground truth boxes
     gt_boxes = []
     for res in results:
          label_path = res.path.replace("images", "labels").replace('.png', '.txt')
          with open(label_path, 'r') as f:
               boxes = []
               for line in f:
                    _, cx, cy, w, h = map(float, line.split())
                    x1 = int((cx - w / 2) * 800)
                    y1 = int((cy - h / 2) * 608)
                    x2 = int((cx + w / 2) * 800)
                    y2 = int((cy + h / 2) * 608)
                    boxes.append((x1, y1, x2, y2))
               gt_boxes.append(boxes)
     return gt_boxes

def save_gt_and_pred(task, results, gt_boxes, n_cols=4, split="val"):

     temp_images = []
     n_rows = (len(results) + n_cols - 1) // n_cols

     # Add annots to each image
     for result, gt in zip(results, gt_boxes):
          result.names[0] = ""
          img = result.plot(conf=True, line_width=1, font_size=5)

          for box in gt:
               x1, y1, x2, y2 = box
               img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

          temp_images.append(img)

     # Combine them into grid
     grid_img = None
     for i in range(n_rows):
          row_images = temp_images[i * n_cols:(i + 1) * n_cols]
          if len(row_images) < n_cols:
               row_images += [np.zeros_like(row_images[0])] * (n_cols - len(row_images)) # empty images

          row_img = cv2.hconcat([cv2.resize(img, (800, 608)) for img in row_images])
          if grid_img is None:
               grid_img = row_img
          else:
               grid_img = cv2.vconcat([grid_img, row_img])

     task.logger.report_image(title="Predictions Grid", series=split, image = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))

@click.command()
@click.option("--dataset", "-d", help="Name of the dataset in the datasets directory")
@click.option("--experiment_name" "-n", default="indices", help="Project name in clear ml")
def main(dataset, experiment_name_n):
     print("Running task for", dataset)
     Task.set_credentials(
          api_host="https://api.clear.ml",
          web_host="https://app.clear.ml",
          files_host="https://files.clear.ml",
          key='R3KF8584CYHXTMATKOBMGBTKOR0GL8',
          secret='iJTV32010g3PEL3Euv0b1cZaClWTuu3dlCx4oQOyLCoYo4rqk2RsSIaVePr-AX-yR7A'
     )

     task = Task.init(project_name=experiment_name_n, task_name=dataset)

     model = YOLO("yolov10s.pt")  # load a pretrained model
     yaml_path = f"datasets/{dataset}" 
     results = model.train(data=f"{yaml_path}/{dataset}.yaml", epochs=100, imgsz=[800, 608], batch=8)


     for split in ("train", "val"):
          results = model.predict(source=f"datasets/{dataset}/images/{split}", conf=0.3, save=False)
          gt = read_ground_truth(results)
          save_gt_and_pred(task, results, gt, split=split)

     task.close()


if __name__ == "__main__":
     main()