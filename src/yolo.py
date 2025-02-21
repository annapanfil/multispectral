
# !pip install ultralytics
# !pip install clearml
# !pip install clearml-agent

from clearml import Task
from ultralytics import YOLO

Task.set_credentials(
     api_host="https://api.clear.ml",
     web_host="https://app.clear.ml",
     files_host="https://files.clear.ml",
     key='R3KF8584CYHXTMATKOBMGBTKOR0GL8',
     secret='iJTV32010g3PEL3Euv0b1cZaClWTuu3dlCx4oQOyLCoYo4rqk2RsSIaVePr-AX-yR7A'
)

task = Task.init(project_name='piles', task_name='piles_random_m13')

model = YOLO("yolov10s.pt")  # load a pretrained model
yaml_path = "/home/anna/Datasets/created/piles_m13_random" 
results = model.train(data=f"{yaml_path}/piles_m13_random.yaml", epochs=50, imgsz=[800, 608], batch=16, workers = 4)

task.close()


