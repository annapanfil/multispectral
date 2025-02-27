Processing multispectral images:
- triggering the camera (Micasense RedEdge-P)
- comparing them with RGB images from H20T
- making a video
- getting configs from bag files from the experiment
- aligning channels
- exporting channels and their combinations
- showing the images
- merging the annotations from various channels
- real litter size estimation
- data analysis
- moving the files to create the dataset
- litter detection with YOLO
- litter detection with custom algorithm (blob detection + SVM)

run the tests: `pytest`

generate documentation: `PYTHONPATH=src python -m pdoc detection -o docs -d google`

train yolo (from src directory):  `ls ../datasets/ | grep pool_* | xargs -I {} python3 -m detection.yolo -d {} -tag "ghost-net index", --tag "pool ds testing", --tag "YOLO10n"`
