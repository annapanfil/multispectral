# Detection of sea surface litter with multispectral camera

Part of the [SeaClear 2.0](https://www.seaclear2.eu/) EU project. **Detection of the surface litter** with DJI Matrice 300 equipped with Micasense RedEdge-P **multispectral camera**. The detection is done with **YOLOv10n** model, fine-tuned for multispectral images with **custom index**. Communication is done via **ROS** topics (ROS1) and optionally **TCP** sockets. The project includes web **GUI** in Dash. 

## Main script (src/)
The main file is triggering the multispectral camera, processing and returning the GPS coordinates of
the detected litter. The pipeline is as follows:
1. Read the image and its coordinates
2. Align the images
3. Preprocess them – normalise, get channels or indices, resize
4. Run model
5. Group the detections
6. Get the coordinates of the detections in photo frame
7. Get the GPS coordinates of the detections and post them to ROS topic.

The script posts to several ROS topics:
- `/camera/trigger` – posted when the photo is taken; `PointStamped` where `frame_id` is image path in the camera memory and `z` is the altitude.
- `/multispectral/detection_image"` – posted when the detection is completed; `Image` with rectangles around each detected pile.
- `/multispectral/pile_pixel_position"` – posted when a pile is detected, for each pile in the image; `PointStamped` where `x, y` are the coordinates of the pile's centre in image frame
- `/multispectral/pile_global_position"` – posted when a pile is detected, for each pile in the image; `NavSatFix` with GPS coordinates of pile's centre (#TODO: consult with Matej what he exactly needs to be posted).
- `"/multispectral/pile_enu_position"` – posted when a pile is detected, for each pile in the image; `PointStamped` with ENU coordinates of pile's centre

It relies on `/dji_osdk_ros/rtk_position`, `/dji_osdk_ros/local_position`, `/dji_osdk_ros/attitude` and `/dji_osdk_ros/local_position` topics.

### Most common scenario
Most commonly one wants to capture the panel image:
```
python3 -m src.get_panel
```

and then run at the same time:
```
python3 -m src.main                         # for capturing and detection
python3 -m src.global_position_publisher    # for determining GPS and ENU position
```

this runs the online detection, where **everything is done on the drone**. Alternatively drone can be only used for capturing the images:
```
python3 -m src.main_drone                   # capturing
```

and processing may be done on the ground:
```
python3 -m src.main_ground                  # preprocessing and detection
python3 -m src.global_position_publisher    # determining GPS and ENU position
```

Both nodes communicate via TCP connection on port 5000 of the ground computer. The IP address of the ground node must be changed in the `main_drone.py` file.

the `src.main` can be run with `-d` flag to run without the attached camera, from the bagfile and photos on the drive.

### Web interface
The web interface shows the map with a drone path and litter position and last detection image with bounding boxes around the litter.
It may be useful for final visualisation as well as debugging. 

To launch the interface create websocket and run the app:
```
roslaunch rosbridge_server rosbridge_websocket.launch port:=9091
python3 -m dev.show.show_map_multispectral_live
```

https://github.com/user-attachments/assets/e122bb07-5755-43c0-8f5e-548f5d57f4d2

If you want to draw the bounding boxes manually, use `python3 -m dev.show.topic_image_viewer`. There you also see last image with the detection. You can draw your own bounding box and send it to rostopics by pressing enter. Press q to quit the program.

https://github.com/user-attachments/assets/d61bcfcc-012b-44f5-a5d7-6a78a1f950e0


### Offline version
If the processing doesn't need to be done online run:
```
python3 -m dev.get_images.capture_images_constantly
```
remember to save the bagfile with ``camera/trigger` topic e.g. with `roslaunch dji_osdk_ros multispectral_lariat_bag.launch` command.

After landing download the videos and run `dev.offline.detect_on_all` script to produce video with detections from all frames. The script has several parameters:
```
  -i, --img_dir TEXT     Directory with images (name in raw_images dir)
  -b, --bag_path TEXT    Path to the bag file (relative to annotated dir)
  -o, --out_path TEXT    Output video path (relative to predicted_videos dir)
  -m, --model_path TEXT  Path to the YOLO model
  -p, --panel TEXT       Panel image number (for alignment)
  -s, --start INTEGER    Start frame number
  -e, --end INTEGER      End frame number
```
example run: `python3 -m dev.offline.detect_on_all -i hamburg_2025_05_15 -b hamburg/bags/matriceBag_multispectral_2025-05-15-12-04-05.bag -o hamburg_yolo_form8.mp4 -p 0017 -s 46 -e 299`

## Model training (train/)

Most common scenario for data collection was to continously take pile images on 5, 10, 15, 20, 25 and 30m above the ground. To process them we have to:
1. copy panel images to each subfolder
1. look for the images containing the litter and save their numbers
1. use `train.dataset_creation.get_configs_from_bag_files.py` script to create a config file for each altitude. Remember to change the parameters inside the file.
1. [Optionally] Export RGB and BReNir images with `train.dataset_creation.export_photo_type` to see aligned photos and choose which ones to annotate (use multirun e.g.: `python3 -m train.dataset_creation.export_photo_type --multirun processing="mandrac2025_5m,..."`). Then choose the images, removing unnecessary file numbers from config files.
1. Use `train.dataset_creation.export_aligned_channels` to export the channels, RGB and
index images for annotation.
1. Annotate the images with [Supervisely](https://app.supervisely.com/projects/) using either bounding boxes or polygons
1. [Optionally] Check the annotations with `dev.check_annotations`
1. [Optionally] If COCO export was used, convert it to YOLO with `train.dataset_creation.export_coco_to_yolo`
1. Merge the labels for all the channels with `train.dataset_creation.merge_labels`
1. [Optionally] augment the images using `train.dataset_creation.augment_dataset`
1. Create the dataset (split, resize, optionally exclude litter or merge it) using `train.dataset_creation.create_dataset` and `run_create_dataset.py`.
1. Train the model using `train.yolo` and `run_yolo_training.sh`.
1. [Optionally] Use `train.model_optimization` to find optimal hyperparameters with Optuna.
1. [Optionally] Quantize model with `train.quantize_model`

## Other scripts (dev\)
Dev directory contains different scripts and notebooks used for:
- triggering the camera (Micasense RedEdge-P) on various ways
- comparing multispectral images with RGB and thermal images from H20T
- offline detection and making a video
- data analysis
- litter detection with custom algorithm (blob detection + SVM)
- visualising the images
- showing maps and videos


## Other tasks

Generate documentation (a bit incomplete): from parent directory run `PYTHONPATH=./multispectral:$PYTHONPATH python -m pdoc multispectral -o multispectral/docs -d google`

Run the tests (only for index evaluation methods): `pytest`

## Requirements
The script requires multiple libraries which can be found in `environment.yaml` file. Also the [micasense library](https://github.com/micasense/imageprocessing) should be downloaded to `libraries` directory. 

For the imports to work PATH should contain the `multispectral/src` folder and PYTHONPATH – `multispectral/libraries/imageprocessing` folder.

Also remember to correctly set ROS_MASTER_URI and ROS_IP and run roscore.

The scripts are currently written for ROS1.

---
© Anna Panfil LARIAT 2025