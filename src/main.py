import subprocess
import time

import click
import cv2
import exiftool
from geometry_msgs.msg import PointStamped
from matplotlib import pyplot as plt
from pathlib import Path
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO

import micasense.capture as capture

from src.main_drone import get_image_from_camera
from src.main_ground import send_outcomes
from src.processing.load import align_from_saved_matrices, find_images, get_irradiance, load_all_warp_matrices
from src.processing.consts import DATASET_BASE_PATH
from src.shapes import Rectangle
from src.utils import greedy_grouping, prepare_image

current_altitude = 10

def position_callback(msg):
    global current_altitude
    current_altitude = msg.point.z


@click.command()
@click.option("--debug", '-d', is_flag=True, default=False, help="Run in debug mode with local images")
def main(debug):
    """Main function to capture images from UAV multispectral camera and detect litter."""
    rospy.init_node("UAV_multispectral_detector")
    rospy.loginfo("Node has been started")
    print("Starting UAV multispectral litter detection...")

    ### CONFIGURATION
    formula = "(N - (E - N))"
    # channels = ["N", "G", formula]
    channels = ["R", "G", "B"]
    is_complex = False
    new_image_size = (800, 608)
    original_image_size = (1456, 1088)

    warp_matrices_dir = f"{DATASET_BASE_PATH}/warp_matrices"
    model_path = "models/sea-form8_sea_aug-random_best.pt"
    panel_path = f"{DATASET_BASE_PATH}/raw_images/temp_panel" # here is saved the panel image when src.get_panel is used
    panel_nr = "0436"

    ### INITIALIZATION
    times = []
    position_sub = rospy.Subscriber("/dji_osdk_ros/local_position", PointStamped, callback=position_callback)   

    warp_matrices = load_all_warp_matrices(warp_matrices_dir)
    panel_names = find_images(Path(panel_path), panel_nr, panel=True, no_panchromatic=True)
    panel_capt = capture.Capture.from_filelist(panel_names)
    
    pos_pixel_pub = rospy.Publisher("/multispectral/pile_pixel_position", PointStamped, queue_size=10)
    image_pub = rospy.Publisher("/multispectral/detection_image", Image, queue_size=10)

    print("Loading model...")
    model = YOLO(model_path)
    model.fuse() # Fuse Conv+BN layers
    model.half()
    model.conf = 0.5

    if not debug:
        url = "http://192.168.1.83/capture"
        params = {
            "block": "true",
            "cache_jpg": "31"
        }

        # TEST CAMERA CONNECTION
        # print("Testing camera connection...")
        rospy.loginfo("Testing camera connection...")
        try:
            output = subprocess.check_output(["ping", "-c", "1", "192.168.1.83"], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            rospy.logerr("Failed to ping the camera")
            exit(1)
    else:
        # get image numbers from bag file
        IMAGE_DIR = f"{DATASET_BASE_PATH}/raw_images/hamburg_2025_05_19/images/"
        TOPIC_NAME = "/camera/trigger"

    ### MAIN LOOP
    # while True:
    while not rospy.is_shutdown():
        start = time.time()

        rospy.loginfo("Capturing photo...")
        # print("Capturing photo...")
        altitude = current_altitude
        if not debug:
            # get images from camera
            paths = get_image_from_camera(url, params)
        else:
            paths = ["/home/anna/Datasets/raw_images/mandrac_2025_04_16/images/0004SET/003/IMG_0604_1.tif"\
                     .replace("_1.tif", f"_{ch}.tif") for ch in range(1, 6)]
            # # get images from local directory
            # try:
            #     msg = rospy.wait_for_message(TOPIC_NAME, PointStamped, timeout=10)
            #     storage_path = msg.header.frame_id.replace("/files/", IMAGE_DIR)
            #     paths = [storage_path.replace("_1.tif", f"_{ch}.tif") for ch in range(1, 6)]
            # except rospy.ROSException:
            #     rospy.logwarn(f"No message received on topic {TOPIC_NAME} within 10s. Retrying...")
            #     continue
            
        group_key = paths[0].rsplit("/", 1)[1].rsplit("_", 1)[0] # for logging purposes

        # Preprocessing
        # print(f"Processing {group_key} with altitude {altitude:.0f} m")
        rospy.loginfo(f"Processing {group_key} with altitude {altitude:.0f} m")
        
        img_capt = capture.Capture.from_filelist(paths)
        img_type = get_irradiance(img_capt, panel_capt, display=False, vignetting=False)
        img_aligned = align_from_saved_matrices(img_capt, img_type, warp_matrices, altitude, allow_closest=True, reference_band=0)

        image = prepare_image(img_aligned, channels, is_complex, new_image_size)    

        input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Detection
        print("Detecting litter...")
        results = model.predict(source=input, save=False, verbose=False) # takes BGR image as input
        pred_bbs = results[0].boxes.xyxy.cpu().numpy()
        pred_bbs = [Rectangle(*bb, "rect") for bb in pred_bbs]

        # Merging
        merged_bbs, _, _ = greedy_grouping(pred_bbs, image.shape[:2], resize_factor=1.5, visualize=False)

        for rect in merged_bbs:
            rect.draw(image, color=(0, 255, 0), thickness=2)
            
        print(f"Found {len(merged_bbs)} litter piles in {group_key}")
        # Sending outcomes
        send_outcomes(merged_bbs, image, group_key, image_pub, pos_pixel_pub)

        end = time.time()
        print(f"Time for this photo: {end - start:.2f} seconds")
        rospy.loginfo(f"Time for this photo: {end - start:.2f} seconds")
        times.append(end - start)
    
    if len(times) != 0:
        print("TIME STATISTICS:")
        print("--------------------")
        print("min: {}, max: {}, avg: {}".format(min(times), max(times), sum(times)/len(times)))
    else:
        print("No images captured, time statistics not available.")

if __name__ == "__main__":
    main()
