import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
from pathlib import Path
import subprocess
import time
import rospy

import click
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
import exiftool
import onnxruntime as ort
import numpy as np
from src.processing.consts import DATASET_BASE_PATH
from src.main_drone import capture_process #get_image_from_camera
from src.timeit import timer

# from src.main_ground import send_outcomes

import micasense.capture as capture
from src.processing.load import align_from_saved_matrices, find_images, get_irradiance, load_all_warp_matrices
from src.processing.consts import DATASET_BASE_PATH
from src.shapes import Rectangle
from src.utils import greedy_grouping, prepare_image
import gc
import shutil
import os
from multiprocessing import Process, Queue, Manager, Event #, LifoQueue
import signal
from typing import List


current_altitude = 25

def position_callback(msg):
    global current_altitude
    current_altitude = msg.point.z

@timer
def send_outcomes(bboxes: List[Rectangle], img: np.array, group_key: str, image_pub, pos_pixel_pub):
    print(f"Sending outcomes for {group_key}")
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = group_key
    msg.height = img.shape[0]
    msg.width = img.shape[1]
    msg.encoding = 'rgb8'
    msg.is_bigendian = False
    msg.step = img.shape[1] * 3
    msg.data = img.tobytes()

    image_pub.publish(msg)

    # cv2.imwrite(f'../out/processed/{group_key}.jpg', img)
    original_image_size = (1456, 1088)
    new_image_size = (800, 608)


    if len(bboxes) > 0:
        for i, bb in enumerate(bboxes):
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = group_key + f"_pile{i}"
            # convert from reduced image coordinates to original image coordinates
            cx = bb.center[0] * (original_image_size[0] / new_image_size[0])
            cy = bb.center[1] * (original_image_size[1] / new_image_size[1])

            msg.point.x = cx
            msg.point.y = cy
            
            pos_pixel_pub.publish(msg)
            rospy.loginfo(f"Sent positions for {group_key} to rostopic")
    else:
        rospy.logwarn("No pile detected. No positions to publish.")
        # rospy.logwarn("No pile detected. Reporting the image center") #TODO: delete
        # msg = PointStamped()
        # msg.header.stamp = rospy.Time.now()
        # msg.header.frame_id = group_key + f"_no_pile"
        # msg.point.x = int(original_image_size[0]/2)
        # msg.point.y = int(original_image_size[1]/2)
        
        # pos_pixel_pub.publish(msg)


@click.command()
@click.option("--debug", '-d', is_flag=True, default=False, help="Run in debug mode with local images")
def main(debug):
    """Main function to capture images from UAV multispectral camera and detect litter."""

    if not debug:
        print("Staring image proces")
        img_queue = Queue(maxsize=1)  # Limit to 1 sets of images, we want the newest made photo eny way
        stop_event = Manager().Event()
        url = "http://192.168.1.83/capture"
        params = {
            'block': 'true',
            'cache_raw': 31,   # All 5 bands (binary 11111)
            'store_capture': 'true', # Save to SD card
            'preview': 'false',
            'cache_jpeg': 0,
        }

        rospy.sleep(1.0)  # wait for publisher to register
        capture_proc = Process(target=capture_process, args=(img_queue, stop_event, debug, url, params))
        # capture_proc.daemon = True  # Terminate with main process
        capture_proc.start()

        def signal_handler(sig, frame):
            print("\nStopping...")
            capture_proc.terminate()
            stop_event.set()
            capture_proc.join(timeout=1.0)
            if capture_proc.is_alive():
                print("Terminating capture process...")
                capture_proc.terminate()
            if img_queue.full():
                old_paths = img_queue.get_nowait()
                temp_dir = old_paths[0].rsplit("/", 1)[0]
                if os.path.exists(temp_dir):
                    print(f"Cleaning up temporary directory: {temp_dir}")
                    shutil.rmtree(temp_dir)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

    rospy.init_node("UAV_multispectral_detector")
    rospy.loginfo("Node has been started")

    ### CONFIGURATION
    formula = "(N - (E - N))"
    channels = ["N", "G", formula]
    # channels = ["R", "G", "B"]
    is_complex = False
    new_image_size = (800, 608)
    original_image_size = (1456, 1088)

    warp_matrices_dir = f"{DATASET_BASE_PATH}/warp_matrices"
    model_path = "/home/lariat/code/onnx/model_2.onnx"
    panel_path = f"{DATASET_BASE_PATH}/raw_images/temp_panel" # here is saved the panel image when src.get_panel is used
    panel_nr = "0000"

    ### INITIALIZATION
    times = []
    position_sub = rospy.Subscriber("/dji_osdk_ros/local_position", PointStamped, callback=position_callback)   

    print("Loading warp matrices and panel image...")
    warp_matrices = load_all_warp_matrices(warp_matrices_dir)
    panel_names = find_images(Path(panel_path), panel_nr, panel=True, no_panchromatic=True)
    panel_capt = capture.Capture.from_filelist(panel_names)

    if (albedo := panel_capt.panel_albedo()) is not None:
        panel_reflectance_by_band = albedo
    else:
        panel_reflectance_by_band = [0.49, 0.49, 0.49, 0.49, 0.49] #RedEdge band_index order
    panel_irradiance = panel_capt.panel_irradiance(panel_reflectance_by_band)
    
    
    pos_pixel_pub = rospy.Publisher("/multispectral/pile_pixel_position", PointStamped, queue_size=10)
    image_pub = rospy.Publisher("/multispectral/detection_image", Image, queue_size=10)
    
    print("Loading model...")
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name

    if not debug:
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
    try: 
        while not rospy.is_shutdown():
            start = time.time()

            altitude = current_altitude

            if not debug:
                #get image from subprocess queue
                s = time.time()
                try:
                    paths = img_queue.get(timeout=5.0)
                except Queue.Empty:
                    rospy.logwarn("Image queue timeout, no images received in 5 seconds. Retrying...")
                    continue
                # paths = img_queue.get()  # 5s timeout timeout=5.0
                e = time.time()
                print(f"Time of waiting for img in queue: {e-s:.2f}")
            else:
                test_img = "/home/lariat/images/raw_images/hamburg_2025_05_19/images/0000SET/000/IMG_0172_1.tif"
                paths = [test_img.replace("_1.tif", f"_{ch}.tif") for ch in range(1, 6)]

                # try:
                #     s=time.time()
                #     msg = rospy.wait_for_message(TOPIC_NAME, PointStamped, timeout=10)
                #     storage_path = msg.header.frame_id.replace("/files/", IMAGE_DIR)
                #     paths = [storage_path.replace("_1.tif", f"_{ch}.tif") for ch in range(1, 6)]
                #     e=time.time()
                #     print(f"Time of waiting for message with img path: {e-s:.2f} seconds")
                # except rospy.ROSException:
                #     rospy.logwarn(f"No message received on topic {TOPIC_NAME} within 10s. Retrying...")
                #     continue
            # group_key = paths[0].rsplit("/", 1)[1].rsplit("_", 1)[0] # for logging purposes
            # Preprocessing
            # print(f"Processing {group_key} with altitude {altitude:.0f} m")
            # rospy.loginfo(f"Processing {group_key} with altitude {altitude:.0f} m")
            
            s = time.time()
            img_capt = capture.Capture.from_filelist(paths)
            e = time.time()
            print(f"Time of Capture.from_filelist: {e-s:.2f} seconds")

            img_type = get_irradiance(img_capt, panel_capt, panel_irradiance, display=False, vignetting=False)

            img_aligned = align_from_saved_matrices(img_capt, img_type, warp_matrices, altitude, allow_closest=True, reference_band=0)

            image = prepare_image(img_aligned, channels, is_complex, new_image_size)
            
            s = time.time()
            # Add batch dimension: (1, C, H, W)
            input = np.transpose(image, (2, 1, 0))
            input = np.expand_dims(input, axis=0)
            
            # Detection
            print("Detecting litter...")
            results = session.run(None, {input_name: input})
            e = time.time()
            print(f"Time of model inference: {e-s:.2f} seconds")

            bbs = []
            for pred_bbs in results[0]:
                for bb in pred_bbs:
                    x1, y1, x2, y2, conf, cls = bb
                    # filter if needed: e.g., if conf > 0.5
                    if conf > 0.5:
                        rect = Rectangle(y1, x1, y2, x2, "rect")
                        bbs.append(rect)        

            # Merging
            print("Merging piles...")
            merged_bbs, _, _ = greedy_grouping(bbs, image.shape[:2], resize_factor=1.5, visualize=False)

            image = (image * 255.0).astype(np.uint8)  # Convert to uint8
            for rect in merged_bbs:
                rect.draw(image, color=(0, 255, 0), thickness=2)
            
            # Sending outcomes
            group_key = "abcd"
            print(f"Found {len(merged_bbs)} litter piles in {group_key}")
            send_outcomes(merged_bbs, image, group_key, image_pub, pos_pixel_pub)


            # Cleanup
            s = time.time()
            if not debug:
                temp_dir = paths[0].rsplit("/", 1)[0]  # Assuming all images are in the same directory
                print(f"Cleaning up temporary directory: {temp_dir}")
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            del img_capt, img_aligned, image, pred_bbs, merged_bbs, results, input, bbs
            gc.collect()  # import gc
            e = time.time()
            print(f"Time of cleanup: {e-s:.2f} seconds")

            end = time.time()
            print(f"Time for this photo: {end - start:.2f} seconds")
            # rospy.loginfo(f"Time for this photo: {end - start:.2f} seconds")
            times.append(end - start)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup when ROS shuts down
        stop_event.set()
        capture_proc.join(2.0)
        if capture_proc.is_alive():
            capture_proc.terminate()
    
    if len(times) != 0:
        print("TIME STATISTICS:")
        print("--------------------")
        print("min: {}, max: {}, avg: {}".format(min(times[1:]), max(times[1:]), sum(times[1:])/len(times[1:])))
    else:
        print("No images captured, time statistics not available.")


if __name__ == "__main__":
    main()
