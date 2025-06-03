"""Main script for online processing when the photos are received from the drone and processed on the ground. (ground_side)"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
import queue
import signal
import socket
import struct
import tempfile
import threading
import time
from typing import List

import cv2
import numpy as np
# from cv_bridge import CvBridge
import exiftool
from matplotlib import pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import zstandard as zstd


from src.shapes import Rectangle
from src.utils import get_real_piles_size, greedy_grouping, prepare_image
from src.processing.load import align_from_saved_matrices, find_images, get_irradiance, load_all_warp_matrices, load_image_set
from src.processing.consts import CAM_HFOV, CAM_VFOV, DATASET_BASE_PATH

import sys
import micasense.capture as capture

#########################

panel_capt = None
warp_matrices = None
model = None
pos_pixel_pub = None
image_pub = None

formula = "(N - (E - N))"
channels = ["N", "G", formula]
is_complex = False
new_image_size = (800, 608)
original_image_size = (1456, 1088)
DEBUG = True

#########################

def receive_and_unpack_data(conn, decompressor):
    # receive the size of the filename and the filename
    try:
        filename_size_data = conn.recv(4)
    except socket.timeout:
        return None
    
    if not filename_size_data:
        return None
    (filename_size,) = struct.unpack('!I', filename_size_data)
    filename_b = conn.recv(filename_size)
    filename = filename_b.decode('utf-8')

    # receive the size of the compressed image and the image 
    size_data = conn.recv(4)
    (size,) = struct.unpack('!I', size_data)
    
    compressed = b''
    while len(compressed) < size:
        chunk = conn.recv(size - len(compressed))
        if not chunk:
            raise ConnectionError("Socket closed before receiving full compressed data")
        compressed += chunk
    raw = decompressor.decompress(compressed)

    return filename, raw


def handle_incoming_image(filename, image_data, image_groups):
    group_key = filename.rsplit("_", 1)[0]
    image_groups[group_key].append(image_data)
    print(f"Received image {filename} ({len(image_groups[group_key])} / 5)")

    if len(image_groups[group_key]) == 5:
        process_whole_img(group_key, image_groups[group_key])
        del image_groups[group_key]


def process_whole_img(group_key, images):
    # save the images to temporary files so micasense can read them
    with tempfile.TemporaryDirectory() as tmpdir:  # Uses /tmp (RAM)
        file_paths = []
        for i, img_data in enumerate(images):
            path = os.path.join(tmpdir, f"{group_key}_{i}.tiff")
            with open(path, "wb") as f:
                f.write(img_data)
            file_paths.append(path)
        
        # Process the images
        try:
            with exiftool.ExifToolHelper() as et:
                altitude = et.get_tags(file_paths[0], ["Composite:GPSAltitude"])[0]["Composite:GPSAltitude"]
        except exiftool.exceptions.ExifToolExecuteError:
            rospy.logwarn(f"ExifToolError: Could not read altitude from exif {file_paths[0]}. Trying to use default value (20).")
            altitude = 20

        rospy.loginfo(f"Processing group {group_key} with altitude {altitude}")

        img_capt = capture.Capture.from_filelist(file_paths)
        img_type = get_irradiance(img_capt, panel_capt, display=False, vignetting=False)
        img_aligned = align_from_saved_matrices(img_capt, img_type, warp_matrices, altitude, allow_closest=True, reference_band=0)
        image = prepare_image(img_aligned, channels, is_complex, new_image_size)

        results = model.predict(source=image, save=False, verbose=False)
        pred_bbs = results[0].boxes.xyxy.cpu().numpy()
        pred_bbs = [Rectangle(*bb, "rect") for bb in pred_bbs]

        # Merge piles
        merged_bbs, merged_img, _ = greedy_grouping(pred_bbs, image.shape[:2], resize_factor=1.5, visualize=False)

        for rect in merged_bbs:
            rect.draw(image, color=(0, 255, 0), thickness=2)
            
        send_outcomes(merged_bbs, image, group_key, image_pub, pos_pixel_pub)

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
        


def exit_threads(executor, server):
    print('Finishing threads, please wait...')

    py_version = sys.version_info
    if ( py_version.major == 3 ) and ( py_version.minor < 9 ):
        # Executor#shutdown does not accept cancel_futures keyword
        # prevent new tasks from being submitted
        executor.shutdown( wait = False )
        while True:
            # cancel all waiting tasks
            try:
                work_item = executor._work_queue.get_nowait()
                                
            except queue.Empty:
                break
                                
            if work_item is not None:
                work_item.future.cancel()

    else:
        executor.shutdown( cancel_futures = True )

    print('Threads finished')
    server.shutdown(socket.SHUT_RDWR)  
    server.close() 
                    
    sys.exit(0)


def main():
    rospy.init_node("litter_detection_publisher")
    rospy.loginfo("Node has been started")

    warp_matrices_dir = f"{DATASET_BASE_PATH}/annotated/warp_matrices"
    model_path = "models/sea-form8_sea_aug-random_best.pt"
    panel_path = f"{DATASET_BASE_PATH}/raw_images/temp_panel"
    panel_nr = "0000"

    image_groups = defaultdict(list)

    # Pre-load all warp matrices and panel
    global panel_capt 
    global warp_matrices
    global pos_pixel_pub
    global image_pub

    warp_matrices = load_all_warp_matrices(warp_matrices_dir)
    panel_names = find_images(Path(panel_path), panel_nr, panel=True, no_panchromatic=True)
    panel_capt = capture.Capture.from_filelist(panel_names)
    
    pos_pixel_pub = rospy.Publisher("/multispectral/pile_pixel_position", PointStamped, queue_size=10)
    image_pub = rospy.Publisher("/multispectral/detection_image", Image, queue_size=10)

    decompressor = zstd.ZstdDecompressor() 
    executor = ThreadPoolExecutor(max_workers=16)
    signal.signal(signal.SIGINT, lambda sig, frame: exit_threads( executor, server ))

    # Initialize model with optimizations
    global model
    model = YOLO(model_path)
    model.fuse() # Fuse Conv+BN layers
    model.half()
    model.conf = 0.5

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(('0.0.0.0', 5000))
        server.listen(1)
        print("Waiting for connection...")
        server.settimeout(1)
        conn = None
        while not rospy.is_shutdown():
            try:
                conn, addr = server.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except socket.timeout:
                continue
            break
        if conn is not None:
            print(f"Connected by {addr}")
            print("Connection established " + str(time.time()))
            
            with conn:
                conn.settimeout(20.0)
                while not rospy.is_shutdown():
                    res = receive_and_unpack_data(conn, decompressor)
                    if res is None:
                        continue
                    filename, raw = res
                    rospy.sleep(1)
                    
                    executor.submit(handle_incoming_image, filename, raw, image_groups)

        print("Finishing")
        server.shutdown(socket.SHUT_RDWR)  
        server.close() 


if __name__ == '__main__':
    main()
