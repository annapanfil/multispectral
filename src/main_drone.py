#!/usr/bin/python 

"""Main script for online processing when the photos are received from the drone and processed on the ground. (Drone side)"""

import os
import socket
import struct
import subprocess
import time
import numpy as np
import requests
from src.timeit import timer
import shutil
import rospy
from geometry_msgs.msg import PointStamped

# import rospy
# import zstandard as zstd

DEBUG = True # local testing (without camera)

if DEBUG:
    # only for local testing
    from src.processing.load import find_images
    from src.processing.consts import DATASET_BASE_PATH
    import cv2
    from pathlib import Path
    from matplotlib import pyplot as plt

def almost_equal(a, b, epsilon):
    return abs(a - b) <= epsilon


def capture_process(img_queue, stop_event, debug, url, params):
    """Continuous process to capture images and add paths to queue."""
    
    try:
        rospy.init_node('camera_capture_node', anonymous=True)
        img_cam_url_publisher = rospy.Publisher("/camera/trigger", PointStamped, queue_size=10)

        print("Capture process started")
        i = 0
        while not stop_event.is_set():
            # check if image was alredy consumed by main process
            if img_queue.empty():
                rospy.loginfo("Capturing photo...")
                output_dir = f"/dev/shm/capture_{i}_{os.getpid()}"
                os.makedirs(output_dir, exist_ok=True)
                i+=1
                paths = get_image_from_camera(output_dir, img_cam_url_publisher, debug, url, params)
                
                # Put the new image in the queue
                img_queue.put(paths)
                print(f"Captured {len(paths)} images.")
    
    #end of process 
    except KeyboardInterrupt:
        pass  # Let main process handle termination
    finally:
        if debug:
            print("Capture process exiting")



@timer
def get_image_from_camera(output_dir, img_cam_url_publisher, debug=False, url=None, params=None):
    """ Capture photo and download it. Save the image to disk and return the path""" 
    if debug:
        # Simulate with test images
        test_img = "/home/lariat/images/raw_images/temp/IMG_0046_1.tif"
        return [test_img.replace("_1.tif", f"_{ch}.tif") for ch in range(1, 6)]
    
    session = requests.Session()
    images_paths = []

    # Trigger capture
    with requests.Session() as session:
        response = session.get(url, params=params)
        data = response.json()
    
        if data.get('status') != 'complete':
            raise RuntimeError(f"Capture failed: {data}")
        
        # Sand path of the photo on camera storage to ROS topic for bagfile recording 
        img_name = data.get("raw_storage_path").get("1") # name of channel 1 image
        trigger_msg=PointStamped()
        trigger_msg.header.frame_id = img_name

        # TODO get position from drone    
        # position_sub = rospy.Subscriber("/dji_osdk_ros/local_position", PointStamped, callback=position_callback)   
        trigger_msg.point.z = 25

        trigger_msg.header.stamp = rospy.Time.now()
        img_cam_url_publisher.publish(trigger_msg)

        # Download all bands
        raw_paths = data.get('raw_cache_path', {})
        for ch, path in raw_paths.items():
            full_url = "http://192.168.1.83" + path

            response = session.get(full_url, timeout=10)
            if response.status_code == 200:
                path = os.path.join(output_dir, f"band_{ch}.tiff")
                with open(path, 'wb') as f:
                    f.write(response.content)
                images_paths.append(path)
    return images_paths
    


def send_compressed_from_dir(path, cctx, sock, del_file=False):
    """ Compress the image and return the filename and compressed image in binary format. """
    filename = os.path.basename(path)
    with open(path, 'rb') as f:
        raw = f.read()
    img_bytes = cctx.compress(raw)
    filename_bytes = filename.encode("utf-8")
    try:
        sock.sendall(struct.pack('!I', len(filename_bytes)) + filename_bytes + struct.pack('!I', len(img_bytes)) + img_bytes)
        if del_file:
            os.remove(path)
    except (ConnectionResetError, BrokenPipeError):
        rospy.loginfo("Connection reset by peer. Exiting.")
        rospy.signal_shutdown("Connection reset by peer")
        raise ConnectionResetError("Connection reset by peer. Exiting.")
        # exit(1)

        

def save_image(output_dir, response, photo_nr, ch):
    img_name = photo_nr.split(".")[0]
    output_file = f"{output_dir}/{img_name}_{ch}.tif"
    print("Saving channel {} to {}".format(ch, output_file))

    with open(output_file, 'wb') as file:
        file.write(response.content)
    
    return output_file
 

if __name__ == "__main__":
    rospy.init_node("UAV_multispectral_publisher")
    rospy.loginfo("Node has been started")

    url = "http://192.168.1.83/capture"
    params = {
        "block": "true",
        "cache_jpg": "31"
    }

    if not DEBUG:
        # TEST CAMERA CONNECTION
        try:
            output = subprocess.check_output(["ping", "-c", "1", "192.168.1.83"], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            rospy.logerr("Failed to ping the camera")
            exit(1)
    else:
        # set local directory and image numbers for testing
        image_dir = f"{DATASET_BASE_PATH}/raw_images/hamburg_2025_05_19/images/0000SET/000"
        image_nr = 154
        end_image_nr = 999

    receiver_ip = "127.0.0.1" # port 5000
    times = []

    cctx = zstd.ZstdCompressor(level=3, threads=8)
    rate = rospy.Rate(3) # 50
    trigger_rate = rospy.Rate(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect((receiver_ip, 5000))
    socket_connected = True

    while not rospy.is_shutdown():
        start = time.time()
        print("Capturing photo")

        if DEBUG:
            # get images from local directory
            paths = [os.path.join(image_dir, "IMG_{:04d}_{ch}.tif".format(image_nr, ch=ch)) for ch in range(1, 6)]
        else:
            paths = get_image_from_camera(url, params)

        for path in paths:
            try:
                send_compressed_from_dir(path, cctx, sock, del_file= not DEBUG) # delete file if not in debug mode
                print("Sent: {}".format(path))
            except (ConnectionResetError, BrokenPipeError):
                socket_connected = False
                break

        end = time.time()
        print("time for this photo: {} ".format(end-start))
        times.append(end-start)
        
        if DEBUG:   
            image_nr += 1
            if image_nr > end_image_nr:
                break
                
        trigger_rate.sleep()

    if socket_connected:
        sock.shutdown(socket.SHUT_RDWR)
    sock.close()

    print("TIME STATISTICS:")
    print("--------------------")
    print("min: {}, max: {}, avg: {}".format(min(times), max(times), sum(times)/len(times)))


