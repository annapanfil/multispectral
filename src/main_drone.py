#!/usr/bin/python 

"""Main script for online processing when the photos are received from the drone and processed on the ground. (Drone side)"""

import os
import socket
import struct
import subprocess
import time
import numpy as np
import requests
import rospy
import zstandard as zstd

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

def get_image_from_camera(url, params):
    """ Capture photo and download it. Save the image to disk and return the path""" 
    
    response = requests.get(url, params=params)
    if response.status_code == 200 and "raw_storage_path" in response.json().keys():
        # get image from camera
        response = response.json()
        output_paths = []
        for ch, path in response.get("raw_cache_path").items()[:-1]: # no panchromatic
            photo_nr = response.get("raw_storage_path").get(ch).split("/")[4]
            photo_url = "http://192.168.1.83" + path
            img_response = requests.get(photo_url)

            output_dir = ("/home/dji/Documents/images/")
            output_path = save_image(output_dir, img_response, photo_nr, ch)
            output_paths.append(output_path)
        return output_paths
    else:
        rospy.logwarn("Couldn't capture image ({})".format(response.json()))
        return None


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
        exit(1)
        

def save_image(output_dir, response, photo_nr, ch):
    output_file = "{}/{}".format(output_dir, photo_nr)
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
    while not rospy.is_shutdown():
        start = time.time()
        print("Capturing photo")

        if DEBUG:
            # get images from local directory
            paths = [os.path.join(image_dir, "IMG_{:04d}_{ch}.tif".format(image_nr, ch=ch)) for ch in range(1, 6)]
        else:
            paths = get_image_from_camera(url, params)

        for path in paths:
            send_compressed_from_dir(path, cctx, sock, del_file= not DEBUG) # delete file if not in debug mode
            print("Sent: {}".format(path))

        end = time.time()
        print("time for this photo: {} ".format(end-start))
        times.append(end-start)
        
        if DEBUG:   
            image_nr += 1
            if image_nr > end_image_nr:
                break
                
        trigger_rate.sleep()
    sock.shutdown(socket.SHUT_RDWR)
    sock.close()

    print("TIME STATISTICS:")
    print("--------------------")
    print("min: {}, max: {}, avg: {}".format(min(times), max(times), sum(times)/len(times)))


