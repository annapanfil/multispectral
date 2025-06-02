#!/usr/bin/python 

"""Main script for online processing when the photos are received from the drone and processed on the ground. (Drone side)"""

import os
import socket
import struct
import subprocess
import numpy as np
import requests
import rospy
# from sensor_msgs.msg import Image
import zstandard as zstd


# only for local testing
from src.processing.load import find_images
import cv2
from pathlib import Path
from matplotlib import pyplot as plt

"""Capture the images, and send them to ros topic."""

# current_altitude = 0
# printing_idx = 0

# def position_callback(msg):
#     global current_altitude
#     current_altitude = msg.point.z   
    
#     global printing_idx
#     if printing_idx % 20 == 0:
#         rospy.loginfo("Current altitude: {}".format(current_altitude))
#     printing_idx += 1

def almost_equal(a, b, epsilon):
    return abs(a - b) <= epsilon

def get_image_from_camera(url, params, output_dir=None):
    """ Capture photo and download it
        if output_dir is not None, save the image to disk and return the path
        else return the image as numpy array""" 
    
    response = requests.get(url, params=params)
    if response.status_code == 200 and "raw_storage_path" in response.json().keys():
        # get image from camera
        channels = []
        for ch, path in response.get("raw_cache_path").items()[-1]: # no panchromatic
            photo_url = "http://192.168.10.254" + path
            img_response = requests.get(photo_url)
            #TODO: check if the metadata is kept
            channels.append(img_response.content) # one channel
            #TODO: If the metadata is not kept, we need to save the image to disk
            #TODO: save to temp
            if output_dir is not None:
               output_path = save_image(output_dir, response, ch)
               return output_path

        img = np.dstack(channels)
        return img
    else:
        rospy.logwarn(f"Couldn't capture image ({response.json()})")
        return None


def get_image_from_local_directory(path = "/home/anna/Datasets/raw_images/pool/realistic_trash/0034SET/000", img_nr = "0004"):
    # get images from local directory
    image_names = find_images(Path(path), img_nr, no_panchromatic=True)

    channels = [cv2.imread(im_name, cv2.IMREAD_GRAYSCALE) for im_name in image_names]
    img = np.dstack(channels)
    return img


def send_compressed_from_dir(path, cctx, sock):
    """ Compress the image and return the filename and compressed image in binary format. """
    filename = os.path.basename(path)
    with open(path, 'rb') as f:
        raw = f.read()
    img_bytes = cctx.compress(raw)
    filename_bytes = filename.encode("utf-8")
    try:
        sock.sendall(struct.pack('!I', len(filename_bytes)) + filename_bytes + struct.pack('!I', len(img_bytes)) + img_bytes)
    except (ConnectionResetError, BrokenPipeError):
        rospy.loginfo("Connection reset by peer. Exiting.")
        exit(1)
        

def save_image(output_dir, response, ch):
    capture_nr = response.get("raw_storage_path").get("1").split("/")[2] # ommiting 000 directory from '/files/0010SET/000/IMG_0001_5.tif'
    photo_nr = response.get("raw_storage_path").get(ch).split("/")[4]
    output_file = os.path.join(output_dir, capture_nr, photo_nr)
    print(f"Saving channel {ch} to {output_file}")

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

    # # TEST CAMERA CONNECTION
    # try:
    #     output = subprocess.check_output(["ping", "-c", "1", "192.168.1.83"], stderr=subprocess.STDOUT)
    # except subprocess.CalledProcessError:
    #     rospy.logerr("Failed to ping the camera")
    #     exit(1)

    image_dir = "/home/anna/Datasets/raw_images/hamburg_2025_05_19/images/0000SET/000"
    image_nr = 150
    ch = 1
    end_image_nr = 999
    receiver_ip = "127.0.0.1" #'10.2.119.163' # port 5000

    cctx = zstd.ZstdCompressor(level=3, threads=8)
    rate = rospy.Rate(3) # 50
    # image_pub = rospy.Publisher(topic_name, Image, queue_size=10)
    trigger_rate = rospy.Rate(1)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.connect((receiver_ip, 5000))
        while not rospy.is_shutdown():
            print("Capturing photo")
            # path = get_image_from_camera(url, params, output_dir=image_dir)
            path = os.path.join(image_dir, "IMG_{:04d}_{ch}.tif".format(image_nr, ch=ch))

            send_compressed_from_dir(path, cctx, sock)
 
            print(f"Sent: {path}")
            ch += 1
            if ch > 5:
                ch = 1
                image_nr += 1
            if image_nr > end_image_nr:
                break
            
        trigger_rate.sleep()
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()

