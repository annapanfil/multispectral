#!/usr/bin/python 

import subprocess
import numpy as np
import requests
import rospy
from sensor_msgs.msg import Image

# only for local testing
from processing.load import find_images
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

if __name__ == "__main__":
    rospy.init_node("UAV_multispectral_publisher")
    rospy.loginfo("Node has been started")

    # url = "http://192.168.1.83/capture"
    # params = {
    #     "block": "true",
    #     "cache_jpg": "31"
    # }

    # try:
    #     output = subprocess.check_output(["ping", "-c", "1", "192.168.1.83"], stderr=subprocess.STDOUT)
    # except subprocess.CalledProcessError:
    #     rospy.logerr("Failed to ping the camera")
    #     exit(1)

    rate = rospy.Rate(3) # 50
    image_pub = rospy.Publisher("/multispectral/image", Image, queue_size=10)


    trigger_rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        # capture photo
        print("Capturing photo")
        # response = requests.get(url, params=params)
        # if response.status_code == 200 and "raw_storage_path" in response.json().keys():
        if True:
            # get image from camera
            channels = []
            # for ch, path in response.get("raw_cache_path").items()[-1]: # no panchromatic
            #     photo_url = "http://192.168.10.254" + path
            #     img_response = requests.get(photo_url)
            #     channels.append(img_response.content) # one channel
            # get images from local directory
            image_names = find_images(Path("/home/anna/Datasets/raw_images/pool/realistic_trash/0034SET/000"), "0004", no_panchromatic=True)
            for im_name in image_names:
                ch = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
                channels.append(ch)
            img = np.dstack(channels)
            print("Image shape: ", img.shape, "dtype: ", img.dtype)
            # plt.imshow(img[:,:, :3])
            # plt.show()

            # Publish the image
            msg = Image()
            msg.header.stamp = rospy.Time.now()
            msg.height = img.shape[0]
            msg.width = img.shape[1]
            msg.encoding = "8UC6"  # or 32FC6, depending on your dtype
            msg.is_bigendian = 0
            msg.step = img.shape[1] * img.shape[2] * img.itemsize
            msg.data = img.tobytes()

            image_pub.publish(msg)
            trigger_rate.sleep()
        else:
            rospy.logwarn("Couldn't capture image")
            print(response.json())

