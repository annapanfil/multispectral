import rospy
from sensor_msgs.msg import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

rospy.init_node('image_publisher_node')
pub = rospy.Publisher('/multispectral/detection_image', Image, queue_size=1)
rate = rospy.Rate(3)
image_dir = "/home/anna/Datasets/raw_images/Hamburg_2025_05_15/images/0001SET/000"

images = sorted([x for x in os.listdir(image_dir) if x.endswith("_1.tif")])

while not rospy.is_shutdown() and len(images) > 0:
    img = cv2.imread(f"{image_dir}/{images.pop(0)}")
    msg = Image()
    msg.height = img.shape[0]
    msg.width = img.shape[1]
    msg.encoding = 'rgb8'
    msg.step = img.shape[1] * 3
    msg.data = img.tobytes()
    pub.publish(msg)
    print(f"Published {msg.header.frame_id}")
    rate.sleep()
