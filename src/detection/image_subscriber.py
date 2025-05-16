import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2

def image_callback(msg):
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
    cv2.imshow("Image", img)
    cv2.waitKey(1)

rospy.init_node('multispectral_image_viewer_node')
rospy.Subscriber('/multispectral/detection_image', Image, image_callback)
rospy.spin()
