#!/usr/bin/python 

"""Capture images from a camera and save them to a ROS topic with altitude information."""

import subprocess
import requests
import rospy
from geometry_msgs.msg import PointStamped


"""Capture the images on the current altitude."""

current_altitude = 0
printing_idx = 0

def position_callback(msg):
    global current_altitude
    current_altitude = msg.point.z   
    
    global printing_idx
    if printing_idx % 20 == 0:
        rospy.loginfo("Current altitude: {}".format(current_altitude))
    printing_idx += 1

def almost_equal(a, b, epsilon):
    return abs(a - b) <= epsilon

if __name__ == "__main__":
    rospy.init_node("Automatic_photo_taker")
    rospy.loginfo("Node has been started")

    url = "http://192.168.1.83/capture"
    params = {
        "block": "true"
    }

    try:
        output = subprocess.check_output(["ping", "-c", "1", "192.168.1.83"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        rospy.logerr("Failed to ping the camera")
        exit(1)

    rate = rospy.Rate(50)
    position_sub = rospy.Subscriber("/dji_osdk_ros/local_position", PointStamped, callback=position_callback)   
    trigger_pub = rospy.Publisher("/camera/trigger", PointStamped, queue_size=10)

    trigger_rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        print("Capturing photo on altitude: " + str(current_altitude))
        response = requests.get(url, params=params)
        if response.status_code == 200 and "raw_storage_path" in response.json().keys():
            img_name = response.json().get("raw_storage_path").get("1") # name of channel 1 image
            trigger_msg=PointStamped()
            trigger_msg.header.frame_id = img_name
            trigger_msg.point.z = current_altitude
            trigger_msg.header.stamp = rospy.Time.now()
            trigger_pub.publish(trigger_msg)
            trigger_rate.sleep()
        else:
            rospy.logwarn("Couldn't capture image")
            print(response.json())

