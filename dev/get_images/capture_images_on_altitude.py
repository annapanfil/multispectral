#!/usr/bin/python 

import subprocess
import requests
import rospy
from geometry_msgs.msg import PointStamped
from src.config import TRIGGER_OUT_TOPIC, LOCAL_POSITION_IN_TOPIC


"""Fly to the to altitudes 30, 25, 20, 15, 10m above the ground in this sequence. You can be wrong by alt_err. On each altitude wait for at least 3 seconds.
photos_per_alt photos will be taken from each altitude"""

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
    altitudes = [30, 25, 20, 15, 10]
    alt_err = 0.5
    photos_per_alt = 3

    try:
        output = subprocess.check_output(["ping", "-c", "1", "192.168.1.83"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        rospy.logerr("Failed to ping the camera")
        exit(1)

    rate = rospy.Rate(50)
    position_sub = rospy.Subscriber(LOCAL_POSITION_IN_TOPIC, PointStamped, callback=position_callback)   
    trigger_pub = rospy.Publisher(TRIGGER_OUT_TOPIC, PointStamped, queue_size=10)
    
    alt_idx = 0
    n_photos_taken = 0

    rospy.loginfo("Fly to altitude {}".format(altitudes[alt_idx]))
    while not rospy.is_shutdown():
        # wait to reach the altitude
        if almost_equal(current_altitude, altitudes[alt_idx], alt_err):
            # take photos
            print("Capturing photo on altitude: " + str(current_altitude))
            response = requests.get(url, params=params)
            rospy.loginfo("Photo {} taken".format(n_photos_taken))

            trigger_msg=PointStamped()
            trigger_msg.point.z = current_altitude
            trigger_msg.header.stamp = rospy.Time.now()
            trigger_pub.publish(trigger_msg)

            n_photos_taken += 1
            if n_photos_taken >= photos_per_alt:
                alt_idx += 1
                n_photos_taken = 0
                if alt_idx > len(altitudes) - 1:
                    break
                rospy.loginfo("Fly to altitude {}".format(altitudes[alt_idx]))

    print("Took {} photos for altitudes: {}".format(photos_per_alt, ", ".join([str(a) for a in altitudes[:alt_idx]])))
    if alt_idx < len(altitudes) - 1:
        print("and {} for altitude {} and further".format(n_photos_taken, altitudes[alt_idx]))