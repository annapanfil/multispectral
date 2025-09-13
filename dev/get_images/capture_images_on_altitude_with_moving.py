#!/usr/bin/python 

import subprocess
import numpy as np
import requests
import rospy
from geometry_msgs.msg import PointStamped
from src.config import TRIGGER_OUT_TOPIC, LOCAL_POSITION_IN_TOPIC


"""Fly to the to altitudes 30, 25, 20, 15, 10m above the ground in this sequence. You can be wrong by alt_err. On each altitude wait for at least 3 seconds and move to left and rigth.
photos_per_alt photos will be taken from each altitude"""

current_altitude = 0
current_position = (0,0)
printing_idx = 0
central_position = (0, 0)
previous_position = (0, 0)



def position_callback(msg):
    global current_altitude
    current_altitude = msg.point.z   
    
    global current_position
    current_position = (msg.point.x, msg.point.y)

    global printing_idx
    if printing_idx % 100 == 0:
        rospy.loginfo("Current altitude: {}, {}m from central position, {}m from previous position".format(current_altitude, distance(current_position, central_position), distance(current_position, previous_position)), )
    printing_idx += 1

def almost_equal(a, b, epsilon):
    return abs(a - b) <= epsilon

def distance(p1, p2):
    return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5

def take_photo(url, current_altitude, trigger_pub):
    print("Capturing photo on altitude: " + str(current_altitude))
    response = requests.get(url, params=params)

    trigger_msg=PointStamped()
    trigger_msg.point.z = current_altitude
    trigger_msg.header.stamp = rospy.Time.now()
    trigger_pub.publish(trigger_msg)

if __name__ == "__main__":
    rospy.init_node("Automatic_photo_taker")
    rospy.loginfo("Node has been started")

    url = "http://192.168.1.83/capture"
    params = {
        "block": "true"
    }
    altitudes = np.array([30, 25, 20, 15, 10])
    alt_err = 0.5
    photos_per_position = 5
    pile_size = 1

    try:
        output = subprocess.check_output(["ping", "-c", "1", "192.168.1.83"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        rospy.logerr("Failed to ping the camera")
        exit(1)

    rate = rospy.Rate(50)
    position_sub = rospy.Subscriber(LOCAL_POSITION_IN_TOPIC, PointStamped, callback=position_callback)   
    trigger_pub = rospy.Publisher(TRIGGER_OUT_TOPIC, PointStamped, queue_size=10)
    
    im_heights = np.tan(np.rad2deg(49.6)) * altitudes
    im_widths = np.tan(np.rad2deg(38.3)) * altitudes

    alt_idx = 0
    n_photos_taken_position = 0
    n_sets_taken_alt = 0

    rospy.loginfo("Fly to altitude {}".format(altitudes[alt_idx]))
    while not rospy.is_shutdown():
        # wait to reach the altitude
        if almost_equal(current_altitude, altitudes[alt_idx], alt_err):
            if n_sets_taken_alt == 0:
                # set it as a center point
                central_position = current_position
                
            else:
                # go to one of the edges
                if not (distance(current_position, central_position) > im_heights[altitudes[alt_idx]]/2 - pile_size \
                    and distance(current_position, central_position) < im_widths[altitudes[alt_idx]]/2 - pile_size):
                    continue # don't take the photo, we're to close to the center
                else: previous_position = current_position
            
            if n_sets_taken_alt > 1:
                # go to the different edge
                if distance(current_position, previous_position) < im_heights[altitudes[alt_idx]]/2:
                    continue # don't take the photo, we're to close to the previous point

            # take photos
            take_photo(url, current_altitude, trigger_pub)

            n_photos_taken_position += 1
            if n_photos_taken_position >= photos_per_position:
                n_sets_taken_alt += 1
                n_photos_taken_position = 0

                if n_sets_taken_alt >= 3: # we want 3 photos in one altitude
                    alt_idx += 1
                    n_sets_taken_alt = 0
                    if alt_idx > len(altitudes) - 1:
                        break
                    rospy.loginfo("Fly to altitude {}".format(altitudes[alt_idx]))
                else:
                    rospy.loginfo("Fly to different point in the altitude {}".format(altitudes[alt_idx]))


    print("Took {} photos for altitudes: {}".format(photos_per_position, ", ".join([str(a) for a in altitudes[:alt_idx]])))
    if alt_idx < len(altitudes) - 1:
        print("and {} for altitude {} and further".format(n_photos_taken_position, altitudes[alt_idx]))
