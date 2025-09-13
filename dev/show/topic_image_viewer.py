"""Get images from a topic and optionally draw rectangles on them and publish the positions for debugging purposes."""

import rospy
import numpy as np
import cv2

from src.main_ground import send_outcomes
from src.shapes import Rectangle
from src.config import DETECTION_IMAGE_OUT_TOPIC, PILE_PIXEL_POSITION_OUT_TOPIC
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image


drawing = False
ix, iy = -1, -1
rect = None
latest_img = None
latest_img_name = None
original_image_size = None
window_size = (669, 500)

def image_callback(msg):
    global latest_img, latest_img_name, rect, original_image_size
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
    rect = None
    latest_img = img
    latest_img_name = msg.header.frame_id
    original_image_size = (msg.width, msg.height)

def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        rect = (ix, iy, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (ix, iy, x, y)
        print(f"Rectangle coordinates: ({ix}, {iy}) to ({x}, {y})")

def process_rectangle(image_pub, pos_pixel_pub):
    global latest_img, rect
    if latest_img is not None and rect is not None:
        x1, y1, x2, y2 = rect
        x1 = int(original_image_size[0] / window_size[0] * x1)
        y1 = int(original_image_size[1] / window_size[1] * y1)
        x2 = int(original_image_size[0] / window_size[0] * x2)
        y2 = int(original_image_size[1] / window_size[1] * y2)
        cv2.rectangle(latest_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        bbox = Rectangle(x1, y1, x2, y2, "pile")
        send_outcomes([bbox], latest_img, f"{latest_img_name}_handdraw", image_pub, pos_pixel_pub)


if __name__ == "__main__":
    rospy.init_node('multispectral_image_viewer_node')
    rospy.loginfo("Starting multispectral image viewer node...")
    rospy.loginfo("Waiting for image topic...")
    cv2.namedWindow("Image")
    cv2.resizeWindow("Image", window_size[0], window_size[1])
    cv2.setMouseCallback("Image", draw_rect)
    rospy.Subscriber(DETECTION_IMAGE_OUT_TOPIC, Image, image_callback)

    pos_pixel_pub = rospy.Publisher(PILE_PIXEL_POSITION_OUT_TOPIC, PointStamped, queue_size=10)
    image_pub = rospy.Publisher(DETECTION_IMAGE_OUT_TOPIC, Image, queue_size=10)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        if latest_img is not None:
            img = latest_img.copy()
            img = cv2.resize(img, window_size)
            if rect:
                x1, y1, x2, y2 = rect
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Image", img)

            if cv2.getWindowProperty("Image", 0) < 0:
                rospy.signal_shutdown("Window closed by user")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                # Publish the rectangle position
                process_rectangle(image_pub, pos_pixel_pub)

            if key == 27: # Escape key
                # Exit program
                rospy.signal_shutdown("Window closed by user")
                break

        rate.sleep()


    print("Shutting down...")
    cv2.destroyAllWindows()
