import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
from detection.processor_main import send_outcomes
from detection.shapes import Rectangle
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image


drawing = False
ix, iy = -1, -1
rect = None
latest_img = None
latest_img_name = None

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

def image_callback(msg):
    global latest_img, latest_img_name, rect
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
    rect = None
    latest_img = img
    latest_img_name = msg.header.frame_id

def process_rectangle(image_pub, pos_pixel_pub):
    global latest_img, rect
    if latest_img is not None and rect is not None:
        x1, y1, x2, y2 = rect
        cv2.rectangle(latest_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        bbox = Rectangle(x1, y1, x2, y2, "pile")
        send_outcomes([bbox], latest_img, f"{latest_img_name}_handdraw", image_pub, pos_pixel_pub)

if __name__ == "__main__":
    rospy.init_node('multispectral_image_viewer_node')
    rospy.loginfo("Starting multispectral image viewer node...")
    rospy.loginfo("Waiting for image topic...")
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rect)
    rospy.Subscriber('/multispectral/detection_image', Image, image_callback)

    pos_pixel_pub = rospy.Publisher("/multispectral/pile_pixel_position", PointStamped, queue_size=10)
    image_pub = rospy.Publisher("/multispectral/detection_image", Image, queue_size=10)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        if latest_img is not None:
            img = latest_img.copy()
            if rect:
                x1, y1, x2, y2 = rect
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Image", img)

            if cv2.getWindowProperty("Image", 0) < 0:
                rospy.signal_shutdown("Window closed by user")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                process_rectangle(image_pub, pos_pixel_pub)

            if key == 27: # Escape key
                rospy.signal_shutdown("Window closed by user")
                break

        rate.sleep()


    print("Shutting down...")
    cv2.destroyAllWindows()
