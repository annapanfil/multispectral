from pathlib import Path
import cv2
import rospy
from libraries.imageprocessing.micasense import capture
from sensor_msgs.msg import Image
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

from src.processing.load import align_from_saved_matrices, find_images, get_irradiance

executor = ThreadPoolExecutor(max_workers=1)  # Limit to 4 threads

PANEL_PATH = "/home/anna/Datasets/raw_images/pool/realistic_trash/0034SET/000"
PANEL_NR = "0000"
panel_capt = None

def processing(image):
    # your long processing here
    rospy.loginfo("Processing image with shape: %s", image.shape)
    # Simulate long processing time
    rospy.sleep(1)
    rospy.loginfo("completed processing")

    
    img_type = get_irradiance(img_capt, panel_capt, display=False, vignetting=False)
    img_aligned = align_from_saved_matrices(img_capt, img_type, warp_matrices_dir, altitude, allow_closest=True, reference_band=0)
    
    cv2.imwrite("/home/anna/code/multispectral/out/image.jpg", image[:,:,:3])
    # input("Press Enter to continue...")  # Wait for user input to proceed
    # img_type = time_decorator(get_irradiance)(img_capt, panel_capt, display=False, vignetting=False)
    # img_aligned = time_decorator(align_from_saved_matrices)(img_capt, img_type, warp_matrices_dir, altitude, allow_closest=True, reference_band=0)


def callback(msg):
    img = np.frombuffer(msg.data, dtype=np.uint8)\
            .reshape(msg.height, msg.width, 5)
    executor.submit(processing, img)

def main():
    rospy.init_node('image_subscriber')
    panel_names = find_images(Path(PANEL_PATH), PANEL_NR, panel=True, no_panchromatic=True)
    global panel_capt 
    panel_capt = capture.Capture.from_filelist(panel_names)

    rospy.Subscriber('/multispectral/image', Image, callback, queue_size=1)
    rospy.spin()

if __name__ == '__main__':
    main()
