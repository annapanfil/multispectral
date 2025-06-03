import rospy
from sensor_msgs.msg import Image
import os
import cv2

if __name__ == "__main__":
    rospy.init_node('image_publisher_node')
    pub = rospy.Publisher('/multispectral/detection_image', Image, queue_size=1)
    rate = rospy.Rate(1/3)
    image_dir = "/home/anna/Datasets/raw_images/hamburg_2025_05_19/images/0000SET"

    image_dirs = [f"{image_dir}/{i:03d}" for i in range(3)]

    images = []
    for id in image_dirs:
        images.extend(sorted([x for x in os.listdir(id) if x.endswith("_1.tif")]))

    while not rospy.is_shutdown() and len(images) > 0:
        group_key = images[0].split("_")[1]

        img = cv2.imread(f"{image_dir}/{int(group_key)//200:03}/{images.pop(0)}")

        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = group_key
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = 'rgb8'
        msg.is_bigendian = False
        msg.step = img.shape[1] * 3
        msg.data = img.tobytes()

        pub.publish(msg)

        print(f"Published {msg.header.frame_id}")
        rate.sleep()
