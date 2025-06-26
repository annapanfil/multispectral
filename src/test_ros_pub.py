import rospy
from std_msgs.msg import String


if __name__ == '__main__':
    rospy.init_node('simple_talker')
    pub = rospy.Publisher('/your_topic', String, queue_size=10)

    rospy.sleep(1.0)  # wait for publisher to register
    pub.publish("hello from python")
