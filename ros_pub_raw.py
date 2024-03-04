#!/usr/bin/env python
import os

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import cv2
from cv_bridge import CvBridge

def pub_img(img_dir, img_topic):

    rospy.init_node('raw', anonymous=True)
    img_pub = rospy.Publisher(img_topic, Image, queue_size=20)

    # Get all image paths from the img_dir
    # fpaths = [f"2d_raw_cam0_0_{n}.png" for n in range(1500)]
    fpaths = [f"2d_raw_cam0_0_{n}.jpg" for n in range(1200, 2000)]

    r = rospy.Rate(2)
    while not rospy.is_shutdown():
        for fidx, fpath  in enumerate(fpaths):
            cv_img_bgr = cv2.imread(os.path.join(img_dir, fpath))    # OpenCV (h, w, nc)
            # cv_img_rgb = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB) # PyTorch expects (B, nc, w, h)
            
            if cv_img_bgr is not None:
                ros_img = CvBridge().cv2_to_imgmsg(cv_img_bgr, "passthrough")
                ros_img.header = Header()
                ros_img.header.stamp = rospy.Time.now()
                ros_img.header.seq = fidx
                img_pub.publish(ros_img)
                rospy.loginfo("Published image %s", fpath)
            else:
                rospy.logwarn("Failed to read image %s", fpath)
            # import pdb; pdb.set_trace()

            r.sleep()


if __name__ == '__main__':
    try:
        img_dir = "/robodata/ecocar_logs/processed/CACCDataset/2d_raw/cam0/0"
        # img_dir = "/robodata/ecocar_logs/processed/CACCDataset/2d_raw/cam0/44"
        img_topic = "img_pre"
        pub_img(img_dir, img_topic)
    except rospy.ROSInterruptException:
        pass