#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped

from pupil_labs.realtime_api.simple import discover_one_device
from pupil_labs.realtime_api.simple import Device
from sensor_msgs.msg import Image  

class PupilOverlayStreamNode:
    def __init__(self):
        rospy.init_node("pupil_gaze_overlay_node")

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/pupil/scene_with_gaze", Image, queue_size=10)  
        self.gaze_pub = rospy.Publisher("/pupil/gaze", PointStamped, queue_size=10)

        rospy.loginfo("Looking for Pupil Invisible device...")
        #ip = "10.15.178.228"
        #ip = "192.168.0.132"
        #ip = "192.168.0.69"
        ip = "137.195.113.254"
        self.device = Device(address=ip, port="8080")
        if self.device is None:
            rospy.logerr("No device found.")
            raise SystemExit(-1)
        rospy.loginfo(f"Connected to device {self.device.phone_name}")

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            try:
                frame, gaze = self.device.receive_matched_scene_video_frame_and_gaze()
                if frame is None or gaze is None:
                    rospy.logwarn("No matched data received.")
                    continue

                # Draw gaze circle
                cv2.circle(
                    frame.bgr_pixels,
                    (int(gaze.x), int(gaze.y)),
                    radius=25,
                    color=(0, 0, 255),
                    thickness=5,
                )

                # Publish as compressed image with transport parameter
                ros_img = self.bridge.cv2_to_imgmsg(frame.bgr_pixels, encoding="bgr8")
                ros_img.header.stamp = rospy.Time.now()
                self.image_pub.publish(ros_img)

                # Publish gaze as PointStamped
                gaze_msg = PointStamped()
                gaze_msg.header.stamp = ros_img.header.stamp
                gaze_msg.point.x = gaze.x
                gaze_msg.point.y = gaze.y
                gaze_msg.point.z = 0.0
                self.gaze_pub.publish(gaze_msg)

                rate.sleep()

            except Exception as e:
                rospy.logwarn(f"Error: {e}")
                continue

    def shutdown(self):
        self.device.close()
        rospy.loginfo("Node shutdown complete.")

if __name__ == "__main__":
    try:
        node = PupilOverlayStreamNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        if "node" in locals():
            node.shutdown()
