#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PointStamped
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from std_msgs.msg import String

class FixationDetectionNode:
    def __init__(self):
        rospy.init_node("fixation_detection_node")

        # Subscribers
        self.gaze_sub = rospy.Subscriber("/pupil/gaze", PointStamped, self.gaze_callback)
        self.bounding_boxes_sub = rospy.Subscriber("/pupil/darknet_ros/bounding_boxes", BoundingBoxes, self.bounding_boxes_callback)

        # Publisher
        self.fixation_pub = rospy.Publisher("fixated_object", String, queue_size=10)


        
        self.gaze = None  
        self.bounding_boxes = [] 
        self.fixation_threshold = 1.5  # Time in seconds to detect fixation
        self.gaze_times = {}  
        self.last_time = rospy.Time.now()

    def gaze_callback(self, msg):
        self.gaze = msg.point

    def bounding_boxes_callback(self, msg):
      
        self.bounding_boxes = msg.bounding_boxes

    def check_fixation(self):
        if self.gaze is None or not self.bounding_boxes:
            return  

        for box in self.bounding_boxes:
            if self.is_gaze_in_box(self.gaze, box):
                # Check if gaze is within this box for the required time
                if box.Class not in self.gaze_times:
                    self.gaze_times[box.Class] = rospy.Time.now()
                else:
                    # If gaze stays within the box, check if threshold time is surpassed
                    if (rospy.Time.now() - self.gaze_times[box.Class]).to_sec() > self.fixation_threshold:
                        rospy.loginfo(f"The user is interested in: {box.Class}")
                        self.fixation_pub.publish(String(data=box.Class))
                        # Reset the timer for that object once fixation is detected
                        self.gaze_times[box.Class] = rospy.Time.now()
            else:
                # Reset timer if gaze leaves the box
                if box.Class in self.gaze_times:
                    del self.gaze_times[box.Class]

    def is_gaze_in_box(self, gaze, box):
        # Check if the gaze is within the box's x and y range
        if box.xmin <= gaze.x <= box.xmax and box.ymin <= gaze.y <= box.ymax:
            return True
        return False

    def run(self):
        rate = rospy.Rate(30)  
        while not rospy.is_shutdown():
            self.check_fixation()
            rate.sleep()

if __name__ == "__main__":
    try:
        node = FixationDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

