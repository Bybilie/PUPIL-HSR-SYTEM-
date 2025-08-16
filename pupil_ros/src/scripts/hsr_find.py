#!/usr/bin/env python3
import rospy
import actionlib
import math
import time
from std_msgs.msg import String
from darknet_ros_msgs.msg import BoundingBoxes
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from controller_manager_msgs.srv import ListControllers

class ObjectSearchNode:
    def __init__(self):
        rospy.init_node("object_search_node")
        rospy.loginfo(f"[Time] Experiment starts! {rospy.Time.now().to_sec():.2f} seconds")

        self.target_label = None
        self.object_found = False
        self.image_width = 1280  # actual camera image width
        self.h_fov = 1.16762527  # horizontal camera FOV in radians

        self.fixed_tilt = 0.0  # no tilt
        self.current_pan = 0.0

        # Scan positions in radians
        self.scan_positions = [0.0, -0.8, 0.8]
        self.scan_index = 0

        self.last_move_time = time.time()
        self.scan_interval = 2.0 

        self.sweep_count = 0
        self.max_sweeps = 2

        self.head_client = actionlib.SimpleActionClient(
            '/hsrb/head_trajectory_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )
        rospy.loginfo("[Init] Waiting for head controller...")
        self.head_client.wait_for_server()

        # Wait for head controller to be running
        rospy.wait_for_service('/hsrb/controller_manager/list_controllers')
        list_controllers = rospy.ServiceProxy('/hsrb/controller_manager/list_controllers', ListControllers)
        while not rospy.is_shutdown():
            controllers = list_controllers().controller
            if any(c.name == 'head_trajectory_controller' and c.state == 'running' for c in controllers):
                break
            rospy.sleep(0.1)
        rospy.loginfo("[Init] Head controller is active.")

        # Subscribers
        self.fixation_sub = rospy.Subscriber("/fixated_object", String, self.fixation_callback)
        self.bbox_sub = rospy.Subscriber("/hsr/darknet_ros/bounding_boxes", BoundingBoxes, self.bbox_callback)

    def fixation_callback(self, msg):
        if self.target_label is None and not self.object_found:
            self.target_label = msg.data
            self.scan_index = 0
            self.sweep_count = 0
            rospy.loginfo(f"[Fixation] Target received: {self.target_label}")
            rospy.loginfo(f"[Time] Fixation accepted at {rospy.Time.now().to_sec():.2f} seconds")
            
            self.move_head(self.scan_positions[self.scan_index], self.fixed_tilt)
            self.last_move_time = time.time()

    def bbox_callback(self, msg):
        if self.target_label is None or self.object_found:
            return

        for box in msg.bounding_boxes:
            if box.Class == self.target_label:
                # Calculate center of bounding box in pixels
                x_center = (box.xmin + box.xmax) / 2.0

                # Normalize offset 
                x_offset = (x_center - self.image_width / 2.0) / (self.image_width / 2.0)

                # Convert normalized offset to pan angle (radians)
                pan_angle = -x_offset * (self.h_fov / 2.0)

                rospy.loginfo(f"[Found] {box.Class} at x_center={x_center:.1f}, pan offset={math.degrees(pan_angle):.2f}°")
                rospy.loginfo(f"[Time] Object found at {rospy.Time.now().to_sec():.2f} seconds")

                # Move head to face the object exactly
                rospy.loginfo(f"[Time] Facing object now{rospy.Time.now().to_sec():.2f} seconds")
            
                self.move_head(pan_angle, self.fixed_tilt)

                # Mark object as found and stop scanning
                self.object_found = True
                rospy.loginfo("[Centering] Object centered. Stopping all scans.")
                rospy.loginfo(f"[Time] Experiment ended at {rospy.Time.now().to_sec():.2f} seconds")
                return

    def move_head(self, pan, tilt):
       
        pan = max(-1.5, min(1.5, pan))
        self.current_pan = pan

        goal = FollowJointTrajectoryGoal()
        traj = JointTrajectory()
        traj.joint_names = ["head_pan_joint", "head_tilt_joint"]

        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.velocities = [0.0, 0.0]
        point.time_from_start = rospy.Duration(2.0)

        traj.points.append(point)
        goal.trajectory = traj

        self.head_client.send_goal(goal)
        self.head_client.wait_for_result()

        rospy.loginfo(f"[MoveHead] Moved to pan={math.degrees(pan):.1f}°, tilt={math.degrees(tilt):.1f}°")

    def run(self):
        rate = rospy.Rate(10) 
        while not rospy.is_shutdown():
            if self.target_label and not self.object_found:
                if time.time() - self.last_move_time > self.scan_interval:
                    self.scan_index += 1
                    if self.scan_index >= len(self.scan_positions):
                        self.scan_index = 0
                        self.sweep_count += 1
                        rospy.loginfo(f"[Sweep] Completed sweep #{self.sweep_count}")
                        if self.sweep_count >= self.max_sweeps:
                            rospy.loginfo(f"[Missed] Target '{self.target_label}' not found after {self.max_sweeps} sweeps.")
                            rospy.loginfo(f"[Time] Object missed at {rospy.Time.now().to_sec():.2f} seconds")
                            self.target_label = None
                            self.sweep_count = 0

                    if not self.object_found:
                        self.move_head(self.scan_positions[self.scan_index], self.fixed_tilt)
                        self.last_move_time = time.time()

            rate.sleep()

if __name__ == "__main__":
    try:
        node = ObjectSearchNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

