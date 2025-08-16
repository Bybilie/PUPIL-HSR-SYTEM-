#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from ultralytics_ros.msg import YoloResult
import time

COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}

class FixationDetectionNode:
    def __init__(self):
        rospy.init_node("fixation_detection_node")

        # Subscribers
        self.gaze_sub = rospy.Subscriber("/pupil/gaze", PointStamped, self.gaze_callback)
        self.yolo_sub = rospy.Subscriber("/yolo_result_pupil", YoloResult, self.yolo_callback)

        # Publisher
        self.fixation_pub = rospy.Publisher("fixated_object", String, queue_size=10)

        
        self.gaze = None 
        self.detections = []  
        self.fixation_threshold = 1.5  
        self.gaze_times = {}  # dict {class_name: start_time}

        rospy.loginfo("FixationDetectionNode initialized.")

    def gaze_callback(self, msg):
        self.gaze = msg.point

    def yolo_callback(self, msg):
        self.detections = msg.detections.detections
        self.check_fixation()

    def check_fixation(self):
        if self.gaze is None or not self.detections:
            return

        current_time = time.time()
        gaze_x = self.gaze.x
        gaze_y = self.gaze.y

        classes_in_gaze = set()
        candidate_boxes = []

        for detection in self.detections:
            bbox = detection.bbox
            center_x = bbox.center.x
            center_y = bbox.center.y
            size_x = bbox.size_x
            size_y = bbox.size_y

            xmin = center_x - size_x / 2
            xmax = center_x + size_x / 2
            ymin = center_y - size_y / 2
            ymax = center_y + size_y / 2

            if not detection.results:
                continue

            class_id = detection.results[0].id
            class_name = COCO_CLASSES.get(class_id, "unknown")

            if xmin <= gaze_x <= xmax and ymin <= gaze_y <= ymax:
                area = size_x * size_y
                candidate_boxes.append((area, class_name))

        if not candidate_boxes:
            # Clear fixation timers for classes no longer fixated
            for cls in list(self.gaze_times.keys()):
                del self.gaze_times[cls]
            return

        # Pick smallest bounding box (by area)
        candidate_boxes.sort(key=lambda x: x[0])
        smallest_area, selected_class = candidate_boxes[0]
        classes_in_gaze.add(selected_class)

        # Fixation timing for selected class only
        if selected_class not in self.gaze_times:
            self.gaze_times[selected_class] = current_time
        else:
            elapsed = current_time - self.gaze_times[selected_class]
            if elapsed > self.fixation_threshold:
                rospy.loginfo(f"Fixation detected on: {selected_class}")
                self.fixation_pub.publish(String(data=selected_class))
                self.gaze_times[selected_class] = current_time

        # Clear fixation timers for classes no longer fixated
        for cls in list(self.gaze_times.keys()):
            if cls != selected_class:
                del self.gaze_times[cls]

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = FixationDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

