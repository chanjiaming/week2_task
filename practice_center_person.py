#!/home/eevee/catkin_ws/src/fz_gemini/gtts_venv/bin/python3

import cv2
import rospy
import subprocess
from ultralytics import YOLO
from std_msgs.msg import Bool


class YoloDetector:
    def __init__(self):
        rospy.init_node('Practice_center_person', anonymous=True)
        rospy.loginfo("Practice_center_person node initialized")
        
        self.model = YOLO('/home/eevee/catkin_ws/src/ffm_pkg/yolov8n.pt')
        self.person_class_id = 0
        self.center_threshold = 30
        self.confidence_threshold = 0.7
        self.subprocess_started = False  # Flag to track if subprocess has been started

        self.centered_pub = rospy.Publisher('/person_centered', Bool, queue_size=10)
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("Failed to open camera")
            rospy.signal_shutdown("Camera error")
        
        self.enable_streaming = True
        self.conversation_process = None  # Track the subprocess

    def draw_and_stream_bounding_box(self, frame, boxes, num_people, person_at_center):
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"People Detected: {num_people}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{person_at_center}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            rospy.signal_shutdown("User requested shutdown")

    def human_detection(self):
        self.check_subprocess()
        """Continuously captures frames and detects people."""
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("Failed to capture image")
                continue  # Skip this iteration if frame capture fails

            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width // 2
            results = self.model.predict(frame)

            person_at_center = False
            closest_distance = float('inf')
            detected_boxes = []
            num_people_above_threshold = 0

            for result in results:
                boxes = result.boxes.xyxy
                class_ids = result.boxes.cls
                confidences = result.boxes.conf  # Get the confidence scores

                for box, class_id, confidence in zip(boxes, class_ids, confidences):
                    class_id = int(class_id)
                    confidence = float(confidence)
                    if class_id == self.person_class_id and confidence > self.confidence_threshold:
                        detected_boxes.append(box)
                        num_people_above_threshold += 1
                        x1, y1, x2, y2 = map(int, box)  # Extract coordinates from the box
                        bbox_center_x = (x1 + x2) // 2  # bbox stands for bounding box
                        distance_to_center = abs(bbox_center_x - frame_center_x)

                        if distance_to_center < closest_distance:
                            closest_distance = distance_to_center
                            
            if num_people_above_threshold > 0:
                person_at_center = closest_distance <= self.center_threshold
            else:
                person_at_center = False
            self.centered_pub.publish(Bool(person_at_center))

            if self.enable_streaming:
                self.draw_and_stream_bounding_box(frame, detected_boxes, num_people_above_threshold, person_at_center)

            if person_at_center and not self.subprocess_started:
                self.start_conversation_subprocess()

    def start_conversation_subprocess(self):
        if self.conversation_process is None or self.conversation_process.poll() is not None:
            rospy.loginfo("Starting conversation subprocess...")

            try:
                self.conversation_process = subprocess.Popen([
                    "gnome-terminal", "--", "bash", "-c",
                    "/home/eevee/catkin_ws/src/fz_gemini/scripts/gemini/bin/python3 "
                    "/home/eevee/catkin_ws/src/fz_gemini/scripts/practice_conversation.py"
                ])
                self.subprocess_started = True
                rospy.loginfo("Subprocess started successfully")
            except Exception as e:
                rospy.logerr(f"Error starting subprocess: {e}")
        else:
            rospy.loginfo("Subprocess is already running.")

    def check_subprocess(self):
        if self.conversation_process is not None:
            if self.conversation_process.poll() is not None:  # Process has finished
                rospy.loginfo("Subprocess finished. Resetting flags.")
                self.subprocess_started = False  
                self.conversation_process = None  
                rospy.sleep(2)  # Give time for full cleanup before allowing restart
            else:
                rospy.logdebug("Subprocess still running: %s", self.conversation_process.poll())
        else:
            rospy.logdebug("No active subprocess.")

    def run(self):
        while not rospy.is_shutdown():
            self.human_detection()
            rospy.sleep(1)  # Sleep to prevent busy-waiting


if __name__ == '__main__':
    detector = YoloDetector()
    detector.run()
