import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Constants for hand waving detection
TIME_LIMIT = 1  # Max time for movement in one direction
WAVE_COUNT = 2  # Number of back and forth movements to consider it a wave
BUFFER_SIZE = 40  # Buffer size to track hand movements
VISIBILITY_THRESHOLD = 0.5


# Buffer for storing x-positions and time for each hand
hand_buffers_right = []
hand_buffers_left = []

initial_positions = {
    'left': None,
    'right': None
}

# Boolean flags for tracking when to initialize movement detection
movement_initialized = {
    'left': False,
    'right': False
}


def is_within_proximity(wrist_x, wrist_y, shoulder_x, shoulder_y,proximity_threshold, visibility, which_hand):
    """Checks if the wrist is within proximity of the shoulder."""
    if visibility < VISIBILITY_THRESHOLD:
        return 0
    
    distance_y = abs(wrist_y - shoulder_y)
    distance_x = abs(wrist_x - shoulder_x)
        
    if which_hand is "right":
        print (f"RIGHT Distance x: {distance_x}")
        print (f"RIGHT Distance Y: {distance_y}")
    else:
        print (f"LEFT Distance x: {distance_x}")
        print (f"LEFT Distance y: {distance_y}")
    return math.sqrt(pow(distance_y,2) + pow(distance_x,2)) <= proximity_threshold


def waving_detection(hand_buffer, noise_threshold, min_distance, which_hand):
    global hand_buffers_left, hand_buffers_right
    wave_count = 0
    last_direction = 0  # 1 for right, -1 for left
    last_position_before_direction_change = hand_buffer[0]
    movement_start_time = time.time()
    
    for i in range(1, len(hand_buffer)):
        current_position = hand_buffer[i]
        previous_position = hand_buffer[i - 1]

        # Calculate movement direction and distance (ignore noise below threshold)
        direction = np.sign(current_position - previous_position)
        if abs(current_position - previous_position) < noise_threshold:
            continue  # Ignore small noisy movements

        # Check if there's a direction change
        if direction != last_direction:
            # Calculate distance from last position before direction change

            distance = abs(current_position - last_position_before_direction_change)
            current_time = time.time()
            # Check if the distance is sufficient and time is within the limit
            time_elapsed = current_time - movement_start_time
            if distance >= min_distance and time_elapsed <= TIME_LIMIT:
                wave_count += 1
                last_direction = direction
                last_position_before_direction_change = current_position
                movement_start_time = current_time  # Update start time
            else:
                if which_hand is 'left':
                    hand_buffers_left = []
                else:
                    hand_buffers_right = []
            print(f"Wave count for {which_hand} hand: {wave_count}")


        # Check if hand returns to initial position and completes the wave
        if wave_count >= WAVE_COUNT:
            return True
    return False




cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Cannot open camera.")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Draw pose landmarks on the original frame
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the image dimensions
        h, w, _ = frame.shape

        # Extract wrist and shoulder landmarks for hand and proximity detection
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        right_visibility = right_wrist.visibility
        left_visibility = left_wrist.visibility
        #print(f"left visibility: {left_visibility}          right visibility: {right_visibility}")
        # Convert normalized wrist and shoulder coordinates to pixel coordinates
        right_wrist_x = int(right_wrist.x *w)
        right_wrist_y = int(right_wrist.y *h)
        left_wrist_x = int(left_wrist.x *w)
        left_wrist_y = int(left_wrist.y *h)
        right_shoulder_x = int(right_shoulder.x *w)
        right_shoulder_y = int(right_shoulder.y *h)
        left_shoulder_x = int(left_shoulder.x *w)
        left_shoulder_y = int(left_shoulder.y *h)

        shoulder_distance = abs(right_shoulder_x - left_shoulder_x)
        proximity_threshold = shoulder_distance * 0.8
        noise_threshold = shoulder_distance * 0.01   #IMPORTANT PARAMETER
        min_distance = shoulder_distance * 0.020
        # Check if wrists are within proximity of their respective shoulders
        if is_within_proximity(right_wrist_x, right_wrist_y, right_shoulder_x, right_shoulder_y, proximity_threshold, right_visibility, "right"):
            movement_initialized['right'] = True
            cv2.putText(frame, 'Right hand detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            hand_buffers_right.append(right_wrist_x)
        else:
            hand_buffers_right = []


        if is_within_proximity(left_wrist_x, left_wrist_y, left_shoulder_x, left_shoulder_y, proximity_threshold, left_visibility, "left"):
            movement_initialized['left'] = True
            cv2.putText(frame, 'left hand detected', (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            hand_buffers_left.append(left_wrist_x)
        else:
            hand_buffers_left = []
        
        if len(hand_buffers_right) > BUFFER_SIZE:
            hand_buffers_right.pop(0)
        if len(hand_buffers_left) > BUFFER_SIZE:
            hand_buffers_left.pop(0)

        if len(hand_buffers_right) > 1:
            if waving_detection(hand_buffers_right, noise_threshold, min_distance, 'right'):
                cv2.putText(frame, 'RIGHT: WELCOME!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

        elif len(hand_buffers_left) > 1:
            if waving_detection(hand_buffers_left, noise_threshold, min_distance, 'left'):
                cv2.putText(frame, 'LEFT: WELCOME!', (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)



    # Display the frame
    cv2.imshow('Hand Waving Detection', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
