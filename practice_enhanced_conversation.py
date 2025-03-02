#!/home/eevee/catkin_ws/src/fz_gemini/gtts_venv/bin/python3

import subprocess
import json
import rospy
import cv2
import pygame
from gtts import gTTS
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

DATABASE_PATH = "/home/eevee/catkin_ws/src/fz_gemini/scripts/database"
FACE_IMAGE_PATH = "/home/eevee/catkin_ws/src/fz_gemini/scripts/temp_face.jpg"

class GeminiChat:
    def __init__(self):
        rospy.init_node("gemini_chat", anonymous=True)

        self.bridge = CvBridge()
        self.face_detected = False
        self.image_sub = rospy.Subscriber("/user_image", Image, self.image_callback)

        pygame.init()  # Initialize Pygame for audio
        rospy.loginfo("Gemini Chat Initialized")
        self.speak("Gemini Chat Initialized")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite(FACE_IMAGE_PATH, frame)
            self.face_detected = True
        except Exception as e:
            rospy.logerr(f"Error saving image: {e}")
            rospy.signal_shutdown("Error saving image.")

    def speak(self, text):
        rospy.loginfo(f"Speaking: {text}")  # Log the text being spoken
        tts = gTTS(text=text, lang='en')
        audio_file = "practice_temp_audio.mp3"  

        tts.save(audio_file)
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            rospy.sleep(0.1)  # Wait for speech to finish

        os.remove(audio_file)

    def check_face_recognition(self):
        """Run DeepFace subprocess to recognize face."""
        try:
            self.speak("Checking face recognition, please wait.")
            result = subprocess.run(
                ["/home/eevee/catkin_ws/src/fz_gemini/deepface_venv/bin/python3",
                "/home/eevee/catkin_ws/src/fz_gemini/scripts/practice_face_recognition.py",
                FACE_IMAGE_PATH],
                capture_output=True,
                text=True
            )

            if result.stderr:
                rospy.logerr(f"Face recognition script error: {result.stderr}")
                self.speak("An error occurred while recognizing the face.")
                return {"status": "error"}

            output = result.stdout.strip()
            if not output:
                rospy.logerr("Face recognition script returned empty output.")
                self.speak("Face recognition did not return any data.")
                return {"status": "error"}

            response = json.loads(output)
            return response

        except json.JSONDecodeError:
            rospy.logerr("Face recognition script returned invalid JSON.")
            self.speak("Face recognition encountered a JSON error.")
            return {"status": "error"}

        except Exception as e:
            rospy.logerr(f"Face recognition error: {e}")
            self.speak("Face recognition encountered an error.")
            return {"status": "error"}


    def run(self):
        #self.speak("Waiting for face image.")
        rospy.loginfo("Waiting for face image...")

        while not self.face_detected and not rospy.is_shutdown():
            rospy.sleep(0.5)  # Wait for image

        face_result = self.check_face_recognition()

        if face_result["status"] == "recognized":
            rospy.loginfo(f"Recognized user: {face_result['name']}")
            self.speak(f"Welcome back, {face_result['name']}.")
            self.start_chat()
        else:
            rospy.loginfo("New user detected, asking questions...")
            self.speak("New user detected. Let's set up your profile.")
            self.ask_user_info()

    def ask_user_info(self):
        """Ask for user's name & favorite thing, then save data."""
        print("Welcome! Before we start, may I know your name?")
        self.speak("Welcome! Before we start, may I know your name?")
        user_name = input("you: ")

        print (f"Nice to meet you, {user_name}. What is your favorite thing?")
        self.speak(f"Nice to meet you, {user_name}. What is your favorite thing?")
        fav_thing = input("You: ")

        # Create user directory
        user_dir = f"{DATABASE_PATH}/{user_name}"
        os.makedirs(user_dir, exist_ok=True)

        # Save the captured face
        cv2.imwrite(f"{user_dir}/face.jpg", cv2.imread(FACE_IMAGE_PATH))

        # Save user data in JSON
        data_path = "/home/eevee/catkin_ws/src/fz_gemini/scripts/user_data.json"
        with open(data_path, "a") as file:
            json.dump({"name": user_name, "favorite": fav_thing}, file)
            file.write("\n")  # Append new line
        print(f"Thank you, {user_name}. Your data has been saved.")
        self.speak(f"Thank you, {user_name}. Your data has been saved.")
        rospy.loginfo("User data saved!")

        self.start_chat()

    def start_chat(self):
        """Start normal Gemini chat conversation."""
        self.speak("How can I assist you today?")
        rospy.loginfo("Gemini: How can I assist you today?")

        while not rospy.is_shutdown():
            user_input = input("You: ")

            if user_input.lower() == "goodbye":
                self.speak("Goodbye! It was nice talking to you.")
                rospy.loginfo("Goodbye!")
                break

            rospy.loginfo(f"User: {user_input}")
            self.speak(f"You said: {user_input}")

if __name__ == "__main__":
    chat = GeminiChat()
    chat.run()
