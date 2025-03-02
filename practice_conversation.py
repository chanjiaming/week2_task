#!/home/eevee/catkin_ws/src/fz_gemini/scripts/gemini/bin/python3

import socket
import os
import pygame
from gtts import gTTS
import rospy
from chan_gemini_config import configure_gemini
import json

# Configure the Google Generative AI
generative_model = configure_gemini()

class GeminiChat:
    def __init__(self):
        self.user_name = None
        self.conversation_over = False  # Track when the user says goodbye
        pygame.init()  # Initialize Pygame
        rospy.loginfo("Gemini Chat Initialized")

    def check_internet_connection(self):
        try:
            socket.create_connection(("8.8.8.8", 53 ), timeout=3)
            return True
        except OSError:
            return False

    def speak(self, text):
        """Speak the given text using gTTS and play it using pygame."""
        tts = gTTS(text=text, lang='en')
        audio_file = "practice_temp_audio.mp3"  # Temporary audio file

        # Save the audio file
        tts.save(audio_file)

        # Load and play the audio file
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Wait until the audio finishes playing
        while pygame.mixer.music.get_busy():
            continue

        # Remove the temporary audio file
        os.remove(audio_file)

    def collect_user_info(self):
        """Collect user's name and favorite thing before starting normal conversation."""
        self.speak("Hello! Before we start, may I know your name?")
        self.user_name = input("Gemini: Hello! Before we start, may I know your name?\nYou: ")

        self.speak(f"Nice to meet you, {self.user_name}! What is your favorite thing?")
        self.favorite_thing = input(f"Gemini: Nice to meet you, {self.user_name}! What is your favorite thing?\nYou: ")

        self.speak(f"{self.favorite_thing} is a great choice!")
        rospy.loginfo(f"User Info - Name: {self.user_name}, Favorite: {self.favorite_thing}")
        self.save_user_data()

    def save_user_data(self):
        """Save user data to a JSON file."""
        user_data = {
            "name": self.user_name,
            "favorite": self.favorite_thing
        }
        with open("/home/eevee/catkin_ws/src/fz_gemini/scripts/user_data.json", "w") as f:
            json.dump(user_data, f, indent=4)
        rospy.loginfo("User data saved successfully.")

    def generate_gemini_response(self, prompt):
        """Generate a response using the Gemini model."""
        if generative_model:
            try:
                response = generative_model.generate_content([prompt])
                if response and response.text:
                    return response.text.strip()
                else:
                    rospy.logerr("Gemini model returned no response.")
            except Exception as e:
                rospy.logerr(f"Failed to generate response using Gemini: {e}")
        else:
            rospy.logerr("Failed to configure Generative AI. Exiting...")
            rospy.signal_shutdown("Failed to configure Generative AI")
        return None

    def run(self):
        self.collect_user_info()
        # Welcome message
        welcome_message = "How can I assist you today?"
        self.speak(welcome_message)
        rospy.loginfo("Gemini: How can I assist you today?")

        while not self.conversation_over and not rospy.is_shutdown():
            user_input = input("You: ")  # Get user input from the terminal

            if "goodbye" or "bye" in user_input.lower():
                self.conversation_over = True
                farewell_message = "Goodbye! It was nice talking to you."
                rospy.loginfo(f"Gemini: {farewell_message}")
                self.speak(farewell_message)
                rospy.signal_shutdown("User ended the conversation")
                break

            # Generate response using Gemini model
            bot_response = self.generate_gemini_response(user_input)

            if bot_response:
                rospy.loginfo(f"Gemini: {bot_response}")
                self.speak(bot_response)
            else:
                error_message = "I didn't quite understand that. Could you please repeat?"
                rospy.loginfo(f"Gemini: {error_message}")
                self.speak(error_message)

if __name__ == "__main__":
    gemini_chat = GeminiChat()
    gemini_chat.run()
