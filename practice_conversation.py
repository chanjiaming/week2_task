#!/home/eevee/catkin_ws/src/fz_gemini/scripts/gemini/bin/python3

import rospy
import socket
import os
import pygame
from gtts import gTTS
from chan_gemini_config import configure_gemini

generative_model = configure_gemini()

class GeminiChat:
    def __init__(self):
        rospy.init_node("gemini_chat", anonymous=True)
        self.user_name = None
        self.conversation_over = False  
        pygame.init()  
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
        audio_file = "practice_temp_audio.mp3"  

        tts.save(audio_file)

        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            continue

        os.remove(audio_file)

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
        rospy.loginfo("Gemini: Hi there! How can I assist you today?")
        welcome_message = "Hi there! How can I assist you today?"
        self.speak(welcome_message)

        while not self.conversation_over and not rospy.is_shutdown():
            user_input = input("You: ")  

            if ("goodbye" or "bye") in user_input.lower():
                self.conversation_over = True
                farewell_message = "Goodbye! It was nice talking to you."
                rospy.loginfo(f"Gemini: {farewell_message}")
                self.speak(farewell_message)
                rospy.signal_shutdown("User ended the conversation")
                break


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
