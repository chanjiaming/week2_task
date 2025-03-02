#!/home/eevee/catkin_ws/src/fz_gemini/deepface_venv/bin/python3

import sys
import json
import os
import cv2
import numpy as np
from deepface import DeepFace

DATABASE_PATH = "/home/eevee/catkin_ws/src/fz_gemini/scripts/database"

def recognize_face(image_path):
    """Compare the input face with stored database faces."""
    try:
        for person in os.listdir(DATABASE_PATH):
            person_path = os.path.join(DATABASE_PATH, person)
            for img in os.listdir(person_path):
                stored_img_path = os.path.join(person_path, img)
                result = DeepFace.verify(image_path, stored_img_path, model_name="VGG-Face")
                if result["verified"]:
                    print(json.dumps({"status": "recognized", "name": person}))
                    return
        print(json.dumps({"status": "not recognized"}))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))

if __name__ == "__main__":
    image_path = sys.argv[1]  # Get image path from CLI argument
    recognize_face(image_path)
