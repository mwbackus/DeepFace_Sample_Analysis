# @title { vertical-output: true }
# Install Deepface: https://github.com/serengil/deepface
!pip install deepface
!pip install opencv-python
!pip install matplotlib

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import csv
import pandas as pd

backends = [
  'opencv',
  'ssd',
  'dlib',
  'mtcnn',
  'retinaface',
  'mediapipe',
  'yolov8',
  'yunet',
]

# List to store results
csv_file = '/content/deepface_sample_image_analysis.csv'

# Creating and writing initial header line to the CSV file
init_data = ["Filename", "Gender", "Race/Ethnicity", "Age"]
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(init_data)

# Directory containing multiple image files
# Images from 'faceimages' are uploaded to the root of the Google Collab files tab
directory_path = '/content/'

# Collect/List all files in the directory
image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

for image_file in image_files:
    # Skip files that are not images (you can add more image extensions if needed)
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Get individual image
    image_to_process_path = os.path.join(directory_path, image_file)

    # Perform DeepFace Facial Analysis for age, gender, race
    result = DeepFace.analyze(img_path=image_to_process_path, detector_backend=backends[4], actions=['age', 'gender', 'race'])

    # Collect information needed from results
    for item in result:
        dominant_gender = item['dominant_gender'].capitalize()
        dominant_race = item['dominant_race'].capitalize()
        age = item['age']

        image_file_name = os.path.basename(image_to_process_path)
        additional_data = [image_file_name, dominant_gender, dominant_race, age]

        # Appending additional data to the CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(additional_data)

# Done!
print(f"\nImage analysis results saved to '{csv_file}'")
