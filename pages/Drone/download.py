# This script is for downloading the model from Hugging Face and saving it locally.
# Run this script only ONCE.

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import os

# 1. Define the model name from Hugging Face and the local path to save it
model_name = "preszzz/drone-audio-detection-05-17-trial-6"
local_folder_name = "drone_model" # You can name this folder anything you like

# Create the directory if it doesn't exist
if not os.path.exists(local_folder_name):
    os.makedirs(local_folder_name)

print(f"Downloading model '{model_name}' to local folder '{local_folder_name}'...")

try:
    # 2. Download and save the feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    feature_extractor.save_pretrained(local_folder_name)
    print("Feature extractor saved successfully.")

    # 3. Download and save the model itself
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.save_pretrained(local_folder_name)
    print("Model saved successfully.")

    print(f"\n✅ All model files are now saved in the '{local_folder_name}' folder.")
    print("You can now run the main application 'interactive_sound_classifier.py'.")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    print("Please check your internet connection and the model name.")

