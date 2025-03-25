import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
RAW_DATA_DIR = "data/lfw"  # Original images
PROCESSED_DATA_DIR = "data/processed"  # Save preprocessed images

# Ensure the processed folder exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Image Processing Function
def preprocess_images(input_dir, output_dir, img_size=(64, 64)):
    """Preprocess images: convert to grayscale, resize, and normalize."""
    for person in tqdm(os.listdir(input_dir)):  # Iterate over each person's folder
        person_path = os.path.join(input_dir, person)
        output_person_path = os.path.join(output_dir, person)

        if not os.path.isdir(person_path):
            continue  # Skip non-directory files
        
        os.makedirs(output_person_path, exist_ok=True)  # Create output subfolder

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            output_img_path = os.path.join(output_person_path, img_name.replace(".jpg", ".npy"))

            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip unreadable images

            # Convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize to 64x64
            img_resized = cv2.resize(img_gray, img_size)

            # Normalize (scale pixel values to 0-1)
            img_normalized = img_resized / 255.0

            # Save processed image as .npy
            np.save(output_img_path, img_normalized)

# Run preprocessing
preprocess_images(RAW_DATA_DIR, PROCESSED_DATA_DIR)

print("Preprocessing complete. Processed images saved in 'data/processed/'.")
