import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, img_size=(64, 64)):
    """
    Convert images to grayscale, resize to uniform size, and normalize pixel values.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img_resized = cv2.resize(img_gray, img_size)       # Resize
        img_normalized = img_resized / 255.0               # Normalize pixel values
        
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, (img_normalized * 255).astype(np.uint8))
    
    print(f"Preprocessing completed. Processed images saved in {output_dir}")

# Example usage
preprocess_images("data/raw", "data/processed")
