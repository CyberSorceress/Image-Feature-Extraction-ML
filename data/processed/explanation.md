# Processed Data

## Overview
This folder contains **preprocessed images** from the LFW dataset.  
All images are:

Converted to **grayscale**  
Resized to **64x64 pixels**  
**Normalized** (pixel values scaled between 0 and 1)  
Saved in **`.npy` format** for efficient loading in Python  


## How to Generate Preprocessed Data
1. **Download the raw LFW dataset** from Kaggle (`atulanandjha/lfwpeople`)  
2. **Extract it to** `data/lfw/`  
3. **Run the preprocessing script**:
   ```sh
   python preprocessing.py
