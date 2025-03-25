# Image-Feature-Extraction-ML
# Image Feature Extraction and Classification

## Problem Statement & Objectives
**Problem Statement:**  
The project aims to explore various image feature extraction techniques and analyze their impact on classification performance across different machine learning models. This involves both traditional and deep learning-based feature extraction methods, followed by training classifiers and evaluating their performance.

**Objectives:**
- Conduct a **literature review** on image feature extraction techniques.
- Implement **traditional feature extraction methods** (e.g., HOG, LBP, Edge Detection).
- Utilize **deep learning-based feature extraction** (e.g., CNN-based models like ResNet, VGG, MobileNet).
- Train **different classifiers** (Logistic Regression, KNN, Decision Trees, Random Forests).
- Analyze classification performance using accuracy, precision, recall, and F1-score.
- Compare computational efficiency and robustness of different feature extraction methods.
- Document findings in a structured report and present key insights.

## Project Structure
```
├── data/                # Dataset (MNIST, CIFAR-10, LFW, or custom dataset)
├── notebooks/           # Jupyter notebooks for experiments
├── models/              # Saved trained models
├── reports/             # Literature review, analysis, and final report
├── presentation/        # Presentation slides
├── src/
│   ├── preprocess.py    # Image preprocessing (grayscale, resizing, normalization)
│   ├── feature_extraction.py  # Extract features using HOG, LBP, CNN models
│   ├── train_models.py  # Train classifiers (Logistic Regression, KNN, etc.)
│   ├── evaluation.py    # Compute accuracy, precision, recall, F1-score
├── README.md            # Project documentation (this file)
├── requirements.txt     # Python dependencies
```

## How to Run the Project
1. **Clone the repository:**
   ```sh
   git clone https://github.com/CyberSorceress/Image-Feature-Extraction-ML.git
   cd Image-Feature-Extraction-ML
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run Preprocessing:**
   ```sh
   python src/preprocess.py
   ```
4. **Extract Features:**
   ```sh
   python src/feature_extraction.py
   ```
5. **Train Classifiers:**
   ```sh
   python src/train_models.py
   ```
6. **Evaluate Performance:**
   ```sh
   python src/evaluation.py
   ```
