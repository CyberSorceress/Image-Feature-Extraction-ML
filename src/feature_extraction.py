import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from skimage import filters
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobile_preprocess
from tensorflow.keras.models import Model

def extract_hog_features(image):
    """Extract HOG features from an image."""
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def extract_lbp_features(image):
    """Extract Local Binary Pattern (LBP) features from an image."""
    lbp = local_binary_pattern(image, P=24, R=3, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(27), density=True)
    return hist

def extract_edge_features(image):
    """Extract edge features using Sobel and Canny operators."""
    sobel_edges = filters.sobel(image)
    canny_edges = cv2.Canny((image * 255).astype(np.uint8), 100, 200)
    return np.hstack((sobel_edges.ravel(), canny_edges.ravel()))

def load_pretrained_model(model_name):
    """Load a pre-trained CNN model for feature extraction."""
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        preprocess_func = vgg_preprocess
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        preprocess_func = resnet_preprocess
    elif model_name == 'MobileNet':
        base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
        preprocess_func = mobile_preprocess
    else:
        raise ValueError("Unsupported model name")
    return Model(inputs=base_model.input, outputs=base_model.output), preprocess_func

def extract_cnn_features(image, model, preprocess_func):
    """Extract deep learning features from an image using a CNN model."""
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_func(image)
    features = model.predict(image)
    return features.flatten()

def extract_features(input_dir, output_dir, method):
    """Extract features from images using the specified method and save them."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model, preprocess_func = None, None
    if method in ['VGG16', 'ResNet50', 'MobileNet']:
        model, preprocess_func = load_pretrained_model(method)
    
    feature_list = []
    labels = []
    
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        
        if method == 'HOG':
            features = extract_hog_features(image)
        elif method == 'LBP':
            features = extract_lbp_features(image)
        elif method == 'EDGE':
            features = extract_edge_features(image)
        elif method in ['VGG16', 'ResNet50', 'MobileNet']:
            features = extract_cnn_features(image, model, preprocess_func)
        else:
            raise ValueError("Unsupported feature extraction method")
        
        feature_list.append(features)
        labels.append(filename.split('_')[0])  # Assuming filename starts with class label
    
    joblib.dump((feature_list, labels), os.path.join(output_dir, f"features_{method}.pkl"))
    print(f"Feature extraction ({method}) completed. Features saved in {output_dir}")

# Example usage
extract_features("data/processed", "data/features", "HOG")
