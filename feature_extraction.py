"""
Feature Extraction Module for Counterfeit Drug Detection System

This module provides functions for extracting features from drug packaging images
that can be used for counterfeit detection.
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict, Any, Optional
import os
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.models import Model


def extract_color_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Extract color histogram features from an image.
    
    Args:
        image: Input image as numpy array (BGR format)
        bins: Number of bins for the histogram
        
    Returns:
        Flattened color histogram features
    """
    # Convert to HSV color space (better for color analysis)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Compute histograms for each channel
    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    
    # Normalize histograms
    h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
    s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
    v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
    
    # Concatenate histograms
    hist_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
    
    return hist_features


def extract_texture_features(image: np.ndarray) -> np.ndarray:
    """
    Extract texture features using Haralick texture features.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Texture features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute GLCM (Gray-Level Co-occurrence Matrix)
    glcm = cv2.createGLCM(gray, 5)  # Using a distance of 5 pixels
    
    # Compute Haralick features
    texture_features = cv2.GLCMTextureFeaturesExtractor_create().compute(glcm)
    
    return texture_features


def extract_hog_features(image: np.ndarray, 
                        cell_size: Tuple[int, int] = (8, 8),
                        block_size: Tuple[int, int] = (2, 2)) -> np.ndarray:
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    
    Args:
        image: Input image as numpy array
        cell_size: Size of cell for HOG computation
        block_size: Size of block for HOG computation
        
    Returns:
        HOG features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to a standard size
    resized = cv2.resize(gray, (128, 128))
    
    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(block_size[0] * cell_size[0], block_size[1] * cell_size[1]),
        _blockStride=(cell_size[0] // 2, cell_size[1] // 2),
        _cellSize=cell_size,
        _nbins=9
    )
    
    # Compute HOG features
    hog_features = hog.compute(resized)
    
    return hog_features.flatten()


def extract_sift_features(image: np.ndarray, max_keypoints: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract SIFT (Scale-Invariant Feature Transform) features.
    
    Args:
        image: Input image as numpy array
        max_keypoints: Maximum number of keypoints to return
        
    Returns:
        Tuple of (keypoints, descriptors)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Limit the number of keypoints
    if keypoints and len(keypoints) > max_keypoints:
        # Sort keypoints by response (strength)
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:max_keypoints]
        # Get corresponding descriptors
        descriptors = descriptors[:max_keypoints]
    
    return keypoints, descriptors


def create_feature_extractor(base_model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
    """
    Create a feature extractor from a pre-trained model.
    
    Args:
        base_model: Pre-trained base model
        layer_name: Name of the layer to extract features from
        
    Returns:
        Feature extractor model
    """
    return Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(layer_name).output
    )


def extract_deep_features(image: np.ndarray, 
                         feature_extractor: tf.keras.Model,
                         preprocess_func: callable = efficientnet_preprocess) -> np.ndarray:
    """
    Extract deep features using a pre-trained CNN.
    
    Args:
        image: Input image as numpy array
        feature_extractor: Feature extractor model
        preprocess_func: Preprocessing function for the specific model
        
    Returns:
        Deep features
    """
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Preprocess the image
    preprocessed = preprocess_func(image)
    
    # Extract features
    features = feature_extractor.predict(preprocessed)
    
    # Flatten if needed
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    
    return features[0]  # Return the features for the single image


def detect_logos(image: np.ndarray, 
                logo_detector: tf.keras.Model,
                confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Detect logos in the image using a trained logo detector.
    
    Args:
        image: Input image as numpy array
        logo_detector: Trained logo detection model
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        List of detected logos with bounding boxes and confidence scores
    """
    # Preprocess image for the detector
    preprocessed = cv2.resize(image, (300, 300))
    preprocessed = preprocessed.astype(np.float32) / 255.0
    preprocessed = np.expand_dims(preprocessed, axis=0)
    
    # Run detection
    class_scores, boxes = logo_detector.predict(preprocessed)
    
    # Process results
    detections = []
    
    # Get image dimensions for scaling boxes
    height, width = image.shape[:2]
    
    for i, (class_score, box) in enumerate(zip(class_scores[0], boxes[0])):
        # Get the class with highest score
        class_id = np.argmax(class_score)
        confidence = class_score[class_id]
        
        # Filter by confidence
        if confidence >= confidence_threshold:
            # Scale box to original image dimensions
            x1, y1, x2, y2 = box
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            
            detections.append({
                'class_id': int(class_id),
                'confidence': float(confidence),
                'box': [x1, y1, x2, y2]
            })
    
    return detections


def compare_features(features1: np.ndarray, 
                    features2: np.ndarray,
                    method: str = 'cosine') -> float:
    """
    Compare two feature vectors and return a similarity score.
    
    Args:
        features1: First feature vector
        features2: Second feature vector
        method: Comparison method ('cosine', 'euclidean', or 'correlation')
        
    Returns:
        Similarity score (higher means more similar)
    """
    if method == 'cosine':
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        similarity = dot_product / (norm1 * norm2)
        
    elif method == 'euclidean':
        # Euclidean distance (converted to similarity)
        distance = np.linalg.norm(features1 - features2)
        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
        
    elif method == 'correlation':
        # Correlation coefficient
        correlation = np.corrcoef(features1, features2)[0, 1]
        similarity = (correlation + 1) / 2  # Scale from [-1, 1] to [0, 1]
        
    else:
        raise ValueError(f"Unsupported comparison method: {method}")
    
    return similarity


def extract_all_features(image: np.ndarray, 
                        deep_feature_extractor: Optional[tf.keras.Model] = None) -> Dict[str, np.ndarray]:
    """
    Extract all types of features from an image.
    
    Args:
        image: Input image as numpy array
        deep_feature_extractor: Optional deep feature extractor model
        
    Returns:
        Dictionary of feature types and their values
    """
    features = {}
    
    # Extract color histogram features
    features['color_hist'] = extract_color_histogram(image)
    
    # Extract HOG features
    features['hog'] = extract_hog_features(image)
    
    # Extract SIFT keypoints and descriptors
    keypoints, descriptors = extract_sift_features(image)
    if descriptors is not None:
        features['sift'] = descriptors
    
    # Extract deep features if extractor is provided
    if deep_feature_extractor is not None:
        features['deep'] = extract_deep_features(image, deep_feature_extractor)
    
    return features


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from tensorflow.keras.applications import EfficientNetB0
    
    # Replace with actual image path
    test_image_path = "path/to/test/image.jpg"
    
    try:
        # Load image
        image = cv2.imread(test_image_path)
        if image is None:
            raise ValueError(f"Could not load image from {test_image_path}")
        
        # Create a deep feature extractor
        base_model = EfficientNetB0(weights='imagenet', include_top=False)
        feature_extractor = create_feature_extractor(base_model, 'top_activation')
        
        # Extract features
        features = extract_all_features(image, feature_extractor)
        
        # Print feature dimensions
        for feature_name, feature_vector in features.items():
            print(f"{feature_name} features shape: {feature_vector.shape}")
        
        # Visualize HOG features
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        plt.subplot(1, 2, 2)
        plt.title("HOG Features")
        plt.plot(features['hog'])
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in example: {e}")
