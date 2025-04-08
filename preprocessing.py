"""
Image Preprocessing Module for Counterfeit Drug Detection System

This module provides functions for preprocessing drug packaging images before
they are fed into the CNN model for feature detection and classification.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as a numpy array
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize image to the target size.
    
    Args:
        image: Input image as numpy array
        target_size: Target size as (width, height)
        
    Returns:
        Resized image
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to range [0, 1].
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def apply_color_correction(image: np.ndarray) -> np.ndarray:
    """
    Apply color correction to enhance image features.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Color corrected image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the original A and B channels
    merged_lab = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image


def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Apply denoising to the image.
    
    Args:
        image: Input image as numpy array
        strength: Denoising strength
        
    Returns:
        Denoised image
    """
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def detect_edges(image: np.ndarray) -> np.ndarray:
    """
    Detect edges in the image using Canny edge detector.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Edge map
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges


def extract_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extract region of interest from the image.
    
    Args:
        image: Input image as numpy array
        roi: Region of interest as (x, y, width, height)
        
    Returns:
        Extracted ROI
    """
    x, y, w, h = roi
    return image[y:y+h, x:x+w]


def augment_image(image: np.ndarray, 
                  rotation_range: int = 10, 
                  zoom_range: float = 0.1,
                  brightness_range: float = 0.2) -> np.ndarray:
    """
    Apply data augmentation to the image.
    
    Args:
        image: Input image as numpy array
        rotation_range: Maximum rotation angle in degrees
        zoom_range: Maximum zoom factor
        brightness_range: Maximum brightness adjustment factor
        
    Returns:
        Augmented image
    """
    # Random rotation
    angle = np.random.uniform(-rotation_range, rotation_range)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    # Random zoom
    zoom = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    zoomed = cv2.resize(rotated, None, fx=zoom, fy=zoom)
    
    # Crop to original size if zoomed in, pad if zoomed out
    zh, zw = zoomed.shape[:2]
    if zoom > 1:
        # Zoomed in - crop center
        start_x = (zw - w) // 2
        start_y = (zh - h) // 2
        zoomed = zoomed[start_y:start_y+h, start_x:start_x+w]
    else:
        # Zoomed out - pad with border
        pad_x = (w - zw) // 2
        pad_y = (h - zh) // 2
        zoomed = cv2.copyMakeBorder(zoomed, pad_y, h-zh-pad_y, pad_x, w-zw-pad_x, cv2.BORDER_CONSTANT)
    
    # Random brightness
    brightness = np.random.uniform(1 - brightness_range, 1 + brightness_range)
    hsv = cv2.cvtColor(zoomed, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * brightness
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return brightened


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224), 
                    enhance_colors: bool = True, reduce_noise: bool = True) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size as (width, height)
        enhance_colors: Whether to apply color enhancement
        reduce_noise: Whether to apply noise reduction
        
    Returns:
        Preprocessed image ready for model input
    """
    # Load image
    image = load_image(image_path)
    
    # Apply color correction if requested
    if enhance_colors:
        image = apply_color_correction(image)
    
    # Apply denoising if requested
    if reduce_noise:
        image = denoise_image(image)
    
    # Resize to target size
    image = resize_image(image, target_size)
    
    # Normalize pixel values
    image = normalize_image(image)
    
    return image


def preprocess_batch(image_paths: List[str], target_size: Tuple[int, int] = (224, 224),
                    enhance_colors: bool = True, reduce_noise: bool = True) -> np.ndarray:
    """
    Preprocess a batch of images.
    
    Args:
        image_paths: List of paths to image files
        target_size: Target size as (width, height)
        enhance_colors: Whether to apply color enhancement
        reduce_noise: Whether to apply noise reduction
        
    Returns:
        Batch of preprocessed images as numpy array
    """
    preprocessed_images = []
    
    for image_path in image_paths:
        try:
            preprocessed = preprocess_image(image_path, target_size, enhance_colors, reduce_noise)
            preprocessed_images.append(preprocessed)
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
    
    return np.array(preprocessed_images)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Replace with actual image path
    test_image_path = "path/to/test/image.jpg"
    
    try:
        # Load and preprocess image
        original = load_image(test_image_path)
        preprocessed = preprocess_image(test_image_path)
        
        # Display results
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        
        plt.subplot(1, 2, 2)
        plt.title("Preprocessed Image")
        plt.imshow(cv2.cvtColor((preprocessed * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in example: {e}")
