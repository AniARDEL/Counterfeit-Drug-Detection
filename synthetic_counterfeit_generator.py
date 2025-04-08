#!/usr/bin/env python3
"""
Synthetic Counterfeit Generator for Pharmaceutical Images

This script generates synthetic counterfeit pharmaceutical images by applying various
transformations to authentic pill images. The transformations are designed to mimic
common characteristics of counterfeit medications, such as color variations, texture
differences, shape distortions, and printing defects.

Usage:
    python synthetic_counterfeit_generator.py

Author: Manus AI
Date: April 2025
"""

import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import argparse
import glob
from pathlib import Path
import shutil

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class CounterfeitGenerator:
    """Class for generating synthetic counterfeit pharmaceutical images."""
    
    def __init__(self, input_dir, output_dir, num_counterfeits_per_image=3):
        """
        Initialize the counterfeit generator.
        
        Args:
            input_dir (str): Directory containing authentic pill images
            output_dir (str): Directory to save synthetic counterfeit images
            num_counterfeits_per_image (int): Number of counterfeit variants to generate per authentic image
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_counterfeits_per_image = num_counterfeits_per_image
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # List of transformation functions
        self.transformations = [
            self.alter_color,
            self.modify_texture,
            self.distort_shape,
            self.modify_imprint,
            self.degrade_quality
        ]
    
    def alter_color(self, image, intensity=0.5):
        """
        Alter the color of the pill image.
        
        Args:
            image (numpy.ndarray): Input image
            intensity (float): Intensity of color alteration (0.0-1.0)
            
        Returns:
            numpy.ndarray: Color-altered image
        """
        # Convert to HSV for easier color manipulation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Randomly shift hue
        hue_shift = random.uniform(-30, 30) * intensity
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Randomly adjust saturation
        sat_factor = random.uniform(0.7, 1.3) * intensity + (1 - intensity)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        
        # Randomly adjust value (brightness)
        val_factor = random.uniform(0.8, 1.2) * intensity + (1 - intensity)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_factor, 0, 255)
        
        # Convert back to BGR
        hsv = hsv.astype(np.uint8)
        altered_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return altered_image
    
    def modify_texture(self, image, intensity=0.5):
        """
        Modify the texture of the pill image.
        
        Args:
            image (numpy.ndarray): Input image
            intensity (float): Intensity of texture modification (0.0-1.0)
            
        Returns:
            numpy.ndarray: Texture-modified image
        """
        # Apply slight blur to simulate smoother texture
        if random.random() < 0.5:
            kernel_size = int(3 + 4 * intensity)
            if kernel_size % 2 == 0:  # Ensure kernel size is odd
                kernel_size += 1
            modified_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            # Add noise to simulate rougher texture
            noise = np.random.normal(0, 15 * intensity, image.shape).astype(np.int16)
            modified_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return modified_image
    
    def distort_shape(self, image, intensity=0.5):
        """
        Apply subtle distortion to the pill shape.
        
        Args:
            image (numpy.ndarray): Input image
            intensity (float): Intensity of shape distortion (0.0-1.0)
            
        Returns:
            numpy.ndarray: Shape-distorted image
        """
        height, width = image.shape[:2]
        
        # Create distortion grid
        distortion_scale = 10 * intensity
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                map_x[y, x] = x + distortion_scale * np.sin(y / 20)
                map_y[y, x] = y + distortion_scale * np.cos(x / 20)
        
        # Apply distortion
        distorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        
        return distorted_image
    
    def modify_imprint(self, image, intensity=0.5):
        """
        Modify the imprint on the pill (text, logo, etc.).
        
        Args:
            image (numpy.ndarray): Input image
            intensity (float): Intensity of imprint modification (0.0-1.0)
            
        Returns:
            numpy.ndarray: Image with modified imprint
        """
        # Convert to grayscale to detect edges/imprints
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate edges to get imprint regions
        kernel = np.ones((3, 3), np.uint8)
        imprint_mask = cv2.dilate(edges, kernel, iterations=1)
        
        # Create a blurred version of the imprint
        blur_factor = int(3 + 5 * intensity)
        if blur_factor % 2 == 0:
            blur_factor += 1
        blurred_image = cv2.GaussianBlur(image, (blur_factor, blur_factor), 0)
        
        # Replace imprint areas with blurred version at varying degrees
        alpha = random.uniform(0.3, 0.8) * intensity
        mask_3d = np.stack([imprint_mask] * 3, axis=2) / 255.0
        modified_image = image * (1 - mask_3d * alpha) + blurred_image * (mask_3d * alpha)
        
        return modified_image.astype(np.uint8)
    
    def degrade_quality(self, image, intensity=0.5):
        """
        Degrade the overall image quality.
        
        Args:
            image (numpy.ndarray): Input image
            intensity (float): Intensity of quality degradation (0.0-1.0)
            
        Returns:
            numpy.ndarray: Quality-degraded image
        """
        # Reduce resolution
        scale_factor = 1 - 0.5 * intensity
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        
        # Downsample and upsample to reduce quality
        downsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        upsampled = cv2.resize(downsampled, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Add JPEG compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(100 - 70 * intensity)]
        _, encoded_img = cv2.imencode('.jpg', upsampled, encode_param)
        degraded_image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        return degraded_image
    
    def generate_counterfeit(self, image_path):
        """
        Generate synthetic counterfeit images from an authentic pill image.
        
        Args:
            image_path (str): Path to authentic pill image
            
        Returns:
            list: List of generated counterfeit images
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                return []
            
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            counterfeits = []
            
            # Generate multiple counterfeit variants
            for i in range(self.num_counterfeits_per_image):
                # Start with the original image
                counterfeit = image.copy()
                
                # Apply random transformations with varying intensities
                num_transforms = random.randint(2, len(self.transformations))
                selected_transforms = random.sample(self.transformations, num_transforms)
                
                for transform_func in selected_transforms:
                    intensity = random.uniform(0.3, 0.8)
                    counterfeit = transform_func(counterfeit, intensity)
                
                # Save the counterfeit image
                output_path = os.path.join(self.output_dir, f"{base_name}_counterfeit_{i+1}.jpg")
                cv2.imwrite(output_path, counterfeit)
                
                counterfeits.append(output_path)
            
            return counterfeits
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return []
    
    def generate_all_counterfeits(self):
        """
        Generate synthetic counterfeits for all authentic images in the input directory.
        
        Returns:
            int: Number of counterfeit images generated
        """
        # Get all image files in the input directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(self.input_dir, '**', ext), recursive=True))
        
        total_counterfeits = 0
        
        # Process each image
        for image_path in tqdm(image_files, desc="Generating counterfeits"):
            counterfeits = self.generate_counterfeit(image_path)
            total_counterfeits += len(counterfeits)
        
        return total_counterfeits


def process_c3pi_dataset(c3pi_dir, output_dir):
    """
    Process the C3PI dataset to extract authentic reference images and organize them.
    
    Args:
        c3pi_dir (str): Directory containing the C3PI dataset
        output_dir (str): Directory to save organized authentic images
        
    Returns:
        str: Path to the directory containing organized authentic images
    """
    reference_dir = os.path.join(c3pi_dir, 'reference')
    authentic_dir = os.path.join(output_dir, 'authentic')
    os.makedirs(authentic_dir, exist_ok=True)
    
    # Find all image files in the reference directory
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(reference_dir, '**', ext), recursive=True))
    
    print(f"Found {len(image_files)} reference images in C3PI dataset")
    
    # Copy a subset of images to the authentic directory
    max_images = min(1000, len(image_files))  # Limit to 1000 images
    selected_images = random.sample(image_files, max_images)
    
    for i, image_path in enumerate(tqdm(selected_images, desc="Organizing authentic images")):
        dest_path = os.path.join(authentic_dir, f"authentic_c3pi_{i+1}{os.path.splitext(image_path)[1]}")
        shutil.copy(image_path, dest_path)
    
    return authentic_dir


def process_trumedicines_dataset(trumedicines_file, output_dir):
    """
    Process the TruMedicines dataset file.
    
    Args:
        trumedicines_file (str): Path to the TruMedicines dataset file
        output_dir (str): Directory to save extracted images
        
    Returns:
        str: Path to the directory containing extracted images
    """
    # This is a placeholder function as we don't know the exact format of the TruMedicines dataset
    # In a real implementation, this would extract and organize the images from the dataset file
    
    print(f"Note: TruMedicines dataset processing is a placeholder. The actual implementation would depend on the dataset format.")
    
    # Create a directory for TruMedicines authentic images
    authentic_dir = os.path.join(output_dir, 'authentic_trumedicines')
    os.makedirs(authentic_dir, exist_ok=True)
    
    return authentic_dir


def process_ultralytics_dataset(ultralytics_dir, output_dir):
    """
    Process the Ultralytics Medical Pills dataset.
    
    Args:
        ultralytics_dir (str): Directory containing the Ultralytics dataset
        output_dir (str): Directory to save organized images
        
    Returns:
        str: Path to the directory containing organized images
    """
    train_dir = os.path.join(ultralytics_dir, 'train', 'images')
    valid_dir = os.path.join(ultralytics_dir, 'valid', 'images')
    authentic_dir = os.path.join(output_dir, 'authentic_ultralytics')
    os.makedirs(authentic_dir, exist_ok=True)
    
    # Find all image files in the train and valid directories
    image_files = []
    for dir_path in [train_dir, valid_dir]:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(dir_path, ext)))
    
    print(f"Found {len(image_files)} images in Ultralytics dataset")
    
    # Copy images to the authentic directory
    for i, image_path in enumerate(tqdm(image_files, desc="Organizing Ultralytics images")):
        dest_path = os.path.join(authentic_dir, f"authentic_ultralytics_{i+1}{os.path.splitext(image_path)[1]}")
        shutil.copy(image_path, dest_path)
    
    return authentic_dir


def main():
    """Main function to run the synthetic counterfeit generator."""
    parser = argparse.ArgumentParser(description="Generate synthetic counterfeit pharmaceutical images")
    parser.add_argument("--c3pi_dir", type=str, default="/home/ubuntu/counterfeit_detection_project/data/raw/c3pi/sampleData",
                        help="Directory containing the C3PI dataset")
    parser.add_argument("--trumedicines_file", type=str, 
                        default="/home/ubuntu/counterfeit_detection_project/data/raw/trumedicines/TruMedicines-Pharmaceutical-images20k.dataset",
                        help="Path to the TruMedicines dataset file")
    parser.add_argument("--ultralytics_dir", type=str, 
                        default="/home/ubuntu/counterfeit_detection_project/data/raw/ultralytics",
                        help="Directory containing the Ultralytics Medical Pills dataset")
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/counterfeit_detection_project/data/processed",
                        help="Directory to save processed and synthetic images")
    parser.add_argument("--num_counterfeits", type=int, default=3,
                        help="Number of counterfeit variants to generate per authentic image")
    
    args = parser.parse_args()
    
    # Create output directories
    authentic_output_dir = os.path.join(args.output_dir, "authentic")
    counterfeit_output_dir = os.path.join(args.output_dir, "counterfeit")
    os.makedirs(authentic_output_dir, exist_ok=True)
    os.makedirs(counterfeit_output_dir, exist_ok=True)
    
    # Process datasets to extract and organize authentic images
    print("\n=== Processing C3PI Dataset ===")
    c3pi_authentic_dir = process_c3pi_dataset(args.c3pi_dir, authentic_output_dir)
    
    print("\n=== Processing TruMedicines Dataset ===")
    trumedicines_authentic_dir = process_trumedicines_dataset(args.trumedicines_file, authentic_output_dir)
    
    print("\n=== Processing Ultralytics Dataset ===")
    ultralytics_authentic_dir = process_ultralytics_dataset(args.ultralytics_dir, authentic_output_dir)
    
    # Generate synthetic counterfeits from authentic images
    print("\n=== Generating Synthetic Counterfeits ===")
    counterfeit_generator = CounterfeitGenerator(
        input_dir=authentic_output_dir,
        output_dir=counterfeit_output_dir,
        num_counterfeits_per_image=args.num_counterfeits
    )
    
    num_counterfeits = counterfeit_generator.generate_all_counterfeits()
    
    print(f"\nGenerated {num_counterfeits} synthetic counterfeit images")
    print(f"Authentic images directory:
(Content truncated due to size limit. Use line ranges to read in chunks)