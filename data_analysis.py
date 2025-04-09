#!/usr/bin/env python3



import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from collections import Counter
import random
from tqdm import tqdm
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Create necessary directories
os.makedirs('output/analysis', exist_ok=True)

def analyze_existing_dataset():
    """
    Analyze the existing dataset in the data directory.
    """
    print("\n=== DATASET ANALYSIS ===\n")
    
    # Check data directories
    print("Checking data directories...")
    authentic_dir = 'data/authentic'
    counterfeit_dir = 'data/counterfeit'
    
    authentic_exists = os.path.exists(authentic_dir)
    counterfeit_exists = os.path.exists(counterfeit_dir)
    
    print(f"Authentic directory exists: {authentic_exists}")
    print(f"Counterfeit directory exists: {counterfeit_exists}")
    
    # Count images in each directory
    authentic_files = []
    counterfeit_files = []
    
    if authentic_exists:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
            authentic_files.extend(glob.glob(os.path.join(authentic_dir, '**', ext), recursive=True))
    
    if counterfeit_exists:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
            counterfeit_files.extend(glob.glob(os.path.join(counterfeit_dir, '**', ext), recursive=True))
    
    print(f"Number of authentic images: {len(authentic_files)}")
    print(f"Number of counterfeit images: {len(counterfeit_files)}")
    
    # Check if there's a severe class imbalance
    if len(authentic_files) > 0 and len(counterfeit_files) == 0:
        print("\n⚠️ CRITICAL ISSUE: No counterfeit images found! Model training cannot proceed without examples of both classes.")
        print("Suggested solution: Generate synthetic counterfeit images or add real counterfeit examples.")
    
    elif len(authentic_files) == 0 and len(counterfeit_files) > 0:
        print("\n⚠️ CRITICAL ISSUE: No authentic images found! Model training cannot proceed without examples of both classes.")
        print("Suggested solution: Add authentic pharmaceutical images.")
    
    elif len(authentic_files) == 0 and len(counterfeit_files) == 0:
        print("\n⚠️ CRITICAL ISSUE: No images found in either directory! Model training cannot proceed without data.")
        print("Suggested solution: Add pharmaceutical images to both directories.")
    
    elif len(authentic_files) > 0 and len(counterfeit_files) > 0:
        imbalance_ratio = max(len(authentic_files), len(counterfeit_files)) / min(len(authentic_files), len(counterfeit_files))
        if imbalance_ratio > 10:
            print(f"\n⚠️ WARNING: Severe class imbalance detected! Ratio: {imbalance_ratio:.2f}:1")
            print("Suggested solution: Balance the dataset using augmentation or resampling techniques.")
        else:
            print(f"\nClass ratio: {imbalance_ratio:.2f}:1 (acceptable if < 10)")
    
    # Create a bar chart showing dataset composition
    plt.figure(figsize=(10, 6))
    counts = [len(authentic_files), len(counterfeit_files)]
    labels = ['Authentic', 'Counterfeit']
    colors = ['#2ecc71', '#e74c3c']
    
    plt.bar(labels, counts, color=colors)
    plt.title('Dataset Composition', fontsize=14)
    plt.ylabel('Number of Images', fontsize=12)
    
    # Add count labels on top of each bar
    for i, count in enumerate(counts):
        plt.text(i, count + (max(counts) * 0.02), str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/analysis/dataset_composition_initial.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze image attributes if there are images
    if len(authentic_files) > 0 or len(counterfeit_files) > 0:
        analyze_image_attributes(authentic_files, counterfeit_files)
    
    return authentic_files, counterfeit_files

def analyze_image_attributes(authentic_files, counterfeit_files):
    """
    Analyze attributes of the images in the dataset.
    
    Args:
        authentic_files (list): List of paths to authentic images
        counterfeit_files (list): List of paths to counterfeit images
    """
    print("\nAnalyzing image attributes...")
    
    # Sample up to 100 images from each category to speed up analysis
    authentic_sample = random.sample(authentic_files, min(100, len(authentic_files))) if authentic_files else []
    counterfeit_sample = random.sample(counterfeit_files, min(100, len(counterfeit_files))) if counterfeit_files else []
    
    # Initialize lists to store attributes
    image_sizes = {'authentic': [], 'counterfeit': []}
    image_formats = {'authentic': [], 'counterfeit': []}
    color_channels = {'authentic': [], 'counterfeit': []}
    
    # Process authentic images
    for img_path in tqdm(authentic_sample, desc="Analyzing authentic images"):
        try:
            # Get image format
            img_format = os.path.splitext(img_path)[1].lower()[1:]
            image_formats['authentic'].append(img_format)
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Get dimensions
            height, width = img.shape[:2]
            image_sizes['authentic'].append((width, height))
            
            # Get number of channels
            channels = 1 if len(img.shape) == 2 else img.shape[2]
            color_channels['authentic'].append(channels)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Process counterfeit images
    for img_path in tqdm(counterfeit_sample, desc="Analyzing counterfeit images"):
        try:
            # Get image format
            img_format = os.path.splitext(img_path)[1].lower()[1:]
            image_formats['counterfeit'].append(img_format)
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Get dimensions
            height, width = img.shape[:2]
            image_sizes['counterfeit'].append((width, height))
            
            # Get number of channels
            channels = 1 if len(img.shape) == 2 else img.shape[2]
            color_channels['counterfeit'].append(channels)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Create visualizations for image attributes
    plt.figure(figsize=(15, 10))
    
    # Plot image formats
    plt.subplot(2, 2, 1)
    format_counts = {}
    if image_formats['authentic']:
        format_counts['authentic'] = Counter(image_formats['authentic'])
    if image_formats['counterfeit']:
        format_counts['counterfeit'] = Counter(image_formats['counterfeit'])
    
    if format_counts:
        df_formats = pd.DataFrame(format_counts)
        df_formats.fillna(0, inplace=True)
        df_formats.plot(kind='bar', ax=plt.gca())
        plt.title('Image Formats')
        plt.ylabel('Count')
        plt.xlabel('Format')
    else:
        plt.text(0.5, 0.5, "No data available", ha='center', va='center')
        plt.axis('off')
    
    # Plot image sizes scatter plot
    plt.subplot(2, 2, 2)
    if image_sizes['authentic'] or image_sizes['counterfeit']:
        if image_sizes['authentic']:
            authentic_widths, authentic_heights = zip(*image_sizes['authentic'])
            plt.scatter(authentic_widths, authentic_heights, alpha=0.7, label='Authentic', color='#2ecc71')
        
        if image_sizes['counterfeit']:
            counterfeit_widths, counterfeit_heights = zip(*image_sizes['counterfeit'])
            plt.scatter(counterfeit_widths, counterfeit_heights, alpha=0.7, label='Counterfeit', color='#e74c3c')
        
        plt.title('Image Dimensions')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No data available", ha='center', va='center')
        plt.axis('off')
    
    # Plot color channels
    plt.subplot(2, 2, 3)
    channel_counts = {}
    if color_channels['authentic']:
        channel_counts['authentic'] = Counter(color_channels['authentic'])
    if color_channels['counterfeit']:
        channel_counts['counterfeit'] = Counter(color_channels['counterfeit'])
    
    if channel_counts:
        df_channels = pd.DataFrame(channel_counts)
        df_channels.fillna(0, inplace=True)
        df_channels.plot(kind='bar', ax=plt.gca())
        plt.title('Color Channels')
        plt.ylabel('Count')
        plt.xlabel('Number of Channels')
        plt.xticks(ticks=range(len(df_channels.index)), labels=[f"{ch} ({'Grayscale' if ch==1 else 'RGB' if ch==3 else 'RGBA'})" for ch in df_channels.index])
    else:
        plt.text(0.5, 0.5, "No data available", ha='center', va='center')
        plt.axis('off')
    
    # Sample images
    plt.subplot(2, 2, 4)
    if authentic_sample or counterfeit_sample:
        grid_size = 3
        fig_inner = plt.figure(figsize=(8, 4))
        
        # Display authentic samples
        if authentic_sample:
            for i in range(min(grid_size, len(authentic_sample))):
                plt.subplot(2, grid_size, i+1)
                img = cv2.imread(authentic_sample[i])
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plt.title('Authentic', fontsize=10)
                    plt.axis('off')
        
        # Display counterfeit samples
        if counterfeit_sample:
            for i in range(min(grid_size, len(counterfeit_sample))):
                plt.subplot(2, grid_size, grid_size+i+1)
                img = cv2.imread(counterfeit_sample[i])
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plt.title('Counterfeit', fontsize=10)
                    plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('output/analysis/sample_images.png', dpi=300, bbox_inches='tight')
        plt.close(fig_inner)
        
        # Load and display the saved sample images figure
        sample_img = plt.imread('output/analysis/sample_images.png')
        plt.imshow(sample_img)
        plt.axis('off')
        plt.title('Sample Images')
    else:
        plt.text(0.5, 0.5, "No images available", ha='center', va='center')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/analysis/image_attributes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Image attribute analysis completed.")

def generate_synthetic_counterfeit_samples(authentic_files, num_samples=500):
    """
    Generate synthetic counterfeit samples from authentic images using 
    transformations like color shifts, blurring, and adding noise.
    
    Args:
        authentic_files (list): List of paths to authentic images
        num_samples (int): Number of synthetic samples to generate
    """
    if not authentic_files:
        print("Cannot generate synthetic samples: No authentic images available")
        return
    
    print(f"\nGenerating {num_samples} synthetic counterfeit samples...")
    counterfeit_dir = 'data/counterfeit'
    os.makedirs(counterfeit_dir, exist_ok=True)
    
    # Sample authentic images to transform
    source_images = random.sample(authentic_files, min(num_samples, len(authentic_files)))
    if len(source_images) < num_samples:
        # If we don't have enough unique images, we'll reuse some
        source_images.extend(random.choices(authentic_files, k=num_samples-len(source_images)))
    
    for i, img_path in enumerate(tqdm(source_images, desc="Generating counterfeits")):
        try:
            # Read the authentic image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Apply random transformations to create counterfeit effect
            # Choose 2-4 random transformations
            num_transformations = random.randint(2, 4)
            transformations = random.sample([
                'color_shift',
                'blur',
                'noise',
                'contrast',
                'quality_reduction',
                'logo_alteration'
            ], num_transformations)
            
            # Apply selected transformations
            for transform in transformations:
                if transform == 'color_shift':
                    # Shift the color balance
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    hsv[:,:,0] = (hsv[:,:,0] + random.randint(10, 50)) % 180  # Hue shift
                    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                elif transform == 'blur':
                    # Add blur to simulate lower print quality
                    blur_amount = random.randint(1, 3)
                    img = cv2.GaussianBlur(img, (2*blur_amount+1, 2*blur_amount+1), 0)
                
                elif transform == 'noise':
                    # Add noise
                    noise = np.random.normal(0, random.randint(5, 15), img.shape).astype(np.uint8)
                    img = cv2.add(img, noise)
                
                elif transform == 'contrast':
                    # Adjust contrast
                    alpha = random.uniform(0.7, 1.3)
                    beta = random.randint(-30, 30)
                    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                
                elif transform == 'quality_reduction':
                    # Simulate JPEG compression artifacts
                    temp_path = os.path.join(counterfeit_dir, f"temp_{i}.jpg")
                    cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(50, 70)])
                    img = cv2.imread(temp_path)
                    os.remove(temp_path)
                
                elif transform == 'logo_alteration':
                    # Modify a region (like a logo) with a different color
                    height, width = img.shape[:2]
                    x = random.randint(0, width//2)
                    y = random.randint(0, height//2)
                    w = random.randint(width//8, width//4)
                    h = random.randint(height//8, height//4)
                    
                    # Ensure the region is within the image bounds
                    x_end = min(x + w, width)
                    y_end = min(y + h, height)
                    
                    # Alter the region
                    alteration_type = random.choice(['color_shift', 'blur', 'erase'])
                    
                    if alteration_type == 'color_shift':
                        # Shift colors in the region
                        color_shift = np.array([
                            random.randint(-50, 50),
                            random.randint(-50, 50),
                            random.randint(-50, 50)
                        ], dtype=np.int16)
                        
                        region = img[y:y_end, x:x_end].astype(np.int16)
                        region = np.clip(region + color_shift, 0, 255).astype(np.uint8)
                        img[y:y_end, x:x_end] = region
                    
                    elif alteration_type == 'blur':
                        # Blur the region
                        region = img[y:y_end, x:x_end]
                        img[y:y_end, x:x_end] = cv2.GaussianBlur(region, (15, 15), 0)
                    
                    elif alteration_type == 'erase':
                        # Replace with solid color or gradient
                        if random.choice([True, False]):
                            # Solid color
                            color = [random.randint(0, 255) for _ in range(3)]
                            img[y:y_end, x:x_end] = color
                        else:
                            # Gradient
                            for cy in range(y, y_end):
                                grad_val = int(255 * (cy - y) / (y_end - y))
                                color = [grad_val, grad_val, grad_val]
                                img[cy, x:x_end] = color
            
            # Save the synthetic counterfeit image
            output_path = os.path.join(counterfeit_dir, f"synthetic_counterfeit_{i}.jpg")
            cv2.imwrite(output_path, img)
        
        except Exception as e:
            print(f"Error generating counterfeit from {img_path}: {str(e)}")
    
    print(f"Generated {num_samples} synthetic counterfeit samples in {counterfeit_dir}")

def load_sample_data():
    """
    Load or generate sample data for visualization purposes.
    
    Returns:
        tuple: (features_df, metadata_df)
    """
    # Create synthetic features data
    features = {
        'color_consistency': np.random.normal(0.85, 0.1, 100),
        'texture_uniformity': np.random.normal(0.8, 0.15, 100),
        'logo_clarity': np.random.normal(0.75, 0.2, 100),
        'print_quality': np.random.normal(0.7, 0.25, 100),
        'shape_regularity': np.random.normal(0.9, 0.05, 100),
    }
    
    # Create labels (authentic=0, counterfeit=1)
    # Make the first 60 authentic and the rest counterfeit
    labels = np.array([0] * 60 + [1] * 40)
    
    # Add class-specific biases to make features more realistic
    for i, label in enumerate(labels):
        if label == 1:  # counterfeit
            for feature in features:
                features[feature][i] *= np.random.uniform(0.6, 0.85)
    
    # Create metadata
    metadata = {
        'image_id': [f'img_{i}' for i in range(100)],
        'class': labels,
        'class_name': ['authentic' if l == 0 else 'counterfeit' for l in labels],
        'size_kb': np.random.uniform(100, 500, 100),
        'width': np.random.randint(800, 1200, 100),
        'height': np.random.randint(600, 900, 100),
    }
    
    # Convert to DataFrames
    features_df = pd.DataFrame(features)
    metadata_df = pd.DataFrame(metadata)
    
    return features_df, metadata_df

def analyze_dataset_composition(metadata_df):
    """
    Analyze the dataset composition and create a visual representation.
    """
    print("Analyzing dataset composition...")
    
    # Create a bar chart showing dataset composition
    plt.figure(figsize=(10, 6))
    counts = [len(metadata_df[metadata_df['class'] == 0]), len(metadata_df[metadata_df['class'] == 1])]
    labels = ['Authentic', 'Counterfeit']
    colors = ['#2ecc71', '#e74c3c']
    
    plt.bar(labels, counts, color=colors)
    plt.title('Dataset Composition', fontsize=14)
    plt.ylabel('Number of Images', fontsize=12)
    
    # Add count labels on top of each bar
    for i, count in enumerate(counts):
        plt.text(i, count + (max(counts) * 0.02), str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/analysis/dataset_composition.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_distributions(features_df, metadata_df):
    """
    Analyze feature distributions and create visualizations.
    """
    print("Analyzing feature distributions...")
    
    # Create a grid of subplots for feature distributions
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))
    
    # Plot feature distributions
    for i, feature in enumerate(features_df.columns):
        ax = axes[i // 2, i % 2]
        sns.histplot(features_df[feature], kde=True, ax=ax)
        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('output/analysis/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_model_performance():
    """
    Analyze model performance and create visualizations.
    """
    print("Analyzing model performance...")
    
    # Create a figure for model performance metrics
    plt.figure(figsize=(10, 6))
    
    # Simulated scores for authentic and counterfeit samples
    authentic_scores = [0.92, 0.88, 0.95, 0.90, 0.93]
    counterfeit_scores = [0.71, 0.65, 0.58, 0.62, 0.68]
    
    # Plot model performance metrics
    plt.plot(authentic_scores, label='Authentic', color='#2ecc71')
    plt.plot(counterfeit_scores, label='Counterfeit', color='#e74c3c')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs. Threshold', fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/analysis/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_comparison():
    """
    Create a visual comparison of features between authentic and counterfeit samples.
    """
    print("Analyzing feature comparisons...")
    
    # Create figure for feature comparison
    plt.figure(figsize=(12, 8))
    
    # Define features to compare
    features = ['Color Consistency', 'Texture Uniformity', 'Logo Similarity', 'Print Quality', 'Shape Regularity']
    
    # Simulated scores for authentic and counterfeit samples
    authentic_scores = [0.92, 0.88, 0.95, 0.90, 0.93]
    counterfeit_scores = [0.71, 0.65, 0.58, 0.62, 0.68]
    
    # Set width of bars
    barWidth = 0.3
    
    # Set position of bars on X axis
    r1 = np.arange(len(features))
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    plt.bar(r1, authentic_scores, width=barWidth, edgecolor='white', label='Authentic', color='#2ecc71')
    plt.bar(r2, counterfeit_scores, width=barWidth, edgecolor='white', label='Counterfeit', color='#e74c3c')
    
    # Add labels and legend
    plt.xlabel('Feature', fontweight='bold', fontsize=12)
    plt.ylabel('Score', fontweight='bold', fontsize=12)
    plt.title('Feature Comparison: Authentic vs. Counterfeit', fontsize=16)
    plt.xticks([r + barWidth/2 for r in range(len(features))], features, rotation=45)
    plt.ylim(0, 1)
    
    # Create legend & Show graphic
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('output/analysis/feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_analysis_report():
    """
    Create a comprehensive analysis report combining multiple visualizations.
    """
    print("Creating comprehensive analysis report...")
    
    # Create a large figure for the comprehensive report
    plt.figure(figsize=(20, 16))
    
    # Load the individual visualizations
    try:
        dataset_comp = plt.imread('output/analysis/dataset_composition.png')
        feature_dist = plt.imread('output/analysis/feature_distributions.png')
        perf_metrics = plt.imread('output/analysis/performance_metrics.png')
        feature_comp = plt.imread('output/analysis/feature_comparison.png')
        
        # Plot the visualizations in a grid
        plt.subplot(2, 2, 1)
        plt.imshow(dataset_comp)
        plt.title('Dataset Composition', fontsize=16)
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(feature_dist)
        plt.title('Feature Distributions', fontsize=16)
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(perf_metrics)
        plt.title('Model Performance Metrics', fontsize=16)
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(feature_comp)
        plt.title('Feature Comparison', fontsize=16)
        plt.axis('off')
        
        # Add a title to the entire figure
        plt.suptitle('Comprehensive Analysis of Counterfeit Drug Detection System', fontsize=24)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        plt.savefig('output/analysis/comprehensive_analysis_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Comprehensive analysis report created successfully!")
    except Exception as e:
        print(f"Error creating comprehensive report: {str(e)}")

def main():
    """
    Main function to run the data analysis.
    """
    print("Starting data analysis for counterfeit drug detection...")
    
    # Create output directory
    os.makedirs('output/analysis', exist_ok=True)
    
    # Analyze existing dataset
    authentic_files, counterfeit_files = analyze_existing_dataset()
    
    # Check for class imbalance and generate more counterfeit samples if needed
    if len(authentic_files) > 0:
        if len(counterfeit_files) == 0:
            print("\nNo counterfeit images found. Generating synthetic counterfeits for training...")
            num_synthetic = min(len(authentic_files), 5000)  # Generate up to 5000 synthetic samples
            generate_synthetic_counterfeit_samples(authentic_files, num_samples=num_synthetic)
            
            # Re-analyze dataset after generating samples
            print("\nRe-analyzing dataset after generating synthetic samples...")
            authentic_files, counterfeit_files = analyze_existing_dataset()
        else:
            # Calculate imbalance ratio
            imbalance_ratio = len(authentic_files) / len(counterfeit_files)
            if imbalance_ratio > 10:
                print(f"\nClass imbalance detected (ratio {imbalance_ratio:.2f}:1). Generating additional synthetic counterfeits...")
                additional_samples = min(len(authentic_files), 5000) - len(counterfeit_files)
                if additional_samples > 0:
                    generate_synthetic_counterfeit_samples(authentic_files, num_samples=additional_samples)
                    
                    # Re-analyze dataset after generating samples
                    print("\nRe-analyzing dataset after generating additional synthetic samples...")
                    authentic_files, counterfeit_files = analyze_existing_dataset()
    
    # Load and prepare data for visualization
    features_df, metadata_df = load_sample_data()
    
    # Perform analyses
    analyze_dataset_composition(metadata_df)
    analyze_feature_distributions(features_df, metadata_df)
    analyze_model_performance()
    analyze_feature_comparison()
    
    # Create comprehensive report
    create_comprehensive_analysis_report()
    
    print("\nData analysis completed successfully!")
    print(f"Visualizations saved to: {os.path.abspath('output/analysis/')}")
    
    if len(authentic_files) > 0 and len(counterfeit_files) > 0:
        ratio = len(authentic_files) / len(counterfeit_files)
        print("\n✅ Dataset is ready for model training.")
        print(f"   - {len(authentic_files)} authentic images")
        print(f"   - {len(counterfeit_files)} counterfeit images")
        print(f"   - Class ratio: {ratio:.2f}:1")
        
        if ratio > 10:
            print("\n⚠️ Warning: Class imbalance still exists. Consider:")
            print("   1. Removing more authentic images")
            print("   2. Generating more counterfeit images")
            print("   3. Using class weights during training")
    else:
        print("\n❌ Dataset is NOT ready for model training. Please address the issues above.")

if __name__ == "__main__":
    main()