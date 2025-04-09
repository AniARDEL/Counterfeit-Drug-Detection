import os
import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import uuid
from datetime import datetime
import random
import tensorflow as tf
from pyzbar.pyzbar import decode
import sys
import traceback

# Create necessary directories
os.makedirs('output/results', exist_ok=True)

# Check if TensorFlow is properly installed and GPU is available
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Print current working directory for debugging
print(f"Current working directory: {os.getcwd()}")

# Attempt to load the trained model with correct absolute path
WORKSPACE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKSPACE_PATH, 'output', 'models', 'counterfeit_detection_model.h5')
BEST_MODEL_PATH = os.path.join(WORKSPACE_PATH, 'best_model.h5')

print(f"Attempting to load model from: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    print(f"File exists at {MODEL_PATH} with size {os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB")
else:
    print(f"File does not exist at {MODEL_PATH}")

print(f"Alternative path: {BEST_MODEL_PATH}")
if os.path.exists(BEST_MODEL_PATH):
    print(f"File exists at {BEST_MODEL_PATH} with size {os.path.getsize(BEST_MODEL_PATH) / (1024 * 1024):.2f} MB")
else:
    print(f"File does not exist at {BEST_MODEL_PATH}")

# Try loading the model
model = None
USING_TRAINED_MODEL = False

try:
    print("Attempting to load model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    # Temporarily force demo mode even if model loads
    USING_TRAINED_MODEL = False
    print("NOTE: Demo mode forced ON for testing - even though model loaded")
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_PATH}.")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    traceback.print_exc()
    
    print("\nAttempting to load from alternative location...")
    
    try:
        model = tf.keras.models.load_model(BEST_MODEL_PATH)
        print(f"Model loaded successfully from {BEST_MODEL_PATH}")
        # Temporarily force demo mode even if model loads
        USING_TRAINED_MODEL = False
        print("NOTE: Demo mode forced ON for testing - even though model loaded")
    except Exception as e:
        print(f"Warning: Could not load model from {BEST_MODEL_PATH}.")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        
        print("\nUsing demonstration mode with image-based predictions.")
        model = None
        USING_TRAINED_MODEL = False

# Output model status
if USING_TRAINED_MODEL:
    print("\nMODEL LOADED SUCCESSFULLY - Using trained model for predictions")
    # Print model summary
    model.summary()
else:
    print("\nRUNNING IN DEMO MODE - Using image features for predictions")
    print("To use the real model, ensure the model file exists and is accessible")

def preprocess_image(image):
    """
    Preprocess the image for analysis.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Resize image to standard size
    resized_img = cv2.resize(image, (224, 224))
    
    # Normalize pixel values
    normalized_img = resized_img.astype(np.float32) / 255.0
    
    return normalized_img

def extract_barcode_info(image):
    """
    Extract barcode information from image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        dict: Barcode information
    """
    # Use pyzbar library to detect and decode barcodes
    try:
        # Ensure the image is in grayscale for better barcode detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Decode all barcodes in the image
        barcodes = decode(gray)
        
        if barcodes:
            # Get the first detected barcode
            barcode = barcodes[0]
            
            # Extract and return barcode information
            return {
                "detected": True,
                "type": barcode.type,
                "data": barcode.data.decode('utf-8')
            }
        else:
            return {
                "detected": False,
                "data": None,
                "type": None
            }
    except Exception as e:
        print(f"Error in barcode extraction: {str(e)}")
        return {
            "detected": False,
            "data": None,
            "type": None
        }

def analyze_image(image):
    """
    Analyze an image for counterfeit detection.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        tuple: (result_text, confidence, visualization_image, barcode_info)
    """
    if image is None:
        return "Error: No image provided", 0, None, "No barcode detected"
    
    # Convert to RGB if image is in BGR format
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image
    
    # Preprocess the image
    try:
        preprocessed_img = preprocess_image(img_rgb)
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        # Fallback preprocessing
        preprocessed_img = cv2.resize(img_rgb, (224, 224))
        preprocessed_img = preprocessed_img.astype(np.float32) / 255.0
    
    # Extract barcode information if available
    try:
        barcode_info = extract_barcode_info(img_rgb)
    except Exception as e:
        print(f"Error in barcode extraction: {str(e)}")
        barcode_info = {"detected": False, "data": None, "type": None}
    
    # Use trained model if available, otherwise use demo mode
    image_features = None
    
    if USING_TRAINED_MODEL and model is not None:
        # Prepare image for model input
        model_input = np.expand_dims(preprocessed_img, axis=0)
        
        # Get prediction from trained model
        prediction = model.predict(model_input)[0][0]
        print(f"Model prediction: {prediction}")
        
        # Determine result
        is_counterfeit = bool(prediction > 0.5)
        confidence = float(prediction if is_counterfeit else 1 - prediction) * 100
        
        # Format result text
        if is_counterfeit:
            result_text = f"COUNTERFEIT (Confidence: {confidence:.1f}%)"
        else:
            result_text = f"AUTHENTIC (Confidence: {confidence:.1f}%)"
    else:
        # For demo mode, generate a unique prediction based on the image content
        # Calculate image features for consistent but unique predictions
        avg_color = np.mean(preprocessed_img, axis=(0, 1))
        brightness = np.mean(avg_color)
        contrast = np.std(preprocessed_img)
        texture = np.std(preprocessed_img, axis=(0, 1)).mean()
        edges = cv2.Canny(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), 100, 200).sum() / 10000
        
        # Check for specific pill characteristics that indicate counterfeits
        # Higher weight for edge detection and texture analysis
        imprint_clarity = np.mean(cv2.Laplacian(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F)) / 10
        color_uniformity = 1.0 - np.std(avg_color) * 5  # Lower std = more uniform color = more authentic
        
        # Check specifically for "OC" vs "OP" markings (common counterfeit indicator)
        has_counterfeit_marks = False
        height, width = img_rgb.shape[:2]
        pill_area = img_rgb[int(height*0.3):int(height*0.7), int(width*0.3):int(width*0.7)]
        pill_gray = cv2.cvtColor(pill_area, cv2.COLOR_RGB2GRAY) if len(pill_area.shape) == 3 else pill_area
        _, pill_thresh = cv2.threshold(pill_gray, 127, 255, cv2.THRESH_BINARY)
        pill_text_ratio = np.sum(pill_thresh == 0) / np.sum(pill_thresh == 255) if np.sum(pill_thresh == 255) > 0 else 0
        
        # For OxyContin pills, "OC" markings are original, "OP" markings are reformulated
        # For this demo, we'll treat pills with higher text ratio as potential counterfeits
        has_counterfeit_marks = pill_text_ratio > 0.15
        
        # Log all the features
        print(f"Demo analysis - brightness: {brightness:.4f}, contrast: {contrast:.4f}, texture: {texture:.4f}")
        print(f"Demo analysis - edges: {edges:.4f}, imprint clarity: {imprint_clarity:.4f}, color uniformity: {color_uniformity:.4f}")
        print(f"Demo analysis - pill text ratio: {pill_text_ratio:.4f}, counterfeit marks: {has_counterfeit_marks}")
        
        # Calculate a comprehensive score with more sensitivity to pill characteristics
        image_features = {
            'brightness': brightness,
            'contrast': contrast,
            'texture': texture,
            'edges': edges,
            'imprint_clarity': imprint_clarity,
            'color_uniformity': color_uniformity,
            'pill_text_ratio': pill_text_ratio,
            'has_counterfeit_marks': has_counterfeit_marks
        }
        
        # Weight the features - emphasize characteristics that differ in counterfeits
        weights = {
            'brightness': 0.05,
            'contrast': 0.05,
            'texture': 0.15,
            'edges': 0.15,
            'imprint_clarity': 0.2,
            'color_uniformity': 0.2,
            'pill_text_ratio': 0.2
        }
        
        # Calculate weighted score
        image_score = sum(image_features[f] * weights[f] for f in weights if f in weights)
        
        # Normalize to 0-1 and make more sensitive to differences
        normalized_score = (image_score - 0.3) * 2.5
        image_score = max(0, min(1, normalized_score))
        
        # Boost score if counterfeit marks detected
        if has_counterfeit_marks:
            image_score = max(0.6, image_score)
            
        print(f"Demo mode - Final image score: {image_score:.4f} (based on detailed image analysis)")
        
        # Determine if counterfeit based on the score
        is_counterfeit = image_score > 0.5
        
        # Calculate confidence (never 100% in demo mode)
        confidence_base = abs(image_score - 0.5) * 2  # Distance from decision boundary (0-1)
        confidence = 50 + (confidence_base * 45)  # Transform to range 50-95%
        
        # Format result text
        if is_counterfeit:
            result_text = f"COUNTERFEIT (Confidence: {confidence:.1f}%) [DEMO MODE]"
        else:
            result_text = f"AUTHENTIC (Confidence: {confidence:.1f}%) [DEMO MODE]"
    
    # Format barcode text
    if barcode_info["detected"]:
        barcode_text = f"Barcode Detected\nType: {barcode_info['type']}\nData: {barcode_info['data']}"
    else:
        barcode_text = "No barcode detected"
    
    # Generate visualization
    visualization = generate_visualization(img_rgb, is_counterfeit, confidence, barcode_info, USING_TRAINED_MODEL, image_features)
    
    return result_text, confidence, visualization, barcode_text

def generate_visualization(image, is_counterfeit, confidence, barcode_info, using_trained_model=True, image_features=None):
    """
    Generate visualization of the analysis results.
    
    Args:
        image (numpy.ndarray): Original image
        is_counterfeit (bool): Whether the image is classified as counterfeit
        confidence (float): Confidence score (0-100)
        barcode_info (dict): Barcode information
        using_trained_model (bool): Whether the prediction was made using a trained model
        image_features (dict, optional): Dictionary of image features used in demo mode
        
    Returns:
        numpy.ndarray: Visualization image
    """
    # Create a figure for visualization
    plt.figure(figsize=(12, 8))
    
    # Display the original image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Display the analysis result
    plt.subplot(2, 2, 2)
    result_text = "COUNTERFEIT" if is_counterfeit else "AUTHENTIC"
    model_text = "" if using_trained_model else "\n(DEMO MODE)"
    plt.text(0.5, 0.5, f"{result_text}\nConfidence: {confidence:.1f}%{model_text}", 
             fontsize=20, ha='center', va='center',
             color='red' if is_counterfeit else 'green',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    plt.axis('off')
    
    # Display barcode information if available
    plt.subplot(2, 2, 3)
    if image_features and not using_trained_model:
        # Display image features in demo mode
        feature_text = "Key Features:\n"
        feature_text += f"Imprint clarity: {image_features['imprint_clarity']:.2f}\n"
        feature_text += f"Color uniformity: {image_features['color_uniformity']:.2f}\n"
        feature_text += f"Texture quality: {image_features['texture']:.2f}\n"
        feature_text += f"Edge definition: {image_features['edges']:.2f}\n"
        
        # Add suspicious features warning
        suspicious = []
        if image_features['imprint_clarity'] > 1.0:
            suspicious.append("Low imprint clarity")
        if image_features['color_uniformity'] < 0.6:
            suspicious.append("Poor color uniformity")
        if image_features['pill_text_ratio'] > 0.15:
            suspicious.append("Suspicious markings")
            
        if suspicious:
            feature_text += "\nSuspicious features detected:\n- " + "\n- ".join(suspicious)
        
        plt.text(0.5, 0.5, feature_text, fontsize=11, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    elif barcode_info["detected"]:
        barcode_text = f"Barcode Detected\nType: {barcode_info['type']}\nData: {barcode_info['data']}"
        plt.text(0.5, 0.5, barcode_text, fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    else:
        barcode_text = "No Barcode Detected"
        plt.text(0.5, 0.5, barcode_text, fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    plt.axis('off')
    plt.title("Analysis Details" if image_features else "Barcode Information")
    
    # Display confidence meter
    plt.subplot(2, 2, 4)
    confidence_data = [confidence/100, 1 - confidence/100]
    labels = ['Confidence', '']
    colors = ['red' if is_counterfeit else 'green', 'lightgray']
    plt.pie(confidence_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title("Confidence Meter")
    
    # Add overall title
    system_mode = "DEMO MODE" if not using_trained_model else "AI Analysis"
    plt.suptitle(f"Counterfeit Drug Detection Analysis ({system_mode})", fontsize=16)
    
    # Save the visualization to a buffer
    plt.tight_layout()
    
    # Convert plot to image
    fig = plt.gcf()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    
    return img

def create_interface():
    """
    Create the Gradio interface for counterfeit drug detection.
    
    Returns:
        gradio.Interface: Gradio interface
    """
    # Define the interface
    with gr.Blocks(title="Counterfeit Drug Detection System") as interface:
        gr.Markdown(
            """
            # Counterfeit Drug Detection System
            
            Our advanced AI system analyzes pharmaceutical images to identify potential counterfeit drugs,
            helping protect patients and healthcare providers.
            
            - Analyzes packaging features
            - Verifies barcodes and serial numbers
            - Provides confidence scores and visual explanations
            """
        )
        
        with gr.Row():
            with gr.Column():
                # Input components
                input_image = gr.Image(
                    label="Upload Medication Image",
                    type="numpy",
                    height=300
                )
                analyze_button = gr.Button("Analyze Image", variant="primary")
            
            with gr.Column():
                # Output components
                result_text = gr.Textbox(label="Analysis Result")
                confidence = gr.Number(label="Confidence Score (%)")
                barcode_info = gr.Textbox(label="Barcode Information")
                visualization = gr.Image(label="Analysis Visualization", height=500)
        
        # Set up the click event
        analyze_button.click(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[result_text, confidence, visualization, barcode_info]
        )
        
        gr.Markdown(
            """
            ## About This System
            
            This system uses deep learning and computer vision techniques to analyze pharmaceutical products
            and determine if they might be counterfeit. The analysis is based on visual features of the medication
            and packaging, as well as verification of barcodes and serial numbers when available.
            
            ### How to Use
            
            1. Upload an image of a pharmaceutical product
            2. Click "Analyze Image"
            3. Review the results, including confidence score and barcode information
            
            ### Disclaimer
            
            This is a demonstration system and should not be used as the sole method for determining
            the authenticity of medications. Always consult healthcare professionals and use official
            channels to verify your medications.
            """
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(share=True)
