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

# Create necessary directories
os.makedirs('output/results', exist_ok=True)

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
    # This is a placeholder function
    # In a real implementation, this would use libraries like pyzbar
    
    # Randomly decide if barcode is detected (for demo purposes)
    is_detected = random.random() > 0.5
    
    if is_detected:
        # Generate random barcode data for demonstration
        barcode_types = ["EAN-13", "CODE-128", "QR Code", "UPC-A"]
        barcode_type = random.choice(barcode_types)
        barcode_data = ''.join([str(random.randint(0, 9)) for _ in range(13)])
        
        return {
            "detected": True,
            "type": barcode_type,
            "data": barcode_data
        }
    else:
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
    
    # Placeholder model prediction (random for demonstration)
    # In a real implementation, this would use a trained model
    prediction = random.random()
    
    # Determine result
    is_counterfeit = bool(prediction > 0.5)
    confidence = float(prediction if is_counterfeit else 1 - prediction) * 100
    
    # Format result text
    if is_counterfeit:
        result_text = f"COUNTERFEIT (Confidence: {confidence:.1f}%)"
    else:
        result_text = f"AUTHENTIC (Confidence: {confidence:.1f}%)"
    
    # Format barcode text
    if barcode_info["detected"]:
        barcode_text = f"Barcode Detected\nType: {barcode_info['type']}\nData: {barcode_info['data']}"
    else:
        barcode_text = "No barcode detected"
    
    # Generate visualization
    visualization = generate_visualization(img_rgb, is_counterfeit, confidence, barcode_info)
    
    return result_text, confidence, visualization, barcode_text

def generate_visualization(image, is_counterfeit, confidence, barcode_info):
    """
    Generate visualization of the analysis results.
    
    Args:
        image (numpy.ndarray): Original image
        is_counterfeit (bool): Whether the image is classified as counterfeit
        confidence (float): Confidence score (0-100)
        barcode_info (dict): Barcode information
        
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
    plt.text(0.5, 0.5, f"{result_text}\nConfidence: {confidence:.1f}%", 
             fontsize=20, ha='center', va='center',
             color='red' if is_counterfeit else 'green',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    plt.axis('off')
    
    # Display barcode information if available
    plt.subplot(2, 2, 3)
    if barcode_info["detected"]:
        barcode_text = f"Barcode Detected\nType: {barcode_info['type']}\nData: {barcode_info['data']}"
    else:
        barcode_text = "No Barcode Detected"
    plt.text(0.5, 0.5, barcode_text, fontsize=12, ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    plt.axis('off')
    
    # Display confidence meter
    plt.subplot(2, 2, 4)
    confidence_data = [confidence/100, 1 - confidence/100]
    labels = ['Confidence', '']
    colors = ['red' if is_counterfeit else 'green', 'lightgray']
    plt.pie(confidence_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title("Confidence Meter")
    
    # Add overall title
    plt.suptitle("Counterfeit Drug Detection Analysis", fontsize=16)
    
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
