"""
Barcode Extraction Module for Counterfeit Drug Detection System

This module provides functions for detecting and extracting barcodes, QR codes,
and serial numbers from drug packaging images.
"""

import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
from typing import List, Dict, Any, Tuple, Optional
import pytesseract
import re


def detect_barcodes(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect and decode barcodes and QR codes in an image.
    
    Args:
        image: Input image as numpy array (BGR format)
        
    Returns:
        List of dictionaries containing decoded barcode information
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to enhance barcodes
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Detect and decode barcodes
    decoded_objects = pyzbar.decode(thresh)
    
    # If no barcodes found, try with original image
    if not decoded_objects:
        decoded_objects = pyzbar.decode(image)
    
    # Process results
    results = []
    for obj in decoded_objects:
        # Get barcode type and data
        barcode_type = obj.type
        barcode_data = obj.data.decode('utf-8')
        
        # Get bounding box
        points = obj.polygon
        if len(points) > 4:
            # If more than 4 points, find the convex hull
            hull = cv2.convexHull(np.array([point for point in points]))
            hull = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            points = hull.reshape(-1, 2)
        else:
            points = np.array([[p.x, p.y] for p in points])
        
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(points)
        
        results.append({
            'type': barcode_type,
            'data': barcode_data,
            'points': points.tolist(),
            'rect': [int(x), int(y), int(w), int(h)]
        })
    
    return results


def enhance_barcode_region(image: np.ndarray, rect: List[int]) -> np.ndarray:
    """
    Enhance a barcode region to improve detection.
    
    Args:
        image: Input image as numpy array
        rect: Bounding rectangle as [x, y, width, height]
        
    Returns:
        Enhanced barcode region
    """
    # Extract region
    x, y, w, h = rect
    roi = image[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations to enhance barcode
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph


def find_potential_barcode_regions(image: np.ndarray) -> List[List[int]]:
    """
    Find potential regions that might contain barcodes.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of bounding rectangles as [x, y, width, height]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Scharr operator to detect edges in both directions
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    
    # Subtract the y-gradient from the x-gradient to get regions with high horizontal gradients
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    # Blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to close gaps and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    
    # Find contours
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on aspect ratio and area
    potential_regions = []
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio
        aspect_ratio = w / float(h)
        
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Filter based on aspect ratio and area
        if (aspect_ratio > 2.5 or aspect_ratio < 0.3) and area > 1000:
            potential_regions.append([x, y, w, h])
    
    return potential_regions


def extract_serial_number(image: np.ndarray, 
                         regex_pattern: str = r'[A-Z0-9]{8,}') -> List[Dict[str, Any]]:
    """
    Extract potential serial numbers from an image using OCR.
    
    Args:
        image: Input image as numpy array
        regex_pattern: Regular expression pattern to match serial numbers
        
    Returns:
        List of dictionaries containing extracted serial numbers and their locations
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to enhance text
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform OCR
    ocr_result = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    
    # Extract text and bounding boxes
    serial_numbers = []
    n_boxes = len(ocr_result['text'])
    
    for i in range(n_boxes):
        # Get text and confidence
        text = ocr_result['text'][i].strip()
        conf = int(ocr_result['conf'][i])
        
        # Skip empty text or low confidence results
        if not text or conf < 60:
            continue
        
        # Check if text matches serial number pattern
        if re.search(regex_pattern, text):
            # Get bounding box
            x = ocr_result['left'][i]
            y = ocr_result['top'][i]
            w = ocr_result['width'][i]
            h = ocr_result['height'][i]
            
            serial_numbers.append({
                'text': text,
                'confidence': conf,
                'rect': [x, y, w, h]
            })
    
    return serial_numbers


def extract_text_near_barcode(image: np.ndarray, 
                             barcode_rect: List[int],
                             margin: int = 50) -> str:
    """
    Extract text near a barcode using OCR.
    
    Args:
        image: Input image as numpy array
        barcode_rect: Barcode bounding rectangle as [x, y, width, height]
        margin: Margin around barcode to include in text extraction
        
    Returns:
        Extracted text
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Extract barcode coordinates
    x, y, w, h = barcode_rect
    
    # Calculate expanded region with margin
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(width, x + w + margin)
    y2 = min(height, y + h + margin)
    
    # Extract region
    roi = image[y1:y2, x1:x2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to enhance text
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform OCR
    text = pytesseract.image_to_string(thresh)
    
    return text.strip()


def parse_barcode_data(barcode_data: str) -> Dict[str, str]:
    """
    Parse barcode data to extract structured information.
    
    Args:
        barcode_data: Raw barcode data string
        
    Returns:
        Dictionary of parsed information
    """
    # Initialize result
    parsed_data = {
        'raw_data': barcode_data,
        'type': 'unknown'
    }
    
    # Check for GS1 format (commonly used in pharmaceuticals)
    if barcode_data.startswith('01') and len(barcode_data) >= 16:
        # GS1 format with GTIN-14
        parsed_data['type'] = 'GS1'
        parsed_data['gtin'] = barcode_data[2:16]
        
        # Extract additional elements if present
        remaining = barcode_data[16:]
        
        # Extract batch/lot number (AI 10)
        lot_match = re.search(r'10([^\s]+)', remaining)
        if lot_match:
            parsed_data['lot'] = lot_match.group(1)
        
        # Extract expiration date (AI 17)
        exp_match = re.search(r'17(\d{6})', remaining)
        if exp_match:
            date_str = exp_match.group(1)
            parsed_data['expiry'] = f'20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}'
        
        # Extract serial number (AI 21)
        serial_match = re.search(r'21([^\s]+)', remaining)
        if serial_match:
            parsed_data['serial'] = serial_match.group(1)
    
    # Check for HIBC format (Health Industry Bar Code)
    elif barcode_data.startswith('+'):
        parsed_data['type'] = 'HIBC'
        parts = barcode_data[1:].split('/')
        
        if len(parts) >= 1:
            parsed_data['product_code'] = parts[0]
        
        if len(parts) >= 2:
            parsed_data['lot'] = parts[1]
        
        if len(parts) >= 3:
            parsed_data['expiry'] = parts[2]
    
    # Check for NDC format (National Drug Code)
    elif re.match(r'\d{5}-\d{4}-\d{2}', barcode_data):
        parsed_data['type'] = 'NDC'
        parsed_data['ndc'] = barcode_data
    
    return parsed_data


def process_image_for_barcodes(image_path: str) -> Dict[str, Any]:
    """
    Process an image to extract all barcode and text information.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing all extracted information
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Initialize results
    results = {
        'barcodes': [],
        'serial_numbers': [],
        'potential_regions': []
    }
    
    # Detect barcodes
    barcodes = detect_barcodes(image)
    
    # Process each barcode
    for barcode in barcodes:
        # Parse barcode data
        parsed_data = parse_barcode_data(barcode['data'])
        
        # Extract text near barcode
        nearby_text = extract_text_near_barcode(image, barcode['rect'])
        
        # Add to results
        results['barcodes'].append({
            **barcode,
            'parsed_data': parsed_data,
            'nearby_text': nearby_text
        })
    
    # If no barcodes found, look for potential regions
    if not barcodes:
        potential_regions = find_potential_barcode_regions(image)
        
        # Process each potential region
        for region in potential_regions:
            # Enhance region
            enhanced = enhance_barcode_region(image, region)
            
            # Try to detect barcode in enhanced region
            region_barcodes = detect_barcodes(enhanced)
            
            if region_barcodes:
                # Add to results
                for barcode in region_barcodes:
                    # Adjust coordinates to original image
                    x, y, w, h = region
                    points = np.array(barcode['points']) + np.array([x, y])
                    rect = [barcode['rect'][0] + x, barcode['rect'][1] + y, 
                           barcode['rect'][2], barcode['rect'][3]]
                    
                    # Parse barcode data
                    parsed_data = parse_barcode_data(barcode['data'])
                    
                    # Extract text near barcode
                    nearby_text = extract_text_near_barcode(image, rect)
                    
                    # Add to results
                    results['barcodes'].append({
                        **barcode,
                        'points': points.tolist(),
                        'rect': rect,
                        'parsed_data': parsed_data,
                        'nearby_text': nearby_text
                    })
            else:
                # Add to potential regions
                results['potential_regions'].append(region)
    
    # Extract serial numbers
    serial_numbers = extract_serial_number(image)
    results['serial_numbers'] = serial_numbers
    
    return results


def visualize_results(image_path: str, results: Dict[str, Any], output_path: Optional[str] = None):
    """
    Visualize barcode detection results on the image.
    
    Args:
        image_path: Path to the original image
        results: Detection results from process_image_for_barcodes
        output_path: Path to save the visualization (optional)
    """
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Draw barcodes
    for barcode in results['barcodes']:
        # Draw polygon
        points = np.array(barcode['points'])
        cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)
        
        # Draw rectangle
        x, y, w, h = barcode['rect']
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add text
        barcode_type = barcode['type']
        barcode_data = barcode['data']
        cv2.putText(vis_image, f"{barcode_type}: {barcode_data}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw serial numbers
    for serial in results['serial_numbers']:
        # Draw rectangle
        x, y, w, h = serial['rect']
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Add text
        text = serial['text']
        conf = serial['confidence']
        cv2.putText(vis_image, f"SN: {text} ({conf}%)", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw potential regions
    for region in results['potential_regions']:
        # Draw rectangle
        x, y, w, h = region
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Add text
        cv2.putText(vis_image, "Potential Barcode", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save or display the visualization
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to {output_path}")
    else:
        # Convert to RGB for display
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        # Display using matplotlib
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image_rgb)
        plt.axis('off')
        plt.title('Barcode Detection Results')
        plt.show()


if __name__ == "__main__":
    # Example usage
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Barcode and Serial Number Extraction')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--output', type=str, help='Path to save visualization')
    parser.add_argument('--json', type=str, help='Path to save results as JSON')
    
    args = parser.parse_args()
    
    try:
        # Process image
        results = process_image_for_barcodes(args.image)
        
        # Print results
        print(f"Found {len(results['barcodes'])} barcodes and {len(results['serial_numbers'])} serial numbers")
        
        # Save results as JSON if requested
        if args.json:
            with open(ar
(Content truncated due to size limit. Use line ranges to read in chunks)