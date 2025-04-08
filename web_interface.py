"""
User Interface Module for Counterfeit Drug Detection System

This module provides a web-based user interface for the integrated
counterfeit drug detection system.
"""

import os
import sys
import json
import base64
from typing import Dict, List, Any, Optional
from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import integrated system
from integrated_system.detector import IntegratedDetectionSystem


# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize detection system
detection_system = None


def allowed_file(filename):
    """
    Check if the file has an allowed extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_detection_system():
    """
    Initialize the detection system if not already initialized.
    """
    global detection_system
    if detection_system is None:
        detection_system = IntegratedDetectionSystem()


@app.route('/')
def index():
    """
    Render the main page.
    """
    return render_template('index.html')


@app.route('/about')
def about():
    """
    Render the about page.
    """
    return render_template('about.html')


@app.route('/detect', methods=['POST'])
def detect():
    """
    Handle image upload and detection request.
    """
    # Check if detection system is initialized
    init_detection_system()
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get reference image if provided
        reference_path = None
        if 'reference' in request.files:
            ref_file = request.files['reference']
            if ref_file.filename != '' and allowed_file(ref_file.filename):
                ref_filename = secure_filename(ref_file.filename)
                reference_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
                ref_file.save(reference_path)
        
        try:
            # Process the image
            results = detection_system.detect_counterfeit(file_path, reference_path)
            
            # Get the result image path
            result_image_path = os.path.join(
                detection_system.config['output_dir'],
                f"result_{os.path.basename(file_path)}"
            )
            
            # Copy result image to result folder
            result_filename = f"result_{filename}"
            result_save_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            
            if os.path.exists(result_image_path):
                # Read and write the image to ensure it's in the right format
                img = cv2.imread(result_image_path)
                cv2.imwrite(result_save_path, img)
            
            # Prepare response
            response = {
                'success': True,
                'is_counterfeit': results['is_counterfeit'],
                'confidence': results['confidence'],
                'result_image': f"/results/{result_filename}"
            }
            
            # Add medicine info if available
            if results['medicine_info']:
                medicine = results['medicine_info']
                response['medicine_info'] = {
                    'name': medicine.get('name', 'Unknown'),
                    'manufacturer': medicine.get('manufacturer', 'Unknown'),
                    'ndc': medicine.get('ndc', 'Unknown'),
                    'description': medicine.get('description', 'Unknown')
                }
            
            # Add component confidences
            response['image_confidence'] = results['image_recognition_results'].get('final_decision', {}).get('confidence', 0.0)
            response['text_confidence'] = results['text_verification_results'].get('confidence', 0.0)
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    """
    Serve result files.
    """
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """
    API endpoint for detection.
    """
    # Check if detection system is initialized
    init_detection_system()
    
    # Get JSON data
    data = request.json
    
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        
        # Save image to file
        filename = f"api_upload_{int(time.time())}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        # Get reference image if provided
        reference_path = None
        if 'reference' in data:
            ref_data = base64.b64decode(data['reference'])
            ref_filename = f"api_reference_{int(time.time())}.jpg"
            reference_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
            
            with open(reference_path, 'wb') as f:
                f.write(ref_data)
        
        # Process the image
        results = detection_system.detect_counterfeit(file_path, reference_path)
        
        # Get the result image path
        result_image_path = os.path.join(
            detection_system.config['output_dir'],
            f"result_{os.path.basename(file_path)}"
        )
        
        # Read result image and encode to base64
        with open(result_image_path, 'rb') as f:
            result_image = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare response
        response = {
            'success': True,
            'is_counterfeit': results['is_counterfeit'],
            'confidence': results['confidence'],
            'result_image': result_image
        }
        
        # Add medicine info if available
        if results['medicine_info']:
            medicine = results['medicine_info']
            response['medicine_info'] = {
                'name': medicine.get('name', 'Unknown'),
                'manufacturer': medicine.get('manufacturer', 'Unknown'),
                'ndc': medicine.get('ndc', 'Unknown'),
                'description': medicine.get('description', 'Unknown')
            }
        
        # Add component confidences
        response['image_confidence'] = results['image_recognition_results'].get('final_decision', {}).get('confidence', 0.0)
        response['text_confidence'] = results['text_verification_results'].get('confidence', 0.0)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def create_html_templates():
    """
    Create HTML templates for the web interface.
    """
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Counterfeit Drug Detection System</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            padding-top: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .result-container {
            margin-top: 30px;
            display: none;
        }
        .result-image {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .loader {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
        .authentic {
            color: green;
            font-weight: bold;
        }
        .counterfeit {
            color: red;
            font-weight: bold;
        }
        .confidence-bar {
            height: 20px;
            margin: 10px 0;
        }
        .medicine-info {
            margin-top: 20px;
            padding: 15px;
            background-color: #e9ecef;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Counterfeit Drug Detection System</h1>
            <p class="lead">Upload an image of a drug package to verify its authenticity</p>
        </div>
        
        <div class="upload-container">
            <form id="upload-form" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file">Upload Drug Package Image:</label>
                    <input type="file" class="form-control-file" id="file" name="file" accept=".jpg,.jpeg,.png" required>
                    <small class="form-text text-muted">Supported formats: JPG, JPEG, PNG (Max size: 16MB)</small>
                </div>
                
                <div class="form-group">
                    <label for="reference">Reference Image (Optional):</label>
                    <input type="file" class="form-control-file" id="reference" name="reference" accept=".jpg,.jpeg,.png">
                    <small class="form-text text-muted">Upload an authentic reference image for comparison (if available)</small>
                </div>
                
                <button type="submit" class="btn btn-primary">Detect Counterfeit</button>
            </form>
            
            <div class="loader">
                <div class="spinner-border text-primary" role="status">
                    <span class="sr-only">Loading...</span>
                </div>
                <p>Processing image, please wait...</p>
            </div>
            
            <div class="result-container">
                <h3>Detection Result:</h3>
                <div id="result-status" class="alert" role="alert"></div>
                
                <div class="row">
                    <div class="col-md-6">
                        <h5>Overall Confidence:</h5>
                        <div class="progress confidence-bar">
                            <div id="overall-confidence" class="progress-bar" role="progressbar" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h5>Component Confidences:</h5>
                        <div>
                            <label>Image Recognition:</label>
                            <div class="progress confidence-bar">
                                <div id="image-confidence" class="progress-bar bg-info" role="progressbar" style="width: 0%"></div>
                            </div>
                        </div>
                        <div>
                            <label>Text Verification:</label>
                            <div class="progress confidence-bar">
                                <div id="text-confidence" class="progress-bar bg-success" role="progressbar" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div id="medicine-info" class="medicine-info" style="display: none;">
                    <h5>Medicine Information:</h5>
                    <table class="table table-sm">
                        <tr>
                            <th>Name:</th>
                            <td id="medicine-name"></td>
                        </tr>
                        <tr>
                            <th>Manufacturer:</th>
                            <td id="medicine-manufacturer"></td>
                        </tr>
                        <tr>
                            <th>NDC:</th>
                            <td id="medicine-ndc"></td>
                        </tr>
                        <tr>
                            <th>Description:</th>
                            <td id="medicine-description"></td>
                        </tr>
                    </table>
                </div>
                
                <div class="mt-4">
                    <h5>Result Visualization:</h5>
                    <img id="result-image" class="result-image" src="" alt="Result Visualization">
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#upload-form').on('submit', function(e) {
                e.preventDefault();
                
                // Show loader
                $('.loader').show();
                $('.result-container').hide();
                
                // Create form data
                var formData = new FormData(this);
                
                // Send AJAX request
                $.ajax({
                    url: '/detect',
                    type: 'POST',
                    data: formData,
                    contentType: false,
                    processData: false,
                    success: function(response) {
                        // Hide loader
                        $('.loader').hide();
                        
                        // Update result status
                        var resultStatus = $('#result-status');
                        if (response.is_counterfeit) {
                            resultStatus.removeClass('alert-success').addClass('alert-danger');
                            resultStatus.html('<span class="counterfeit">COUNTERFEIT</span> - This medicine appears to be counterfeit.');
                        } else {
                            resultStatus.removeClass('alert-danger').addClass('alert-success');
                            resultStatus.html('<span class="authentic">AUTHENTIC</span> - This medicine appears to be authentic.'
(Content truncated due to size limit. Use line ranges to read in chunks)