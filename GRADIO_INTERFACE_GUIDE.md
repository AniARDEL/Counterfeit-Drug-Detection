# Gradio Interface for Counterfeit Drug Detection

This document provides instructions for using the new Gradio-based interface for the Counterfeit Drug Detection system.

## Overview

The Gradio interface replaces the previous Flask-based web interface with a simpler, more user-friendly interface that provides the same core functionality:
- Upload medication images for analysis
- Receive analysis results with confidence scores
- View visualizations of the analysis
- Get barcode information when available

## Installation

Before running the Gradio interface, ensure you have the required dependencies installed:

```bash
pip install gradio opencv-python numpy pillow matplotlib
```

## Running the Interface

To start the Gradio interface:

1. Navigate to the project directory:
   ```bash
   cd counterfeit_detection_project
   ```

2. Run the Gradio application:
   ```bash
   python gradio_app.py
   ```

3. The interface will start and provide two URLs:
   - A local URL (e.g., http://127.0.0.1:7860) for accessing on your own computer
   - A public URL (e.g., https://xxxxx.gradio.live) that can be shared with others for temporary access

## Using the Interface

1. **Upload an Image**:
   - Click on the image upload area or drag and drop a medication image
   - The interface accepts common image formats (JPG, PNG, etc.)

2. **Analyze the Image**:
   - Click the "Analyze Image" button
   - The system will process the image and display the results

3. **Review Results**:
   - Analysis Result: Shows whether the medication is classified as authentic or counterfeit
   - Confidence Score: Displays the confidence level of the classification (0-100%)
   - Barcode Information: Shows detected barcode data if available
   - Analysis Visualization: Provides a visual representation of the analysis results

## Advantages of the Gradio Interface

- **Simpler Setup**: No need to run a separate Flask server
- **Easier Sharing**: Automatically generates a shareable link
- **Interactive UI**: More intuitive user experience
- **Responsive Design**: Works well on different devices and screen sizes
- **Faster Development**: Easier to update and maintain

## Technical Details

The Gradio interface uses the same underlying analysis engine as the previous Flask interface:
- Image preprocessing for standardization
- Feature extraction for visual analysis
- Barcode detection and verification
- Confidence scoring and result visualization

## Troubleshooting

If you encounter issues:

1. **Interface doesn't start**:
   - Ensure all dependencies are installed
   - Check for port conflicts with other applications

2. **Image upload fails**:
   - Verify the image format is supported
   - Check that the image file isn't corrupted

3. **Analysis doesn't complete**:
   - Ensure the image contains a clear view of the medication
   - Try with a different image to rule out image-specific issues

## Limitations

This is a demonstration system with the following limitations:
- Uses simulated model predictions for demonstration purposes
- Barcode detection is simulated for the demo
- Not intended for actual verification of medications

Always consult healthcare professionals and use official channels to verify your medications.
