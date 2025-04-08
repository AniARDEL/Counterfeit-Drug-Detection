# Counterfeit Drug Detection System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)

A machine learning-based system for detecting counterfeit drugs using a combination of image recognition and text-based verification techniques.

## Overview

This project implements a comprehensive solution for identifying counterfeit pharmaceutical products by analyzing multiple factors:

- **Drug packaging features** (logos, fonts, color schemes, and holograms)
- **Barcodes and serial numbers** verification against a database of authentic medicines
- **Visual comparison** with reference images of authentic products

The system uses deep learning and artificial intelligence to provide a reliable and efficient verification mechanism that can help protect consumers from potentially harmful counterfeit medications.

## System Architecture

The counterfeit drug detection system is built with a modular architecture consisting of six main components:

1. **Input Module**: Handles image acquisition and preprocessing
2. **Image Recognition Module**: Analyzes visual features using convolutional neural networks
3. **Text Verification Module**: Extracts and verifies barcodes and serial numbers
4. **Database Module**: Stores authentic medicine information for verification
5. **Analysis Engine**: Combines evidence from multiple sources to make a final determination
6. **User Interface**: Provides interaction capabilities and result visualization

```
┌─────────────────┐     ┌─────────────────────────┐
│                 │     │                         │
│  Input Module   │────▶│  Image Recognition      │
│  (Preprocessing)│     │  Module (CNN, SSD)      │
│                 │     │                         │
└─────────────────┘     └─────────────┬───────────┘
        │                             │
        │                             │
        │                             ▼
        │               ┌─────────────────────────┐
        │               │                         │
        └──────────────▶│  Analysis Engine        │◀───┐
                        │  (Decision Fusion)      │    │
                        │                         │    │
                        └─────────────┬───────────┘    │
                                      │                │
                                      │                │
                                      ▼                │
┌─────────────────┐     ┌─────────────────────────┐   │
│                 │     │                         │   │
│  User Interface │◀────│  Result Visualization   │   │
│  (Web-based)    │     │  & Reporting            │   │
│                 │     │                         │   │
└─────────────────┘     └─────────────────────────┘   │
                                                       │
┌─────────────────┐     ┌─────────────────────────┐   │
│                 │     │                         │   │
│  Text           │────▶│  Database Module        │───┘
│  Verification   │     │  (Authentication DB)    │
│  Module         │     │                         │
└─────────────────┘     └─────────────────────────┘
```

## Key Features

### Image Recognition Module
- Preprocessing pipeline for image normalization and enhancement
- Convolutional Neural Network (CNN) models for feature extraction
- Single Shot Detector (SSD) for identifying packaging elements
- Siamese networks for comparing with reference images
- Feature extraction for color, texture, and structural analysis

### Text Verification Module
- Barcode and QR code detection and decoding
- Serial number extraction using OCR
- Database verification of extracted identifiers
- Support for multiple barcode formats (GS1, HIBC, NDC)
- Fuzzy matching for handling minor variations

### Integrated System
- Weighted confidence scoring from multiple verification methods
- Comprehensive result visualization
- Logging and reporting capabilities
- Configuration options for adjusting verification parameters

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenCV
- TensorFlow 2.x
- Flask (for web interface)
- SQLite (for database)

### Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/AniARDEL/Counterfeit-Drug-Detection.git
   cd Counterfeit-Drug-Detection
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Initialize the database with sample data:
   ```
   python src/text_verification/database.py --load_samples
   ```

5. Run the web interface:
   ```
   python src/integrated_system/web_interface.py
   ```

6. Access the web interface at http://localhost:5000

## Usage

### Web Interface

The system provides a web-based interface for easy interaction:

1. **Upload Image**: Upload an image of the drug packaging to be verified
2. **Reference Image** (Optional): Upload an authentic reference image for comparison
3. **View Results**: The system will display the verification result with confidence scores and visualizations

### API Usage

The system also provides an API for integration with other applications:

```python
from integrated_system.detector import detect_from_image

# Detect counterfeit
results = detect_from_image(
    image_path="path/to/image.jpg",
    reference_image_path="path/to/reference.jpg",  # Optional
    config_path="path/to/config.json"  # Optional
)

# Process results
is_counterfeit = results['is_counterfeit']
confidence = results['confidence']
medicine_info = results['medicine_info']

print(f"Detection Result: {'COUNTERFEIT' if is_counterfeit else 'AUTHENTIC'}")
print(f"Confidence: {confidence:.2f}")
```

### Command Line Usage

For batch processing or testing, command-line interfaces are available:

```
# Run integrated detection
python src/integrated_system/detector.py --image path/to/image.jpg --reference path/to/reference.jpg

# Run evaluation
python tests/evaluator.py --test_data tests/data/test_data.json
```

## Project Structure

```
Counterfeit-Drug-Detection/
├── src/
│   ├── image_recognition/
│   │   ├── preprocessing.py
│   │   ├── model.py
│   │   ├── feature_extraction.py
│   │   └── detector.py
│   ├── text_verification/
│   │   ├── barcode_extraction.py
│   │   ├── database.py
│   │   └── verifier.py
│   └── integrated_system/
│       ├── detector.py
│       └── web_interface.py
├── tests/
│   ├── evaluator.py
│   └── data/
│       └── test_data.json
├── documentation.md
├── system_architecture.md
├── research_summary.md
└── requirements.md
```

## Performance Evaluation

The system's performance is evaluated using a test dataset of authentic and counterfeit drug samples. Key metrics include:

- **Accuracy**: Percentage of correctly classified samples
- **Precision**: Ability to avoid false positives (authentic drugs classified as counterfeit)
- **Recall**: Ability to detect all counterfeit drugs
- **F1 Score**: Harmonic mean of precision and recall

The evaluation framework generates detailed reports and visualizations to help understand the system's strengths and weaknesses.

## Future Enhancements

Potential areas for future development include:

1. **Chemical Analysis Integration**: Incorporate spectroscopic data for chemical composition verification
2. **Mobile Application**: Develop a mobile app for field verification
3. **Blockchain Integration**: Use blockchain for secure, tamper-proof verification records
4. **Expanded Database**: Include more pharmaceutical products and manufacturers
5. **Real-time Video Analysis**: Support verification through video streams

## Contributing

Contributions to improve the counterfeit drug detection system are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. World Health Organization. (2018). Substandard and falsified medical products. https://www.who.int/news-room/fact-sheets/detail/substandard-and-falsified-medical-products
2. Kovacs, S., Hawes, S. E., Maley, S. N., Mosites, E., Wong, L., & Stergachis, A. (2014). Technologies for detecting falsified and substandard drugs in low and middle-income countries. PloS one, 9(3), e90601.
3. Mackey, T. K., & Nayyar, G. (2017). A review of existing and emerging digital technologies to combat the global trade in fake medicines. Expert opinion on drug safety, 16(5), 587-602.
