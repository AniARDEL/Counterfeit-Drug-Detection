# Counterfeit Drug Detection System

![Counterfeit Drug Detection](https://www.fda.gov/files/Counterfeit-medicine-tablet-pill-capsule-600x400-Purchased-Adobe-Stock.jpg)

## Overview

The Counterfeit Drug Detection System is an advanced machine learning solution designed to identify counterfeit medications through a combination of image recognition and text-based verification techniques. By leveraging deep learning and artificial intelligence, our model analyzes multiple factors simultaneously:

- Drug packaging features (logos, fonts, color schemes, and holograms)
- Barcodes and serial numbers for verification against a database of authentic medicines
- Visual characteristics including color consistency, print quality, and physical attributes

This approach enables rapid, accessible, and accurate identification of potential counterfeit medications without requiring specialized equipment, making it suitable for use by healthcare providers, pharmacists, regulatory agencies, and even consumers.

## Key Features

- **Multi-modal Analysis**: Combines visual inspection and text verification for comprehensive assessment
- **High Security Focus**: Prioritizes catching all counterfeit medications (zero false negatives)
- **User-friendly Interface**: Web-based UI for easy upload and analysis
- **Detailed Reporting**: Provides confidence scores and visual explanations
- **Robust Architecture**: Modular design for easy maintenance and extension

## System Architecture

Our counterfeit drug detection system employs a modular architecture with six main components:

1. **Input Module**: Handles image acquisition and preprocessing
2. **Image Recognition Module**: Analyzes visual features using convolutional neural networks
3. **Text Verification Module**: Extracts and verifies barcodes and text information
4. **Database Module**: Stores authentic product information for verification
5. **Analysis Engine**: Combines evidence and makes final determination
6. **User Interface**: Provides interaction and visualization capabilities

## Dataset

The system was trained and evaluated using multiple pharmaceutical image datasets:

1. **NIH/NLM Computational Photography Project for Pill Identification (C3PI)**: High-quality reference images of authentic medications
2. **Ultralytics Medical Pills Dataset**: Annotated images of various pharmaceutical pills
3. **TruMedicines Pharmaceutical Tablets Dataset**: Diverse pharmaceutical tablet images
4. **Synthetic Counterfeit Samples**: Generated through controlled modifications to authentic images

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.15+
- OpenCV 4.8+
- Gradio 5.23+
- Additional dependencies in requirements.txt

### Setup

1. Clone the repository:

```bash
git clone https://github.com/AniARDEL/Counterfeit-Drug-Detection.git
cd Counterfeit-Drug-Detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python gradio_app.py
```

4. Access the web interface in your browser at the URL displayed in the terminal (typically http://127.0.0.1:7860)

## Usage

### Web Interface

1. Navigate to the web interface
2. Upload an image of the medication or packaging
3. Click "Analyze Image"
4. Review the results, including:
   - Authentication status
   - Confidence score
   - Visual explanation
   - Detected features

### API Usage

```python
from counterfeit_detection import CounterfeitDetector

# Initialize detector
detector = CounterfeitDetector()

# Analyze image
result = detector.analyze("path/to/image.jpg")

# Print results
print(f"Authentication: {'Authentic' if result['is_authentic'] else 'Counterfeit'}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Features: {result['features']}")
```

## Performance

Our final model achieved the following performance metrics on the test dataset:

- **Accuracy**: 75%
- **Precision (Authentic)**: 100%
- **Recall (Authentic)**: 50%
- **F1-Score (Authentic)**: 67%
- **Precision (Counterfeit)**: 67%
- **Recall (Counterfeit)**: 100%
- **F1-Score (Counterfeit)**: 80%

The model is deliberately biased toward classifying samples as counterfeit, ensuring that no counterfeit medications are missed (0% false negative rate), which is the primary security objective.

![Comprehensive Analysis](https://private-us-east-1.manuscdn.com/sessionFile/1Pzq8XLFcnKYyD52oI66gp/sandbox/Wn7fIe1vZaYld5IvRcpqRt-images_1744112829823_na1fn_L2hvbWUvdWJ1bnR1L2NvdW50ZXJmZWl0X2RldGVjdGlvbl9wcm9qZWN0L291dHB1dC9hbmFseXNpcy9jb21wcmVoZW5zaXZlX2FuYWx5c2lzX3JlcG9ydA.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMVB6cThYTEZjbktZeUQ1Mm9JNjZncC9zYW5kYm94L1duN2ZJZTF2WmFZbGQ1SXZSY3BxUnQtaW1hZ2VzXzE3NDQxMTI4Mjk4MjNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyTnZkVzUwWlhKbVpXbDBYMlJsZEdWamRHbHZibDl3Y205cVpXTjBMMjkxZEhCMWRDOWhibUZzZVhOcGN5OWpiMjF3Y21Wb1pXNXphWFpsWDJGdVlXeDVjMmx6WDNKbGNHOXlkQS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=tj~uZT1m8lH3gBKMqMR8w8SCRaSvLios6LNjyCUWIgGwttZriBhz-pFk3HNnYm2Mm~MqQXcZlrTPmhmv1rVTSnAm~i1FA0H9nbTcy-927Zc8kqTGFV8tt2MNeXSQaYBPB3nACLOY~UivtBv~wfEMbciLjkXIToVEFy0ohX7pNG2NfLe7LvKF7ZnCAJlBX1~mXyvp3B7Sb9asT41wbrGCgnrnNiEu~5j5PX0HeP7Enb4JDCjLeyvbVt1nWPCHz4Ym1ofmbCD9iRXDJrjRYxpKjwXntOAjt~6RZeWQZwM4KRP5KnZqhcStyzUlMWRkcnaD~Tzok2T0r1vYD5KFwRuaXw__)

## Project Structure

```
counterfeit_detection_project/
├── gradio_app.py               # Gradio web interface
├── modeling_pipeline.py        # Main modeling pipeline
├── synthetic_counterfeit_generator.py # Generator for synthetic counterfeit samples
├── test_and_evaluate.py        # Testing and evaluation scripts
├── data_analysis.py            # Data analysis utilities
├── requirements.txt            # Project dependencies
├── FINAL_REPORT.md             # Comprehensive project report
├── GRADIO_INTERFACE_GUIDE.md   # Guide for using the Gradio interface
├── output/                     # Output directory
│   └── results/                # Analysis results
└── .gradio/                    # Gradio cache (not tracked in git)
```

## Future Enhancements

1. **Dataset Expansion**: Incorporating more diverse authentic samples and real counterfeit examples
2. **Advanced Feature Extraction**: Implementing spectroscopic analysis capabilities
3. **Explainability Enhancements**: Developing more detailed visualization of decision factors
4. **Continuous Learning**: Implementing a feedback mechanism to incorporate verified results
5. **Regional Customization**: Developing region-specific models for different markets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NIH National Library of Medicine for the C3PI dataset
- Ultralytics for the Medical Pills dataset
- TruMedicines for the Pharmaceutical Tablets dataset
- World Health Organization for counterfeit medication statistics and information
