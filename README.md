# Counterfeit Drug Detection System

![Counterfeit Drug Detection](https://www.fda.gov/files/Counterfeit-medicine-tablet-pill-capsule-600x400-Purchased-Adobe-Stock.jpg)

## Overview

The Counterfeit Drug Detection System uses machine learning to identify counterfeit medications through computer vision techniques. Our EfficientNetB0-based Model 2 analyzes multiple factors simultaneously:

- Drug packaging features (logos, fonts, color schemes, and holograms)
- Visual characteristics including color consistency, print quality, and physical attributes
- Texture patterns and imprint clarity
- Suspicious text markings and modifications

This approach enables rapid, accessible, and accurate identification of potential counterfeit medications without requiring specialized equipment, making it suitable for use by healthcare providers, pharmacists, regulatory agencies, and even consumers.

## Key Features

- **EfficientNetB0 Transfer Learning**: Leveraging pre-trained image recognition capabilities
- **High Performance in Real-world Conditions**: 95% overall accuracy with 99.7% authentic recall
- **Fast Processing**: 76ms per image analysis time
- **Advanced Synthetic Generation**: Five specialized counterfeit simulation techniques
- **User-friendly Interface**: Gradio-based web UI with detailed visual explanations
- **Error Handling**: Recovery mechanisms for corrupted images and processing errors
- **Feature-specific Analysis**: Detailed breakdown of suspicious visual elements

## System Architecture

Our implementation employs a modular architecture with five main components:

1. **Data Analysis**: Analyzes dataset composition with recursive image discovery and corrupted file detection
2. **Image Recognition Module**: Processes and analyzes visual features using transfer learning with pre-trained EfficientNetB0
3. **Modeling Pipeline**: Two-phase training (11-epoch initial training followed by 10-epoch fine-tuning) with comprehensive metrics tracking
4. **Evaluation System**: Detailed performance visualization with feature importance analysis
5. **Gradio Interface**: Interactive user interface with confidence visualization and mode transparency

## Dataset

The system was trained and evaluated using a carefully curated dataset:

1. **Authentic Images (8,469 samples)**:

   - NIH/NLM Computational Photography Project for Pill Identification (C3PI): High-quality reference images
   - DailyMed: FDA-approved medication information and images
   - NLM20: National Library of Medicine pharmaceutical dataset

2. **Counterfeit Images (999 samples)**:
   - Synthetically generated using five complementary simulation techniques:
     - **Color Shift**: Modifies hue and saturation to simulate color differences
     - **Noise Addition**: Adds random noise to simulate lower quality printing
     - **Blur Application**: Applies Gaussian blur to simulate poor focus
     - **Contrast/Brightness Adjustment**: Modifies image lighting attributes
     - **Fake Text Overlay**: Adds simulated text markings

The dataset maintains a balanced class ratio of 8.48:1 (authentic to counterfeit), which reflects realistic pharmaceutical market conditions.

## Model Performance

Our EfficientNetB0-based model demonstrates strong performance on a realistic, imbalanced dataset:

| Metric                | Performance |
| --------------------- | ----------- |
| Overall Accuracy      | 95%         |
| Authentic Precision   | 95.1%       |
| Authentic Recall      | 99.7%       |
| Counterfeit Precision | 95.8%       |
| Counterfeit Recall    | 56.5%       |
| F1-Score              | 95%         |
| AUC-ROC               | 0.935       |
| Training Time         | 1.5 hours   |
| Inference Time/Image  | 76ms        |

The high authentic recall (99.7%) is particularly valuable in pharmaceutical contexts, minimizing disruption to legitimate supply chains. While counterfeit recall is lower (56.5%), this represents a reasonable trade-off in real-world conditions where false positives would be highly problematic.

## Technical Highlights

Our implementation provides several technical advantages:

1. **EfficientNetB0-based Transfer Learning**:

   - Pre-trained ImageNet weights providing excellent feature extraction capabilities
   - Two-phase training approach: 11-epoch initial training followed by 10-epoch fine-tuning

2. **Speed Optimizations**:

   - Parallel preprocessing pipeline
   - 40% reduction in memory utilization
   - Recursive image discovery with intelligent filtering
   - Advanced error recovery mechanisms

3. **Robust Data Handling**:

   - Validates images before processing to prevent errors
   - Automatically finds images across nested directory structures
   - Supports multiple image formats (PNG, JPEG, BMP)

4. **Detailed Visualization**:
   - Feature importance analysis showing focus on text/imprint quality (35%)
   - Detailed breakdown of key authentication factors
   - Clear indication of model operation mode
   - Graphical confidence visualization

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
   - Authentication status with confidence percentage
   - Visual explanation with highlighted suspicious areas
   - Detailed breakdown of detection factors
   - Processing time and model information

### Model Training and Evaluation

To train and evaluate the model:

1. Run the comprehensive pipeline:

```bash
python drug_detection_model.py
```

This will:

- Preprocess the dataset with validation
- Build and train the EfficientNetB0 architecture
- Fine-tune the model for optimal performance
- Evaluate and generate performance metrics
- Launch the Gradio interface for interactive testing

## Project Structure

```
counterfeit_detection_project/
├── drug_detection_model.py     # Improved detection model with dual architecture
├── gradio_app.py               # Enhanced Gradio web interface
├── modeling_pipeline.py        # Original modeling pipeline
├── data_analysis.py            # Data analysis and synthetic generation
├── test_and_evaluate.py        # Testing and evaluation scripts
├── synthetic_counterfeit_generator.py # Advanced counterfeit generator
├── requirements.txt            # Project dependencies
├── REPORT.md                   # Comprehensive report on enhanced system
├── GRADIO_INTERFACE_GUIDE.md   # Guide for using the Gradio interface
├── data/                       # Dataset directory
│   ├── authentic/              # Authentic pharmaceutical images
│   └── counterfeit/            # Counterfeit or synthetic images
├── output/                     # Output directory
│   ├── models/                 # Trained models
│   ├── evaluation/             # Evaluation metrics and visualizations
│   ├── analysis/               # Analysis results
│   └── results/                # Analysis results for users
└── .gradio/                    # Gradio cache (not tracked in git)
```

## Future Enhancements

1. **Multi-modal Analysis**: Incorporating spectroscopic data alongside visual analysis
2. **3D Analysis**: Adding support for 3D scanning and volumetric analysis
3. **Region-Specific Models**: Specialized models for different pharmaceutical markets
4. **Mobile Deployment**: Optimized models for smartphone-based detection
5. **Continuous Learning**: Online learning to adapt to new counterfeiting techniques
6. **Drug-Specific Models**: Specialized models for high-risk medication categories

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
- DailyMed for pharmaceutical product information and images
- World Health Organization for counterfeit medication statistics and information
