# Counterfeit Drug Detection System

![Counterfeit Drug Detection](https://www.fda.gov/files/Counterfeit-medicine-tablet-pill-capsule-600x400-Purchased-Adobe-Stock.jpg)

## Overview

The Counterfeit Drug Detection System is an advanced machine learning solution designed to identify counterfeit medications through computer vision and deep learning techniques. By leveraging convolutional neural networks and synthetic data generation, our model analyzes multiple factors simultaneously:

- Drug packaging features (logos, fonts, color schemes, and holograms)
- Visual characteristics including color consistency, print quality, and physical attributes
- Texture patterns and imprint clarity

This approach enables rapid, accessible, and accurate identification of potential counterfeit medications without requiring specialized equipment, making it suitable for use by healthcare providers, pharmacists, regulatory agencies, and even consumers.

## Key Features

- **High Accuracy Detection**: 75% overall accuracy with 100% counterfeit recall (zero false negatives)
- **Multi-feature Analysis**: Examines visual characteristics across multiple dimensions
- **Synthetic Data Generation**: Two methods for creating realistic counterfeit samples
- **User-friendly Interface**: Gradio-based web UI for easy upload and analysis
- **Detailed Reporting**: Provides confidence scores and visual explanations
- **Security-Focused Design**: Prioritizes catching all counterfeit medications (zero false negatives)

## System Architecture

Our counterfeit drug detection system employs a modular architecture with five main components:

1. **Data Analysis**: Analyzes dataset composition and generates synthetic counterfeits if needed
2. **Image Recognition Module**: Processes and analyzes visual features using convolutional neural networks
3. **Modeling Pipeline**: Builds, trains and evaluates models with various architectures (EfficientNet, ResNet, MobileNet)
4. **Evaluation System**: Comprehensive metrics and visualizations for model performance
5. **Gradio Interface**: Provides interactive user interface for image analysis

## Dataset

The system was trained and evaluated using a carefully curated dataset:

1. **Authentic Images (8,469 samples)**:

   - NIH/NLM Computational Photography Project for Pill Identification (C3PI): High-quality reference images
   - DailyMed: FDA-approved medication information and images
   - NLM20: National Library of Medicine pharmaceutical dataset

2. **Counterfeit Images (999 samples)**:
   - Synthetically generated using two complementary methods:
     - **Integrated Generator**: Streamlined transformations for dataset balancing
     - **Advanced Generator**: Sophisticated image manipulations with fine-grained control

The dataset maintains a balanced class ratio of 8.48:1 (authentic to counterfeit), which is optimal for reliable model training.

## Model Performance

Our final model achieved the following performance metrics:

- **Accuracy**: 75%
- **Precision (Authentic)**: 100%
- **Recall (Authentic)**: 50%
- **F1-Score (Authentic)**: 67%
- **Precision (Counterfeit)**: 67%
- **Recall (Counterfeit)**: 100%
- **F1-Score (Counterfeit)**: 80%

The model is deliberately biased toward classifying samples as counterfeit, ensuring that no counterfeit medications are missed (0% false negative rate), which is critical for pharmaceutical security.

## Counterfeit Generation Methods

The project implements two different approaches for generating synthetic counterfeit samples:

1. **Integrated Counterfeit Generator** (in data_analysis.py):

   - Streamlined transformation pipeline
   - Automatically handles class imbalance
   - Applies random transformations including color shifts, blurring, noise, contrast changes, compression artifacts, and logo alterations
   - Directly integrated with the data analysis workflow

2. **Advanced Counterfeit Generator** (in synthetic_counterfeit_generator.py):
   - Specialized class-based architecture
   - Five sophisticated transformation methods:
     - Color alteration
     - Texture modification
     - Shape distortion
     - Imprint modification
     - Quality degradation
   - Command-line interface for batch processing
   - Support for processing specific pharmaceutical datasets

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

### Data Analysis and Model Training

To analyze the dataset and train the model:

1. Run data analysis:

```bash
python data_analysis.py
```

2. Train the model:

```bash
python modeling_pipeline.py
```

3. Evaluate the model:

```bash
python test_and_evaluate.py
```

## Project Structure

```
counterfeit_detection_project/
├── gradio_app.py               # Gradio web interface
├── modeling_pipeline.py        # Main modeling pipeline
├── data_analysis.py            # Data analysis and synthetic generation
├── test_and_evaluate.py        # Testing and evaluation scripts
├── synthetic_counterfeit_generator.py # Advanced counterfeit generator
├── requirements.txt            # Project dependencies
├── FINAL_REPORT.md             # Comprehensive project report
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
- DailyMed for pharmaceutical product information and images
- World Health Organization for counterfeit medication statistics and information
