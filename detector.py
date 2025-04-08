"""
Main module for the image recognition component of the Counterfeit Drug Detection System.

This module integrates preprocessing, feature extraction, and model components
to provide a complete pipeline for detecting counterfeit drugs based on packaging images.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional
import json
import matplotlib.pyplot as plt

# Import local modules
from preprocessing import preprocess_image, preprocess_batch
from feature_extraction import extract_all_features, compare_features, detect_logos
from model import (
    create_base_model, 
    build_feature_extraction_model,
    build_siamese_network,
    build_ssd_model,
    compile_model,
    train_model,
    evaluate_model,
    predict
)


class CounterfeitDetector:
    """
    Main class for counterfeit drug detection using image recognition.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the counterfeit detector.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Default configuration
        self.config = {
            'input_shape': (224, 224, 3),
            'base_model_name': 'EfficientNetB0',
            'model_type': 'classification',
            'num_classes': 2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'model_save_path': 'models/best_model.h5',
            'feature_extraction_layer': 'top_activation',
            'similarity_threshold': 0.85,
            'confidence_threshold': 0.7
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        
        # Initialize models
        self.base_model = None
        self.classification_model = None
        self.siamese_model = None
        self.ssd_model = None
        self.feature_extractor = None
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config['model_save_path']), exist_ok=True)
    
    def build_models(self):
        """
        Build all required models for counterfeit detection.
        """
        # Create base model
        self.base_model = create_base_model(
            input_shape=self.config['input_shape'],
            base_model_name=self.config['base_model_name']
        )
        
        # Build classification model
        if self.config['model_type'] == 'classification' or self.config['model_type'] == 'all':
            self.classification_model = build_feature_extraction_model(
                self.base_model,
                num_classes=self.config['num_classes']
            )
            self.classification_model = compile_model(
                self.classification_model,
                learning_rate=self.config['learning_rate'],
                model_type='classification'
            )
        
        # Build siamese model
        if self.config['model_type'] == 'siamese' or self.config['model_type'] == 'all':
            self.siamese_model = build_siamese_network(self.base_model)
            self.siamese_model = compile_model(
                self.siamese_model,
                learning_rate=self.config['learning_rate'],
                model_type='siamese'
            )
        
        # Build SSD model
        if self.config['model_type'] == 'ssd' or self.config['model_type'] == 'all':
            self.ssd_model = build_ssd_model(
                input_shape=(300, 300, 3),  # SSD typically uses 300x300
                num_classes=4  # Assuming 4 classes: background, logo, text, hologram
            )
            self.ssd_model = compile_model(
                self.ssd_model,
                learning_rate=self.config['learning_rate'],
                model_type='ssd'
            )
        
        # Create feature extractor
        if self.base_model is not None:
            self.feature_extractor = tf.keras.Model(
                inputs=self.base_model.input,
                outputs=self.base_model.get_layer(self.config['feature_extraction_layer']).output
            )
    
    def load_models(self, model_paths: Dict[str, str]):
        """
        Load pre-trained models from disk.
        
        Args:
            model_paths: Dictionary mapping model types to file paths
        """
        if 'classification' in model_paths and os.path.exists(model_paths['classification']):
            self.classification_model = tf.keras.models.load_model(model_paths['classification'])
            print(f"Loaded classification model from {model_paths['classification']}")
        
        if 'siamese' in model_paths and os.path.exists(model_paths['siamese']):
            self.siamese_model = tf.keras.models.load_model(model_paths['siamese'])
            print(f"Loaded siamese model from {model_paths['siamese']}")
        
        if 'ssd' in model_paths and os.path.exists(model_paths['ssd']):
            self.ssd_model = tf.keras.models.load_model(model_paths['ssd'])
            print(f"Loaded SSD model from {model_paths['ssd']}")
        
        # Extract base model from one of the loaded models
        if self.classification_model is not None:
            # Assuming the base model is the first part of the classification model
            self.base_model = self.classification_model.layers[0]
            
            # Create feature extractor
            self.feature_extractor = tf.keras.Model(
                inputs=self.base_model.input,
                outputs=self.base_model.get_layer(self.config['feature_extraction_layer']).output
            )
    
    def train(self, 
             train_data: Tuple[np.ndarray, np.ndarray],
             validation_data: Tuple[np.ndarray, np.ndarray]):
        """
        Train the models with the provided data.
        
        Args:
            train_data: Tuple of (x_train, y_train)
            validation_data: Tuple of (x_val, y_val)
        
        Returns:
            Training history
        """
        if self.classification_model is None:
            raise ValueError("Classification model not initialized. Call build_models() first.")
        
        history = train_model(
            self.classification_model,
            train_data,
            validation_data,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            model_save_path=self.config['model_save_path']
        )
        
        return history
    
    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray]):
        """
        Evaluate the models on test data.
        
        Args:
            test_data: Tuple of (x_test, y_test)
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.classification_model is None:
            raise ValueError("Classification model not initialized. Call build_models() first.")
        
        metrics = evaluate_model(self.classification_model, test_data)
        
        return metrics
    
    def detect_counterfeit(self, image_path: str, reference_image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect if a drug package is counterfeit.
        
        Args:
            image_path: Path to the image to check
            reference_image_path: Path to a reference authentic image (optional)
        
        Returns:
            Dictionary with detection results
        """
        # Preprocess the image
        image = preprocess_image(image_path)
        
        results = {}
        
        # If we have a classification model, use it
        if self.classification_model is not None:
            # Ensure image has batch dimension
            if len(image.shape) == 3:
                image_batch = np.expand_dims(image, axis=0)
            
            # Get classification prediction
            class_predictions = self.classification_model.predict(image_batch)
            counterfeit_probability = class_predictions[0][1]  # Assuming class 1 is counterfeit
            
            results['classification'] = {
                'counterfeit_probability': float(counterfeit_probability),
                'is_counterfeit': counterfeit_probability > self.config['confidence_threshold']
            }
        
        # If we have a reference image and siamese model, compare them
        if reference_image_path and self.siamese_model is not None:
            # Preprocess reference image
            reference_image = preprocess_image(reference_image_path)
            
            # Ensure images have batch dimension
            if len(image.shape) == 3:
                image_batch = np.expand_dims(image, axis=0)
            if len(reference_image.shape) == 3:
                reference_batch = np.expand_dims(reference_image, axis=0)
            
            # Get similarity prediction
            similarity = self.siamese_model.predict([image_batch, reference_batch])[0][0]
            
            results['similarity'] = {
                'similarity_score': float(similarity),
                'is_counterfeit': similarity < self.config['similarity_threshold']
            }
        
        # If we have a feature extractor, extract and compare features
        if reference_image_path and self.feature_extractor is not None:
            # Load original images (not preprocessed) for feature extraction
            original_image = cv2.imread(image_path)
            original_reference = cv2.imread(reference_image_path)
            
            # Extract features
            image_features = extract_all_features(original_image, self.feature_extractor)
            reference_features = extract_all_features(original_reference, self.feature_extractor)
            
            # Compare features
            feature_similarities = {}
            for feature_type in image_features.keys():
                if feature_type in reference_features:
                    if feature_type == 'sift':
                        # For SIFT, we need special handling (not implemented here)
                        continue
                    
                    similarity = compare_features(
                        image_features[feature_type],
                        reference_features[feature_type]
                    )
                    feature_similarities[feature_type] = float(similarity)
            
            # Calculate average similarity
            if feature_similarities:
                avg_similarity = sum(feature_similarities.values()) / len(feature_similarities)
                
                results['feature_comparison'] = {
                    'feature_similarities': feature_similarities,
                    'average_similarity': avg_similarity,
                    'is_counterfeit': avg_similarity < self.config['similarity_threshold']
                }
        
        # If we have an SSD model, detect packaging elements
        if self.ssd_model is not None:
            # Load original image for object detection
            original_image = cv2.imread(image_path)
            
            # Resize for SSD input
            ssd_input = cv2.resize(original_image, (300, 300))
            ssd_input = ssd_input.astype(np.float32) / 255.0
            ssd_input = np.expand_dims(ssd_input, axis=0)
            
            # Get detections
            class_scores, boxes = self.ssd_model.predict(ssd_input)
            
            # Process detections
            detections = []
            height, width = original_image.shape[:2]
            
            for i, (class_score, box) in enumerate(zip(class_scores[0], boxes[0])):
                class_id = np.argmax(class_score)
                confidence = class_score[class_id]
                
                if confidence >= self.config['confidence_threshold']:
                    # Scale box to original image dimensions
                    x1, y1, x2, y2 = box
                    x1 = int(x1 * width)
                    y1 = int(y1 * height)
                    x2 = int(x2 * width)
                    y2 = int(y2 * height)
                    
                    detections.append({
                        'class_id': int(class_id),
                        'class_name': ['background', 'logo', 'text', 'hologram'][class_id],
                        'confidence': float(confidence),
                        'box': [x1, y1, x2, y2]
                    })
            
            results['object_detection'] = {
                'detections': detections
            }
        
        # Combine results to make final decision
        is_counterfeit = False
        confidence = 0.5
        
        if 'classification' in results:
            is_counterfeit = results['classification']['is_counterfeit']
            confidence = results['classification']['counterfeit_probability']
        
        if 'similarity' in results:
            # If similarity check strongly suggests counterfeit, override classification
            if results['similarity']['is_counterfeit'] and results['similarity']['similarity_score'] < 0.3:
                is_counterfeit = True
                confidence = max(confidence, 1.0 - results['similarity']['similarity_score'])
        
        if 'feature_comparison' in results:
            # If feature comparison strongly suggests counterfeit, consider it
            if results['feature_comparison']['is_counterfeit'] and results['feature_comparison']['average_similarity'] < 0.3:
                is_counterfeit = True
                confidence = max(confidence, 1.0 - results['feature_comparison']['average_similarity'])
        
        results['final_decision'] = {
            'is_counterfeit': is_counterfeit,
            'confidence': float(confidence)
        }
        
        return results
    
    def visualize_results(self, image_path: str, results: Dict[str, Any], output_path: Optional[str] = None):
        """
        Visualize detection results on the image.
        
        Args:
            image_path: Path to the original image
            results: Detection results from detect_counterfeit
            output_path: Path to save the visualization (optional)
        """
        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        
        # Display the image
        plt.imshow(image_rgb)
        
        # Draw object detections if available
        if 'object_detection' in results and 'detections' in results['object_detection']:
            for detection in results['object_detection']['detections']:
                x1, y1, x2, y2 = detection['box']
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Draw bounding box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                    fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                # Add label
                plt.text(x1, y1 - 10, f"{class_name}: {confidence:.2f}", 
                        color='red', fontsize=12, backgroundcolor='white')
        
        # Add final decision as title
        if 'final_decision' in results:
            decision = "COUNTERFEIT" if results['final_decision']['is_counterfeit'] else "AUTHENTIC"
            confidence = results['final_decision']['confidence']
            plt.title(f"Detection Result: {decision} (Confidence: {confidence:.2f})", fontsize=16)
        
        # Add additional information
        inf
(Content truncated due to size limit. Use line ranges to read in chunks)