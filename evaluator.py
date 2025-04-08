"""
Testing Framework for Counterfeit Drug Detection System

This module provides functions and classes for testing and evaluating
the performance of the counterfeit drug detection system.
"""

import os
import sys
import json
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import integrated system
from integrated_system.detector import IntegratedDetectionSystem


class SystemEvaluator:
    """
    Class for evaluating the performance of the counterfeit drug detection system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the system evaluator.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Default configuration
        self.config = {
            'test_data_dir': 'tests/data',
            'results_dir': 'tests/results',
            'metrics_file': 'tests/metrics.json',
            'plots_dir': 'tests/plots'
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        
        # Create directories if they don't exist
        for dir_path in [self.config['test_data_dir'], self.config['results_dir'], self.config['plots_dir']]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize detection system
        self.detection_system = IntegratedDetectionSystem()
        
        # Initialize metrics
        self.metrics = {
            'overall': {},
            'image_recognition': {},
            'text_verification': {},
            'by_category': {}
        }
    
    def load_test_data(self, test_data_file: str) -> List[Dict[str, Any]]:
        """
        Load test data from a JSON file.
        
        Args:
            test_data_file: Path to the test data file
            
        Returns:
            List of test cases
        """
        with open(test_data_file, 'r') as f:
            test_data = json.load(f)
        
        return test_data
    
    def run_tests(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run tests on the detection system.
        
        Args:
            test_data: List of test cases
            
        Returns:
            Dictionary with test results
        """
        results = []
        
        # Process each test case
        for i, test_case in enumerate(test_data):
            print(f"Processing test case {i+1}/{len(test_data)}: {test_case['id']}")
            
            # Get image path
            image_path = os.path.join(self.config['test_data_dir'], test_case['image'])
            
            # Get reference image path if available
            reference_path = None
            if 'reference_image' in test_case and test_case['reference_image']:
                reference_path = os.path.join(self.config['test_data_dir'], test_case['reference_image'])
            
            # Process the image
            try:
                start_time = time.time()
                detection_result = self.detection_system.detect_counterfeit(image_path, reference_path)
                processing_time = time.time() - start_time
                
                # Add test case info to result
                result = {
                    'test_case_id': test_case['id'],
                    'image': test_case['image'],
                    'ground_truth': test_case['is_counterfeit'],
                    'predicted': detection_result['is_counterfeit'],
                    'confidence': detection_result['confidence'],
                    'image_confidence': detection_result['image_recognition_results'].get('final_decision', {}).get('confidence', 0.0),
                    'text_confidence': detection_result['text_verification_results'].get('confidence', 0.0),
                    'processing_time': processing_time,
                    'category': test_case.get('category', 'unknown')
                }
                
                # Save result visualization
                result_image_path = os.path.join(
                    self.detection_system.config['output_dir'],
                    f"result_{os.path.basename(image_path)}"
                )
                
                if os.path.exists(result_image_path):
                    result_save_path = os.path.join(
                        self.config['results_dir'],
                        f"{test_case['id']}_result.jpg"
                    )
                    # Copy result image
                    img = cv2.imread(result_image_path)
                    cv2.imwrite(result_save_path, img)
                    result['result_image'] = result_save_path
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing test case {test_case['id']}: {e}")
                results.append({
                    'test_case_id': test_case['id'],
                    'image': test_case['image'],
                    'ground_truth': test_case['is_counterfeit'],
                    'error': str(e)
                })
        
        # Save results to file
        results_file = os.path.join(self.config['results_dir'], 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance metrics from test results.
        
        Args:
            results: List of test results
            
        Returns:
            Dictionary with performance metrics
        """
        # Filter out results with errors
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("No valid results to calculate metrics")
            return self.metrics
        
        # Extract ground truth and predictions
        y_true = [r['ground_truth'] for r in valid_results]
        y_pred = [r['predicted'] for r in valid_results]
        
        # Calculate overall metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        self.metrics['overall'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'confusion_matrix': cm.tolist()
        }
        
        # Calculate metrics for image recognition
        image_y_pred = []
        for r in valid_results:
            image_conf = r['image_confidence']
            image_y_pred.append(image_conf >= 0.7)  # Using 0.7 as threshold
        
        image_cm = confusion_matrix(y_true, image_y_pred)
        image_tn, image_fp, image_fn, image_tp = image_cm.ravel()
        
        image_accuracy = (image_tp + image_tn) / (image_tp + image_tn + image_fp + image_fn)
        image_precision = image_tp / (image_tp + image_fp) if (image_tp + image_fp) > 0 else 0
        image_recall = image_tp / (image_tp + image_fn) if (image_tp + image_fn) > 0 else 0
        image_f1_score = 2 * image_precision * image_recall / (image_precision + image_recall) if (image_precision + image_recall) > 0 else 0
        
        self.metrics['image_recognition'] = {
            'accuracy': image_accuracy,
            'precision': image_precision,
            'recall': image_recall,
            'f1_score': image_f1_score,
            'confusion_matrix': image_cm.tolist()
        }
        
        # Calculate metrics for text verification
        text_y_pred = []
        for r in valid_results:
            text_conf = r['text_confidence']
            text_y_pred.append(text_conf >= 0.7)  # Using 0.7 as threshold
        
        text_cm = confusion_matrix(y_true, text_y_pred)
        text_tn, text_fp, text_fn, text_tp = text_cm.ravel()
        
        text_accuracy = (text_tp + text_tn) / (text_tp + text_tn + text_fp + text_fn)
        text_precision = text_tp / (text_tp + text_fp) if (text_tp + text_fp) > 0 else 0
        text_recall = text_tp / (text_tp + text_fn) if (text_tp + text_fn) > 0 else 0
        text_f1_score = 2 * text_precision * text_recall / (text_precision + text_recall) if (text_precision + text_recall) > 0 else 0
        
        self.metrics['text_verification'] = {
            'accuracy': text_accuracy,
            'precision': text_precision,
            'recall': text_recall,
            'f1_score': text_f1_score,
            'confusion_matrix': text_cm.tolist()
        }
        
        # Calculate metrics by category
        categories = set(r['category'] for r in valid_results if 'category' in r)
        
        for category in categories:
            category_results = [r for r in valid_results if r.get('category') == category]
            
            if not category_results:
                continue
            
            category_y_true = [r['ground_truth'] for r in category_results]
            category_y_pred = [r['predicted'] for r in category_results]
            
            category_cm = confusion_matrix(category_y_true, category_y_pred)
            category_tn, category_fp, category_fn, category_tp = category_cm.ravel()
            
            category_accuracy = (category_tp + category_tn) / (category_tp + category_tn + category_fp + category_fn)
            category_precision = category_tp / (category_tp + category_fp) if (category_tp + category_fp) > 0 else 0
            category_recall = category_tp / (category_tp + category_fn) if (category_tp + category_fn) > 0 else 0
            category_f1_score = 2 * category_precision * category_recall / (category_precision + category_recall) if (category_precision + category_recall) > 0 else 0
            
            self.metrics['by_category'][category] = {
                'accuracy': category_accuracy,
                'precision': category_precision,
                'recall': category_recall,
                'f1_score': category_f1_score,
                'confusion_matrix': category_cm.tolist(),
                'sample_count': len(category_results)
            }
        
        # Calculate average processing time
        avg_time = sum(r['processing_time'] for r in valid_results) / len(valid_results)
        self.metrics['overall']['avg_processing_time'] = avg_time
        
        # Save metrics to file
        with open(self.config['metrics_file'], 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        return self.metrics
    
    def generate_plots(self, results: List[Dict[str, Any]]):
        """
        Generate performance plots from test results.
        
        Args:
            results: List of test results
        """
        # Filter out results with errors
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("No valid results to generate plots")
            return
        
        # Extract data
        y_true = np.array([r['ground_truth'] for r in valid_results])
        confidences = np.array([r['confidence'] for r in valid_results])
        image_confidences = np.array([r['image_confidence'] for r in valid_results])
        text_confidences = np.array([r['text_confidence'] for r in valid_results])
        
        # Generate ROC curve
        self._generate_roc_curve(y_true, confidences, image_confidences, text_confidences)
        
        # Generate precision-recall curve
        self._generate_pr_curve(y_true, confidences, image_confidences, text_confidences)
        
        # Generate confusion matrix visualization
        self._generate_confusion_matrix(self.metrics['overall']['confusion_matrix'])
        
        # Generate component comparison
        self._generate_component_comparison()
        
        # Generate category performance comparison
        self._generate_category_comparison()
    
    def _generate_roc_curve(self, y_true, confidences, image_confidences, text_confidences):
        """
        Generate ROC curve plot.
        
        Args:
            y_true: Ground truth labels
            confidences: Overall confidence scores
            image_confidences: Image recognition confidence scores
            text_confidences: Text verification confidence scores
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate ROC curve and AUC for overall system
        fpr, tpr, _ = roc_curve(y_true, confidences)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Overall (AUC = {roc_auc:.3f})', linewidth=2)
        
        # Calculate ROC curve and AUC for image recognition
        fpr_img, tpr_img, _ = roc_curve(y_true, image_confidences)
        roc_auc_img = auc(fpr_img, tpr_img)
        plt.plot(fpr_img, tpr_img, label=f'Image Recognition (AUC = {roc_auc_img:.3f})', linewidth=2, linestyle='--')
        
        # Calculate ROC curve and AUC for text verification
        fpr_txt, tpr_txt, _ = roc_curve(y_true, text_confidences)
        roc_auc_txt = auc(fpr_txt, tpr_txt)
        plt.plot(fpr_txt, tpr_txt, label=f'Text Verification (AUC = {roc_auc_txt:.3f})', linewidth=2, linestyle=':')
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(self.config['plots_dir'], 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store AUC values in metrics
        self.metrics['overall']['auc'] = roc_auc
        self.metrics['image_recognition']['auc'] = roc_auc_img
        self.metrics['text_verification']['auc'] = roc_auc_txt
    
    def _generate_pr_curve(self, y_true, confidences, image_confidences, text_confidences):
        """
        Generate precision-recall curve plot.
        
        Args:
            y_true: Ground truth labels
            confidences: Overall confidence scores
            image_confidences: Image recognition confidence scores
            text_confidences: Text verification confidence scores
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate precision-recall curve for overall system
        precision, recall, _ = precision_recall_curve(y_true, confidences)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'Overall (AUC = {pr_auc:.3f})', linewidth=2)
        
        # Calculate precision-recall curve for image recognition
        precision_img, recall_img, _ = precision_recall_curve(y_true, image_confidences)
        pr_auc_img = auc(recall_img, precision_img)
        plt.plot(recall_img, precision_img, label=f'Image Recognition (AUC = {pr_auc_img:.3f})', linewidth=2, linestyle='--')
        
        # Calculate precision-recall curve for text verification
        precision_txt, recall_txt, _ = precision_recall_curve(y_tr
(Content truncated due to size limit. Use line ranges to read in chunks)