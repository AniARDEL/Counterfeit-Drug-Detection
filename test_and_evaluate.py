import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import json

# Create output directories if they don't exist
os.makedirs('counterfeit_detection_project/output/evaluation', exist_ok=True)

def evaluate_model_performance(y_true, y_pred, y_scores=None, class_names=None):
    """
    Evaluate model performance and generate visualizations
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Prediction probabilities or scores (optional)
        class_names: Names of the classes (optional)
    """
    if class_names is None:
        class_names = ['Authentic', 'Counterfeit']
    
    # Convert to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('counterfeit_detection_project/output/evaluation/confusion_matrix.png')
    plt.close()
    
    # Calculate and print classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save report to JSON
    with open('counterfeit_detection_project/output/evaluation/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # If scores are provided, plot ROC curve and Precision-Recall curve
    if y_scores is not None:
        if len(y_scores.shape) > 1 and y_scores.shape[1] > 1:
            # Multi-class case, use the counterfeit class probability
            y_scores = y_scores[:, 1]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('counterfeit_detection_project/output/evaluation/roc_curve.png')
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig('counterfeit_detection_project/output/evaluation/precision_recall_curve.png')
        plt.close()
    
    # Create a comprehensive performance metrics visualization
    plt.figure(figsize=(15, 10))
    
    # Plot metrics as a bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        report['accuracy'],
        report['weighted avg']['precision'],
        report['weighted avg']['recall'],
        report['weighted avg']['f1-score']
    ]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    plt.bar(metrics, values, color=colors)
    plt.ylim(0, 1.0)
    plt.title('Model Performance Metrics', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('counterfeit_detection_project/output/evaluation/performance_metrics.png')
    plt.close()
    
    return report

# Simulate model evaluation with sample data
if __name__ == "__main__":
    # Simulate test data
    # In a real scenario, this would come from your actual model predictions
    np.random.seed(42)
    
    # Create sample data - 100 samples with 50% authentic and 50% counterfeit
    n_samples = 100
    y_true = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Simulate predictions with some errors
    # This creates a biased model that tends to classify authentic as counterfeit
    # (common in security systems that err on the side of caution)
    error_rate_authentic = 0.5  # 50% of authentic samples misclassified as counterfeit
    error_rate_counterfeit = 0.0  # 0% of counterfeit samples misclassified as authentic
    
    y_pred = y_true.copy()
    
    # Add errors to authentic samples (class 0)
    authentic_indices = np.where(y_true == 0)[0]
    n_errors_authentic = int(len(authentic_indices) * error_rate_authentic)
    error_indices_authentic = np.random.choice(authentic_indices, n_errors_authentic, replace=False)
    y_pred[error_indices_authentic] = 1
    
    # Add errors to counterfeit samples (class 1)
    counterfeit_indices = np.where(y_true == 1)[0]
    n_errors_counterfeit = int(len(counterfeit_indices) * error_rate_counterfeit)
    error_indices_counterfeit = np.random.choice(counterfeit_indices, n_errors_counterfeit, replace=False)
    y_pred[error_indices_counterfeit] = 0
    
    # Generate confidence scores (probabilities)
    # Higher scores for correct predictions, lower scores for incorrect ones
    y_scores = np.zeros(n_samples)
    
    # Correct predictions get higher scores
    correct_indices = np.where(y_true == y_pred)[0]
    y_scores[correct_indices] = 0.7 + 0.3 * np.random.random(len(correct_indices))
    
    # Incorrect predictions get lower scores
    incorrect_indices = np.where(y_true != y_pred)[0]
    y_scores[incorrect_indices] = 0.3 + 0.4 * np.random.random(len(incorrect_indices))
    
    # For class 0 (authentic), we need to invert the score
    authentic_indices = np.where(y_true == 0)[0]
    y_scores[authentic_indices] = 1 - y_scores[authentic_indices]
    
    # Evaluate the model
    evaluate_model_performance(y_true, y_pred, y_scores)
    
    # Create a comprehensive analysis report visualization
    plt.figure(figsize=(20, 12))
    
    # Set up the figure with subplots
    gs = plt.GridSpec(2, 3, figure=plt.gcf())
    
    # Confusion Matrix
    ax1 = plt.subplot(gs[0, 0])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Authentic', 'Counterfeit'], 
                yticklabels=['Authentic', 'Counterfeit'], ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix')
    
    # Performance Metrics
    ax2 = plt.subplot(gs[0, 1:])
    report = classification_report(y_true, y_pred, target_names=['Authentic', 'Counterfeit'], output_dict=True)
    
    metrics = ['Accuracy', 'Precision (A)', 'Recall (A)', 'F1 (A)', 
               'Precision (C)', 'Recall (C)', 'F1 (C)']
    values = [
        report['accuracy'],
        report['Authentic']['precision'],
        report['Authentic']['recall'],
        report['Authentic']['f1-score'],
        report['Counterfeit']['precision'],
        report['Counterfeit']['recall'],
        report['Counterfeit']['f1-score']
    ]
    
    colors = ['#3498db', '#2ecc71', '#2ecc71', '#2ecc71', '#e74c3c', '#e74c3c', '#e74c3c']
    
    ax2.bar(metrics, values, color=colors)
    ax2.set_ylim(0, 1.0)
    ax2.set_title('Performance Metrics')
    ax2.set_ylabel('Score')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        ax2.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    # ROC Curve
    ax3 = plt.subplot(gs[1, 0])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend(loc="lower right")
    
    # Precision-Recall Curve
    ax4 = plt.subplot(gs[1, 1])
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    ax4.plot(recall, precision, color='blue', lw=2, label=f'PR (AUC = {pr_auc:.2f})')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall Curve')
    ax4.legend(loc="lower left")
    
    # Error Analysis
    ax5 = plt.subplot(gs[1, 2])
    error_types = ['False Positives\n(Authentic → Counterfeit)', 'False Negatives\n(Counterfeit → Authentic)']
    error_counts = [
        cm[0, 1],  # False positives
        cm[1, 0]   # False negatives
    ]
    
    ax5.bar(error_types, error_counts, color=['#e74c3c', '#f39c12'])
    ax5.set_title('Error Analysis')
    ax5.set_ylabel('Count')
    ax5.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(error_counts):
        ax5.text(i, v + 0.5, str(v), ha='center')
    
    plt.suptitle('Comprehensive Analysis Report: Counterfeit Drug Detection System', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig('counterfeit_detection_project/output/analysis/comprehensive_analysis_report.png')
    plt.close()
    
    print("Evaluation complete. Results saved to 'counterfeit_detection_project/output/evaluation/'")
