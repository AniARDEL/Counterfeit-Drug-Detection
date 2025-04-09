#!/usr/bin/env python3
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from tqdm import tqdm
import argparse
import json
import random
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class DataPreprocessor:
    """Class for preprocessing pharmaceutical images for model training."""
    
    def __init__(self, img_size=(224, 224), augmentation_strength=0.5):
        """
        Initialize the data preprocessor.
        
        Args:
            img_size (tuple): Target image size (height, width)
            augmentation_strength (float): Strength of data augmentation (0.0-1.0)
        """
        self.img_size = img_size
        self.augmentation_strength = augmentation_strength
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def create_data_augmentation_model(self):
        """
        Create a data augmentation model for training.
        
        Returns:
            tf.keras.Sequential: Data augmentation model
        """
        # Define augmentation strength based on the parameter
        rotation_range = int(20 * self.augmentation_strength)
        width_shift = height_shift = 0.2 * self.augmentation_strength
        zoom_range = 0.2 * self.augmentation_strength
        
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(rotation_range / 360.0),
            layers.RandomTranslation(height_shift, width_shift),
            layers.RandomZoom(zoom_range),
            layers.RandomBrightness(0.2 * self.augmentation_strength),
            layers.RandomContrast(0.2 * self.augmentation_strength),
        ])
        
        return data_augmentation
    
    def load_dataset(self, authentic_dir, counterfeit_dir, max_samples=None):
        """
        Load and preprocess the dataset from authentic and counterfeit directories.
        
        Args:
            authentic_dir (str): Directory containing authentic pill images
            counterfeit_dir (str): Directory containing counterfeit pill images
            max_samples (int, optional): Maximum number of samples per class
            
        Returns:
            tuple: (images, labels, filenames)
        """
        # Get image file paths
        authentic_files = []
        counterfeit_files = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            authentic_files.extend(glob.glob(os.path.join(authentic_dir, '**', ext), recursive=True))
            counterfeit_files.extend(glob.glob(os.path.join(counterfeit_dir, '**', ext), recursive=True))
        
        # Limit the number of samples if specified
        if max_samples is not None:
            if len(authentic_files) > max_samples:
                authentic_files = random.sample(authentic_files, max_samples)
            if len(counterfeit_files) > max_samples:
                counterfeit_files = random.sample(counterfeit_files, max_samples)
        
        print(f"Loading {len(authentic_files)} authentic images and {len(counterfeit_files)} counterfeit images")
        
        # Combine file paths and create labels
        all_files = authentic_files + counterfeit_files
        labels = [0] * len(authentic_files) + [1] * len(counterfeit_files)  # 0 for authentic, 1 for counterfeit
        
        # Load and preprocess images
        images = []
        valid_indices = []
        valid_files = []
        
        for i, file_path in enumerate(tqdm(all_files, desc="Preprocessing images")):
            img = self.load_and_preprocess_image(file_path)
            if img is not None:
                images.append(img)
                valid_indices.append(i)
                valid_files.append(file_path)
        
        # Filter labels for valid images
        labels = [labels[i] for i in valid_indices]
        
        return np.array(images), np.array(labels), valid_files
    
    def prepare_dataset_for_training(self, authentic_dir, counterfeit_dir, test_size=0.2, val_size=0.1, max_samples=None):
        """
        Prepare the dataset for training, validation, and testing.
        
        Args:
            authentic_dir (str): Directory containing authentic pill images
            counterfeit_dir (str): Directory containing counterfeit pill images
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of training data to use for validation
            max_samples (int, optional): Maximum number of samples per class
            
        Returns:
            tuple: (train_ds, val_ds, test_ds, test_labels, test_files)
        """
        # Load and preprocess the dataset
        images, labels, filenames = self.load_dataset(authentic_dir, counterfeit_dir, max_samples)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
            images, labels, filenames, test_size=test_size, stratify=labels, random_state=42
        )
        
        # Split training set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), stratify=y_train, random_state=42
        )
        
        # Create data augmentation model
        data_augmentation = self.create_data_augmentation_model()
        
        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(buffer_size=len(X_train))
        train_ds = train_ds.batch(32)
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(32)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(32)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        
        print(f"Dataset prepared: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} testing samples")
        
        return train_ds, val_ds, test_ds, y_test, test_files


class ModelBuilder:
    """Class for building and training counterfeit detection models."""
    
    def __init__(self, img_size=(224, 224), model_type="efficientnet"):
        """
        Initialize the model builder.
        
        Args:
            img_size (tuple): Input image size (height, width)
            model_type (str): Type of model to build ("efficientnet", "resnet", "mobilenet", "custom")
        """
        self.img_size = img_size
        self.model_type = model_type
        self.input_shape = (*img_size, 3)  # RGB images
    
    def build_model(self):
        """
        Build a model for counterfeit detection.
        
        Returns:
            tf.keras.Model: Built model
        """
        if self.model_type == "efficientnet":
            return self._build_efficientnet_model()
        elif self.model_type == "resnet":
            return self._build_resnet_model()
        elif self.model_type == "mobilenet":
            return self._build_mobilenet_model()
        elif self.model_type == "custom":
            return self._build_custom_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _build_efficientnet_model(self):
        """
        Build an EfficientNetB0-based model.
        
        Returns:
            tf.keras.Model: EfficientNet-based model
        """
        # Load pre-trained EfficientNetB0 without top layers
        base_model = applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Build the model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        return model
    
    def _build_resnet_model(self):
        """
        Build a ResNet50-based model.
        
        Returns:
            tf.keras.Model: ResNet-based model
        """
        # Load pre-trained ResNet50 without top layers
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Build the model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        return model
    
    def _build_mobilenet_model(self):
        """
        Build a MobileNetV2-based model.
        
        Returns:
            tf.keras.Model: MobileNet-based model
        """
        # Load pre-trained MobileNetV2 without top layers
        base_model = applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Build the model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        return model
    
    def _build_custom_model(self):
        """
        Build a custom CNN model.
        
        Returns:
            tf.keras.Model: Custom CNN model
        """
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        return model
    
    def train_model(self, train_ds, val_ds, epochs=50, fine_tune=True, fine_tune_epochs=20):
        """
        Train the model.
        
        Args:
            train_ds (tf.data.Dataset): Training dataset
            val_ds (tf.data.Dataset): Validation dataset
            epochs (int): Number of training epochs
            fine_tune (bool): Whether to fine-tune the base model
            fine_tune_epochs (int): Number of fine-tuning epochs
            
        Returns:
            tuple: (trained_model, training_history)
        """
        # Build the model
        model = self.build_model()
        
        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5),
            ModelCheckpoint(
                filepath='best_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train the model
        print("Training the model...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tune the model if specified
        if fine_tune and self.model_type in ["efficientnet", "resnet", "mobilenet"]:
            print("Fine-tuning the model...")
            
            # Unfreeze the base model
            base_model = model.layers[0]
            base_model.trainable = True
            
            # Recompile the model with a lower learning rate
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            # Train the model with fine-tuning
            fine_tune_history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=fine_tune_epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine the histories
            for key in fine_tune_history.history:
                history.history[key].extend(fine_tune_history.history[key])
        
        return model, history


class ModelEvaluator:
    """Class for evaluating trained counterfeit detection models."""
    
    def __init__(self, output_dir="output"):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir (str): Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    
    def evaluate_model(self, model, test_ds, test_labels, test_files):
        """
        Evaluate the model on the test dataset.
        
        Args:
            model (tf.keras.Model): Trained model
            test_ds (tf.data.Dataset): Test dataset
            test_labels (numpy.ndarray): True labels for the test dataset
            test_files (list): List of file paths for the test dataset
            
        Returns:
            dict: Evaluation results
        """
        print("Evaluating the model...")
        
        # Get model predictions
        predictions = model.predict(test_ds)
        predictions = predictions.flatten()
        predicted_labels = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = np.mean(predicted_labels == test_labels)
        cm = confusion_matrix(test_labels, predicted_labels)
        report = classification_report(test_labels, predicted_labels, target_names=["Authentic", "Counterfeit"], output_dict=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Authentic", "Counterfeit"], yticklabels=["Authentic", "Counterfeit"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "evaluation", "confusion_matrix.png"), dpi=300)
        plt.close()
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(test_labels, predictions)
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
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "evaluation", "roc_curve.png"), dpi=300)
        plt.close()
        
        # Save evaluation results
        results = {
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "roc_auc": roc_auc
        }
        
        with open(os.path.join(self.output_dir, "evaluation", "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        # Save predictions for manual inspection
        df = pd.DataFrame({
            "file": test_files,
            "true_label": test_labels,
            "predicted_label": predicted_labels,
            "confidence": predictions
        })
        df.to_csv(os.path.join(self.output_dir, "evaluation", "predictions.csv"), index=False)
        
        # Print summary
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(test_labels, predicted_labels, target_names=["Authentic", "Counterfeit"]))
        
        return results


def main():
    """
    Main function to run the modeling pipeline.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the counterfeit detection modeling pipeline")
    parser.add_argument("--authentic_dir", type=str, default="data/authentic",
                       help="Directory containing authentic drug images")
    parser.add_argument("--counterfeit_dir", type=str, default="data/counterfeit",
                       help="Directory containing counterfeit drug images")
    parser.add_argument("--model_type", type=str, default="efficientnet",
                       choices=["efficientnet", "resnet", "mobilenet", "custom"],
                       help="Type of model to train")
    parser.add_argument("--img_size", type=int, default=224,
                       help="Image size for training (square)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--fine_tune", action="store_true",
                       help="Whether to fine-tune the model")
    parser.add_argument("--fine_tune_epochs", type=int, default=20,
                       help="Number of fine-tuning epochs")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Directory to save output files")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples per class")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(img_size=(args.img_size, args.img_size))
    
    # Check if data directories exist
    if not os.path.exists(args.authentic_dir):
        print(f"Error: Authentic directory {args.authentic_dir} does not exist!")
        return
    
    if not os.path.exists(args.counterfeit_dir):
        print(f"Error: Counterfeit directory {args.counterfeit_dir} does not exist!")
        return
    
    # Prepare dataset
    print("Preparing dataset...")
    train_ds, val_ds, test_ds, test_labels, test_files = preprocessor.prepare_dataset_for_training(
        args.authentic_dir, args.counterfeit_dir, max_samples=args.max_samples
    )
    
    # Initialize model builder
    model_builder = ModelBuilder(
        img_size=(args.img_size, args.img_size),
        model_type=args.model_type
    )
    
    # Train the model
    model, history = model_builder.train_model(
        train_ds, 
        val_ds, 
        epochs=args.epochs,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.fine_tune_epochs
    )
    
    # Save the model
    model_path = os.path.join(args.output_dir, "models", "counterfeit_detection_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_history.png"), dpi=300)
    plt.close()
    
    # Evaluate the model
    evaluator = ModelEvaluator(output_dir=args.output_dir)
    evaluator.evaluate_model(model, test_ds, test_labels, test_files)
    
    print("\nModeling pipeline completed successfully!")


if __name__ == "__main__":
    main()
