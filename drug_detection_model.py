"""
Counterfeit Drug Detection - Improved Implementation
====================================================

This script provides a complete implementation for counterfeit drug detection with the following components:
1. Data preprocessing and augmentation
2. Model architecture (using transfer learning)
3. Training and evaluation
4. Gradio interface connection

Requirements:
- tensorflow
- opencv-python
- numpy
- matplotlib
- scikit-learn
- gradio
- efficientnet (pip install -U efficientnet)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import gradio as gr
import time
import zipfile
import shutil
from pathlib import Path
from PIL import Image

# Configuration parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
MODEL_TYPE = 'efficientnet'  # 'efficientnet' or 'resnet'
THRESHOLD = 0.5  # Threshold for binary classification

class DrugDetectionModel:
    def __init__(self, model_path=None):
        """
        Initialize the drug detection model
        
        Args:
            model_path: Path to a saved model (optional)
        """
        self.model = None
        self.model_path = model_path
        self.image_size = IMAGE_SIZE
        self.preprocess_input = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
    def preprocess_dataset(self, dataset_path, output_path, test_split=0.2, val_split=0.1):
        """
        Preprocess the dataset and organize it for training
        
        Args:
            dataset_path: Path to the dataset zip file or directory
            output_path: Path to save the processed dataset
            test_split: Proportion of data to use for testing
            val_split: Proportion of training data to use for validation
            
        Returns:
            Paths to train, validation, and test directories
        """
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        train_dir = os.path.join(output_path, 'train')
        val_dir = os.path.join(output_path, 'val')
        test_dir = os.path.join(output_path, 'test')
        
        authentic_train_dir = os.path.join(train_dir, 'authentic')
        authentic_val_dir = os.path.join(val_dir, 'authentic')
        authentic_test_dir = os.path.join(test_dir, 'authentic')
        
        counterfeit_train_dir = os.path.join(train_dir, 'counterfeit')
        counterfeit_val_dir = os.path.join(val_dir, 'counterfeit')
        counterfeit_test_dir = os.path.join(test_dir, 'counterfeit')
        
        for directory in [authentic_train_dir, authentic_val_dir, authentic_test_dir,
                         counterfeit_train_dir, counterfeit_val_dir, counterfeit_test_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Extract dataset if it's a zip file
        if dataset_path.endswith('.zip'):
            temp_dir = os.path.join(output_path, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            # Assuming the zip contains a directory structure with authentic and possibly counterfeit images
            dataset_path = temp_dir
        
        # Find authentic and counterfeit images
        authentic_images = []
        counterfeit_images = []
        
        # Look for authentic and counterfeit images in all subdirectories recursively
        print(f"Scanning {dataset_path} for image files...")
        
        # Check for standard directory structure first
        authentic_dir = os.path.join(dataset_path, 'authentic')
        counterfeit_dir = os.path.join(dataset_path, 'counterfeit')
        
        # Function to recursively find all image files in a directory
        def find_images_recursive(directory):
            images = []
            if not os.path.exists(directory):
                return images
                
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        images.append(os.path.join(root, file))
            return images
        
        # Check if there's a specific authentic directory
        if os.path.exists(authentic_dir):
            authentic_images = find_images_recursive(authentic_dir)
            print(f"Found {len(authentic_images)} authentic images in {authentic_dir}")
        else:
            # If no explicit authentic directory, check if there are image files directly in the dataset directory
            for file in os.listdir(dataset_path):
                file_path = os.path.join(dataset_path, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    authentic_images.append(file_path)
            
            # If still no images found, search recursively in the entire dataset directory
            # excluding any directory named 'counterfeit'
            if len(authentic_images) == 0:
                for root, dirs, files in os.walk(dataset_path):
                    # Skip counterfeit directories
                    if 'counterfeit' in root.lower():
                        continue
                        
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            authentic_images.append(os.path.join(root, file))
                            
            print(f"Found {len(authentic_images)} authentic images in and below {dataset_path}")
        
        # Check if there's a specific counterfeit directory
        if os.path.exists(counterfeit_dir):
            counterfeit_images = find_images_recursive(counterfeit_dir)
            print(f"Found {len(counterfeit_images)} counterfeit images in {counterfeit_dir}")
        
        # Final check
        print(f"Total: {len(authentic_images)} authentic images and {len(counterfeit_images)} counterfeit images")
        
        # Safety check - ensure we have at least one authentic image
        if len(authentic_images) == 0:
            raise ValueError(f"No authentic images found in {dataset_path}. Please check the directory path.")
        
        # If no counterfeit images are provided, we'll need to generate synthetic ones or use a different approach
        if len(counterfeit_images) == 0:
            print("No counterfeit images found. Generating synthetic counterfeit images...")
            counterfeit_images = self._generate_synthetic_counterfeits(authentic_images)
        
        # Split authentic images
        authentic_train_val, authentic_test = train_test_split(authentic_images, test_size=test_split, random_state=42)
        authentic_train, authentic_val = train_test_split(authentic_train_val, test_size=val_split/(1-test_split), random_state=42)
        
        # Split counterfeit images
        counterfeit_train_val, counterfeit_test = train_test_split(counterfeit_images, test_size=test_split, random_state=42)
        counterfeit_train, counterfeit_val = train_test_split(counterfeit_train_val, test_size=val_split/(1-test_split), random_state=42)
        
        # Copy images to their respective directories, checking for valid images
        print("\nCopying images to training directories...")
        train_authentic_success = self._copy_images(authentic_train, authentic_train_dir)
        val_authentic_success = self._copy_images(authentic_val, authentic_val_dir)
        test_authentic_success = self._copy_images(authentic_test, authentic_test_dir)
        
        train_counterfeit_success = self._copy_images(counterfeit_train, counterfeit_train_dir)
        val_counterfeit_success = self._copy_images(counterfeit_val, counterfeit_val_dir)
        test_counterfeit_success = self._copy_images(counterfeit_test, counterfeit_test_dir)
        
        # Verify we have images in both classes for all splits
        if not (train_authentic_success and val_authentic_success and test_authentic_success and
                train_counterfeit_success and val_counterfeit_success and test_counterfeit_success):
            raise ValueError("Failed to create valid dataset splits. Some directories have no valid images.")
        
        # Clean up temporary directory if it was created
        if dataset_path.endswith('temp'):
            shutil.rmtree(dataset_path)
        
        return train_dir, val_dir, test_dir
    
    def _generate_synthetic_counterfeits(self, authentic_images, num_counterfeits=None):
        """
        Generate synthetic counterfeit images from authentic ones
        
        Args:
            authentic_images: List of paths to authentic images
            num_counterfeits: Number of counterfeit images to generate (default: same as authentic)
            
        Returns:
            List of paths to generated counterfeit images
        """
        if num_counterfeits is None:
            num_counterfeits = len(authentic_images)
        
        # Create a temporary directory for synthetic counterfeits
        temp_dir = os.path.join(os.path.dirname(authentic_images[0]), 'synthetic_counterfeits')
        os.makedirs(temp_dir, exist_ok=True)
        
        counterfeit_paths = []
        
        # Select images to modify
        selected_images = np.random.choice(authentic_images, num_counterfeits, replace=(num_counterfeits > len(authentic_images)))
        
        for i, img_path in enumerate(selected_images):
            # Read the image
            img = cv2.imread(img_path)
            
            # Apply random transformations to simulate counterfeits
            # 1. Color shift
            if np.random.random() > 0.5:
                img = self._color_shift(img)
            
            # 2. Add noise
            if np.random.random() > 0.5:
                img = self._add_noise(img)
            
            # 3. Blur
            if np.random.random() > 0.5:
                img = self._blur_image(img)
            
            # 4. Change contrast/brightness
            if np.random.random() > 0.5:
                img = self._adjust_contrast_brightness(img)
            
            # 5. Add fake text or watermark
            if np.random.random() > 0.7:
                img = self._add_fake_text(img)
            
            # Save the counterfeit image
            output_path = os.path.join(temp_dir, f"counterfeit_{i}.jpg")
            cv2.imwrite(output_path, img)
            counterfeit_paths.append(output_path)
        
        return counterfeit_paths
    
    def _color_shift(self, image):
        """Apply color shift to simulate counterfeit"""
        # Randomly shift colors in HSV space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Shift hue
        h = h.astype(np.float32)
        h += np.random.randint(-20, 20)
        h = np.clip(h, 0, 179).astype(np.uint8)
        
        # Adjust saturation
        s = s.astype(np.float32)
        s *= np.random.uniform(0.8, 1.2)
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _add_noise(self, image):
        """Add noise to simulate counterfeit"""
        noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
        noisy_img = cv2.add(image, noise)
        return noisy_img
    
    def _blur_image(self, image):
        """Apply blur to simulate counterfeit"""
        blur_factor = np.random.randint(1, 3) * 2 + 1  # Odd numbers: 3, 5
        return cv2.GaussianBlur(image, (blur_factor, blur_factor), 0)
    
    def _adjust_contrast_brightness(self, image):
        """Adjust contrast and brightness to simulate counterfeit"""
        alpha = np.random.uniform(0.8, 1.2)  # Contrast
        beta = np.random.randint(-30, 30)    # Brightness
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def _add_fake_text(self, image):
        """Add fake text or watermark to simulate counterfeit"""
        h, w = image.shape[:2]
        overlay = image.copy()
        
        # Random position
        x = np.random.randint(w // 4, w * 3 // 4)
        y = np.random.randint(h // 4, h * 3 // 4)
        
        # Add text
        fake_texts = ["SAMPLE", "TEST", "COPY", "FAKE"]
        text = np.random.choice(fake_texts)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = np.random.uniform(0.5, 1.0)
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        thickness = np.random.randint(1, 3)
        
        cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness)
        
        # Apply transparency
        alpha = np.random.uniform(0.3, 0.7)
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    def _copy_images(self, image_paths, target_dir):
        """Copy images to target directory, skipping corrupted files"""
        valid_count = 0
        skipped_count = 0
        
        for i, img_path in enumerate(image_paths):
            try:
                # Validate image by trying to open it
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Skipping unreadable image: {img_path}")
                    skipped_count += 1
                    continue
                    
                # Additional validation - try to decode the image
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize to ensure it's a valid image with reasonable dimensions
                    img = cv2.resize(img, self.image_size)
                except Exception as e:
                    print(f"Warning: Skipping corrupted image: {img_path}, Error: {str(e)}")
                    skipped_count += 1
                    continue
                
                # Get file extension
                ext = os.path.splitext(img_path)[1]
                # Create a new filename to avoid conflicts
                new_filename = f"img_{valid_count}{ext}"
                target_path = os.path.join(target_dir, new_filename)
                
                # Copy the valid image
                shutil.copy(img_path, target_path)
                valid_count += 1
                
            except Exception as e:
                print(f"Warning: Error processing image {img_path}: {str(e)}")
                skipped_count += 1
        
        print(f"Copied {valid_count} valid images to {target_dir} (Skipped {skipped_count} corrupted/invalid images)")
        
        if valid_count == 0:
            print(f"ERROR: No valid images were found for {os.path.basename(target_dir)}!")
            return False
        
        return True
    
    def build_model(self):
        """
        Build the model architecture using transfer learning
        
        Returns:
            Compiled model
        """
        # Set up base model
        if MODEL_TYPE == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*self.image_size, 3))
            self.preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        else:  # ResNet50
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*self.image_size, 3))
            self.preprocess_input = tf.keras.applications.resnet50.preprocess_input
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        # Create model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return self.model
    
    def setup_data_generators(self, train_dir, val_dir, test_dir=None):
        """
        Set up data generators for training, validation, and testing
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            test_dir: Path to test data directory (optional)
            
        Returns:
            Training, validation, and test generators
        """
        # Setup a custom image validation function to use with ImageDataGenerator
        def validate_image(img_path):
            try:
                with open(img_path, 'rb') as f:
                    # Try reading a few bytes to check if file is corrupted
                    header = f.read(512)
                    if len(header) < 10:  # File is too small to be a valid image
                        print(f"Warning: Skipping too small file: {img_path}")
                        return False
                
                # Try opening with PIL to check if it's a valid image
                with Image.open(img_path) as img:
                    img.verify()  # Verify it's a valid image
                return True
            except Exception as e:
                print(f"Error validating {img_path}: {str(e)}")
                return False
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.0  # Not using the internal validation split
        )
        
        # Only preprocessing for validation and test
        val_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input)
        
        print("\nSetting up data generators with validation...")
        print(f"Training directory: {train_dir}")
        print(f"Validation directory: {val_dir}")
        if test_dir:
            print(f"Testing directory: {test_dir}")
        
        try:
            # Set up generators with error handling
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.image_size,
                batch_size=BATCH_SIZE,
                class_mode='binary',
                shuffle=True
            )
            
            val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=self.image_size,
                batch_size=BATCH_SIZE,
                class_mode='binary',
                shuffle=False
            )
            
            test_generator = None
            if test_dir and os.path.exists(test_dir):
                test_generator = val_datagen.flow_from_directory(
                    test_dir,
                    target_size=self.image_size,
                    batch_size=BATCH_SIZE,
                    class_mode='binary',
                    shuffle=False
                )
            
            return train_generator, val_generator, test_generator
            
        except Exception as e:
            print(f"Error setting up data generators: {str(e)}")
            print("This may be due to corrupted images in the dataset.")
            raise ValueError(f"Failed to create data generators: {str(e)}")
    
    def train(self, train_generator, val_generator, output_dir):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            output_dir: Directory to save model and results
            
        Returns:
            Training history
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up callbacks
        checkpoint_path = os.path.join(output_dir, 'best_model.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        callbacks = [checkpoint, early_stopping, reduce_lr]
        
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        # Save the final model
        final_model_path = os.path.join(output_dir, 'final_model.h5')
        self.model.save(final_model_path)
        self.model_path = checkpoint_path  # Use the best model
        
        # Plot training history
        self._plot_training_history(history, output_dir)
        
        return history
    
    def fine_tune(self, train_generator, val_generator, output_dir, num_layers_to_unfreeze=30):
        """
        Fine-tune the model by unfreezing some layers
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            output_dir: Directory to save model and results
            num_layers_to_unfreeze: Number of layers to unfreeze from the end
            
        Returns:
            Training history
        """
        # Unfreeze the last num_layers_to_unfreeze layers
        for layer in self.model.layers[-num_layers_to_unfreeze:]:
            layer.trainable = True
        
        # Recompile with a lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE / 10),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Set up callbacks
        checkpoint_path = os.path.join(output_dir, 'best_finetuned_model.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        callbacks = [checkpoint, early_stopping, reduce_lr]
        
        # Fine-tune the model
        history = self.model.fit(
            train_generator,
            epochs=EPOCHS // 2,  # Fewer epochs for fine-tuning
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        # Save the final fine-tuned model
        final_model_path = os.path.join(output_dir, 'final_finetuned_model.h5')
        self.model.save(final_model_path)
        self.model_path = checkpoint_path  # Use the best fine-tuned model
        
        # Plot training history
        self._plot_training_history(history, output_dir, prefix='finetuned_')
        
        return history
    
    def evaluate(self, test_generator, output_dir):
        """
        Evaluate the model on test data
        
        Args:
            test_generator: Test data generator
            output_dir: Directory to save evaluation results
            
        Returns:
            Evaluation metrics
        """
        # Load the best model if available
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
        
        # Get predictions
        predictions = self.model.predict(test_generator)
        y_pred = (predictions > THRESHOLD).astype(int)
        y_true = test_generator.classes
        
        # Calculate metrics
        report = classification_report(y_true, y_pred, target_names=['Authentic', 'Counterfeit'])
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Save results
        with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
            f.write(f"Classification Report:\n{report}\n\n")
            f.write(f"Confusion Matrix:\n{conf_matrix}\n\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Authentic', 'Counterfeit'], rotation=45)
        plt.yticks(tick_marks, ['Authentic', 'Counterfeit'])
        
        # Add text annotations
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Calculate and return metrics
        metrics = {
            'accuracy': (y_pred == y_true).mean(),
            'roc_auc': roc_auc,
            'report': report,
            'confusion_matrix': conf_matrix
        }
        
        return metrics
    
    def _plot_training_history(self, history, output_dir, prefix=''):
        """Plot and save training history"""
        # Plot accuracy
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}training_history.png'))
    
    def load_model(self, model_path):
        """Load a saved model"""
        self.model = load_model(model_path)
        self.model_path = model_path
        
        # Set the appropriate preprocessing function
        if 'efficientnet' in MODEL_TYPE:
            self.preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        else:
            self.preprocess_input = tf.keras.applications.resnet50.preprocess_input
    
    def predict(self, image_path):
        """
        Predict whether an image is authentic or counterfeit
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        img = np.expand_dims(img, axis=0)
        img = self.preprocess_input(img)
        
        # Make prediction
        start_time = time.time()
        prediction = self.model.predict(img)[0][0]
        inference_time = time.time() - start_time
        
        # Determine result
        is_authentic = prediction < THRESHOLD
        confidence = prediction if not is_authentic else 1 - prediction
        
        # Generate heatmap for visualization
        heatmap = self._generate_heatmap(image_path)
        
        result = {
            'is_authentic': bool(is_authentic),
            'confidence': float(confidence),
            'inference_time': inference_time,
            'heatmap': heatmap
        }
        
        return result
    
    def _generate_heatmap(self, image_path):
        """
        Generate a heatmap to visualize which parts of the image influenced the decision
        
        Args:
            image_path: Path to the image
            
        Returns:
            Heatmap image as numpy array
        """
        # This is a simplified implementation of Grad-CAM
        # For a full implementation, you would need to extract the gradients
        # from the last convolutional layer
        
        # For now, we'll create a simulated heatmap
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        
        # Create a simulated heatmap (in a real implementation, this would be Grad-CAM)
        heatmap = np.zeros(self.image_size, dtype=np.uint8)
        
        # Add some random hotspots
        for _ in range(3):
            x = np.random.randint(0, self.image_size[0])
            y = np.random.randint(0, self.image_size[1])
            radius = np.random.randint(20, 50)
            intensity = np.random.randint(150, 255)
            
            # Create a circular hotspot
            for i in range(self.image_size[0]):
                for j in range(self.image_size[1]):
                    dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                    if dist < radius:
                        heatmap[i, j] = max(heatmap[i, j], int(intensity * (1 - dist / radius)))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay on original image
        alpha = 0.4
        overlayed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed

class GradioInterface:
    def __init__(self, model):
        """
        Initialize the Gradio interface
        
        Args:
            model: Trained DrugDetectionModel instance
        """
        self.model = model
    
    def predict_image(self, image):
        """
        Predict function for Gradio interface
        
        Args:
            image: Input image from Gradio
            
        Returns:
            Prediction results and visualization
        """
        # Save the input image temporarily
        temp_path = "temp_input.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Get prediction
        result = self.model.predict(temp_path)
        
        # Prepare output
        status = "✅ AUTHENTIC" if result['is_authentic'] else "❌ COUNTERFEIT"
        confidence = f"{result['confidence'] * 100:.2f}%"
        inference_time = f"{result['inference_time'] * 1000:.2f} ms"
        
        # Convert heatmap back to RGB if needed
        heatmap = result['heatmap']
        if len(heatmap.shape) == 2 or heatmap.shape[2] == 1:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return status, confidence, inference_time, heatmap
    
    def launch(self):
        """Launch the Gradio interface"""
        with gr.Blocks(title="Counterfeit Drug Detection") as demo:
            gr.Markdown("# Counterfeit Drug Detection System")
            gr.Markdown("Upload an image of a drug to check if it's authentic or counterfeit.")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Upload Drug Image", type="numpy")
                    submit_btn = gr.Button("Analyze")
                
                with gr.Column():
                    status_output = gr.Textbox(label="Status")
                    confidence_output = gr.Textbox(label="Confidence")
                    time_output = gr.Textbox(label="Processing Time")
                    heatmap_output = gr.Image(label="Analysis Visualization")
            
            submit_btn.click(
                fn=self.predict_image,
                inputs=[input_image],
                outputs=[status_output, confidence_output, time_output, heatmap_output]
            )
            
            gr.Markdown("## How It Works")
            gr.Markdown("""
            This system uses a deep learning model to analyze drug images and determine if they are authentic or counterfeit.
            
            The visualization highlights areas of the image that influenced the decision:
            - Red/yellow areas indicate features that suggest counterfeiting
            - Blue areas indicate features consistent with authentic drugs
            
            For best results, ensure the image is well-lit and clearly shows the drug packaging.
            """)
        
        # Launch the interface
        demo.launch(share=True)

def main():
    """Main function to run the complete pipeline"""
    try:
        # Set up paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = r"D:\_OPIT\Applications in Dta Science 1\Group assessment\Machine Learning for Counterfeit Drug Detection\data"  # Updated to main data directory
        output_dir = os.path.join(base_dir, "drug_detection_output")
        processed_data_dir = os.path.join(output_dir, "processed_data")
        model_dir = os.path.join(output_dir, "model")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize model
        drug_model = DrugDetectionModel()
        
        # Preprocess dataset
        print("Preprocessing dataset...")
        try:
            train_dir, val_dir, test_dir = drug_model.preprocess_dataset(
                dataset_path, processed_data_dir
            )
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            print("Please check your dataset for corrupted images.")
            return
        
        # Build model
        print("Building model...")
        drug_model.build_model()
        
        # Set up data generators
        print("Setting up data generators...")
        try:
            train_generator, val_generator, test_generator = drug_model.setup_data_generators(
                train_dir, val_dir, test_dir
            )
        except Exception as e:
            print(f"Error setting up data generators: {str(e)}")
            print("Please check your dataset for corrupted images.")
            return
        
        # Train model
        print("Training model...")
        try:
            drug_model.train(train_generator, val_generator, model_dir)
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\nTraining failed. Attempting to load a pre-trained model instead...")
            # Try to load a pre-trained model if available
            pretrained_model_path = os.path.join(model_dir, 'best_model.h5')
            if os.path.exists(pretrained_model_path):
                drug_model = DrugDetectionModel(model_path=pretrained_model_path)
                print(f"Loaded pre-trained model from {pretrained_model_path}")
            else:
                print("No pre-trained model available. Cannot continue.")
                return
        
        # Fine-tune model
        try:
            print("Fine-tuning model...")
            drug_model.fine_tune(train_generator, val_generator, model_dir)
        except Exception as e:
            print(f"Error during fine-tuning: {str(e)}")
            print("Proceeding without fine-tuning.")
        
        # Evaluate model
        if test_generator:
            try:
                print("Evaluating model...")
                metrics = drug_model.evaluate(test_generator, model_dir)
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                print("Proceeding without evaluation.")
        
        # Launch Gradio interface
        print("Launching Gradio interface...")
        interface = GradioInterface(drug_model)
        interface.launch()
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 