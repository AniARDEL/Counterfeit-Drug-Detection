"""
CNN Model for Counterfeit Drug Detection System

This module implements a Convolutional Neural Network (CNN) model for detecting
counterfeit drugs based on packaging features.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


def create_base_model(input_shape: Tuple[int, int, int] = (224, 224, 3), 
                     base_model_name: str = 'EfficientNetB0') -> tf.keras.Model:
    """
    Create a base model using a pre-trained CNN architecture.
    
    Args:
        input_shape: Input shape as (height, width, channels)
        base_model_name: Name of the pre-trained model to use
        
    Returns:
        Base model with pre-trained weights
    """
    if base_model_name == 'EfficientNetB0':
        base_model = applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'ResNet50':
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'MobileNetV2':
        base_model = applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    # Freeze the base model layers
    base_model.trainable = False
    
    return base_model


def build_feature_extraction_model(base_model: tf.keras.Model, 
                                  num_classes: int = 2) -> tf.keras.Model:
    """
    Build a feature extraction model on top of the base model for binary classification.
    
    Args:
        base_model: Pre-trained base model
        num_classes: Number of output classes (2 for binary classification)
        
    Returns:
        Complete model for training
    """
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def build_siamese_network(base_model: tf.keras.Model) -> tf.keras.Model:
    """
    Build a siamese network for comparing authentic and test images.
    
    Args:
        base_model: Pre-trained base model for feature extraction
        
    Returns:
        Siamese model for similarity comparison
    """
    # Input layers for the two images
    input_a = layers.Input(shape=(224, 224, 3))
    input_b = layers.Input(shape=(224, 224, 3))
    
    # Feature extraction using the same base model
    processed_a = base_model(input_a)
    processed_b = base_model(input_b)
    
    # Global pooling to get feature vectors
    vector_a = layers.GlobalAveragePooling2D()(processed_a)
    vector_b = layers.GlobalAveragePooling2D()(processed_b)
    
    # L1 distance between the two feature vectors
    l1_distance = layers.Lambda(
        lambda tensors: tf.abs(tensors[0] - tensors[1])
    )([vector_a, vector_b])
    
    # Dense layers for similarity prediction
    x = layers.Dense(128, activation='relu')(l1_distance)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    prediction = layers.Dense(1, activation='sigmoid')(x)
    
    # Create the siamese model
    siamese_model = tf.keras.Model(inputs=[input_a, input_b], outputs=prediction)
    
    return siamese_model


def build_ssd_model(input_shape: Tuple[int, int, int] = (300, 300, 3),
                   num_classes: int = 4) -> tf.keras.Model:
    """
    Build a Single Shot MultiBox Detector (SSD) model for object detection.
    
    Args:
        input_shape: Input shape as (height, width, channels)
        num_classes: Number of object classes to detect
        
    Returns:
        SSD model for object detection
    """
    # Base feature extraction network
    base_model = applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Extract feature maps from different layers
    C3 = base_model.get_layer('block_6_expand_relu').output  # 38x38
    C4 = base_model.get_layer('block_13_expand_relu').output  # 19x19
    C5 = base_model.get_layer('out_relu').output  # 10x10
    
    # Additional feature maps
    C6 = layers.Conv2D(256, kernel_size=1, padding='same', activation='relu')(C5)
    C6 = layers.Conv2D(512, kernel_size=3, strides=2, padding='same', activation='relu')(C6)  # 5x5
    
    C7 = layers.Conv2D(128, kernel_size=1, padding='same', activation='relu')(C6)
    C7 = layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu')(C7)  # 3x3
    
    # Prediction heads for each feature map
    # For simplicity, we're using a simplified version of SSD
    # In a full implementation, we would have multiple aspect ratios and scales
    
    # Class prediction heads
    cls_3 = layers.Conv2D(num_classes, kernel_size=3, padding='same')(C3)
    cls_3 = layers.Reshape((-1, num_classes))(cls_3)
    
    cls_4 = layers.Conv2D(num_classes, kernel_size=3, padding='same')(C4)
    cls_4 = layers.Reshape((-1, num_classes))(cls_4)
    
    cls_5 = layers.Conv2D(num_classes, kernel_size=3, padding='same')(C5)
    cls_5 = layers.Reshape((-1, num_classes))(cls_5)
    
    cls_6 = layers.Conv2D(num_classes, kernel_size=3, padding='same')(C6)
    cls_6 = layers.Reshape((-1, num_classes))(cls_6)
    
    cls_7 = layers.Conv2D(num_classes, kernel_size=3, padding='same')(C7)
    cls_7 = layers.Reshape((-1, num_classes))(cls_7)
    
    # Box prediction heads
    box_3 = layers.Conv2D(4, kernel_size=3, padding='same')(C3)
    box_3 = layers.Reshape((-1, 4))(box_3)
    
    box_4 = layers.Conv2D(4, kernel_size=3, padding='same')(C4)
    box_4 = layers.Reshape((-1, 4))(box_4)
    
    box_5 = layers.Conv2D(4, kernel_size=3, padding='same')(C5)
    box_5 = layers.Reshape((-1, 4))(box_5)
    
    box_6 = layers.Conv2D(4, kernel_size=3, padding='same')(C6)
    box_6 = layers.Reshape((-1, 4))(box_6)
    
    box_7 = layers.Conv2D(4, kernel_size=3, padding='same')(C7)
    box_7 = layers.Reshape((-1, 4))(box_7)
    
    # Concatenate predictions from different feature maps
    cls_output = layers.Concatenate(axis=1)([cls_3, cls_4, cls_5, cls_6, cls_7])
    box_output = layers.Concatenate(axis=1)([box_3, box_4, box_5, box_6, box_7])
    
    cls_output = layers.Activation('softmax')(cls_output)
    
    # Create the SSD model
    ssd_model = tf.keras.Model(
        inputs=base_model.input, 
        outputs=[cls_output, box_output]
    )
    
    return ssd_model


def compile_model(model: tf.keras.Model, 
                 learning_rate: float = 0.001,
                 model_type: str = 'classification') -> tf.keras.Model:
    """
    Compile the model with appropriate loss function and optimizer.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for the optimizer
        model_type: Type of model ('classification', 'siamese', or 'ssd')
        
    Returns:
        Compiled model
    """
    if model_type == 'classification':
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    elif model_type == 'siamese':
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    elif model_type == 'ssd':
        # For SSD, we use different losses for classification and box regression
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                'cls_output': 'categorical_crossentropy',
                'box_output': 'mse'
            },
            loss_weights={
                'cls_output': 1.0,
                'box_output': 1.0
            },
            metrics={
                'cls_output': 'accuracy'
            }
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def train_model(model: tf.keras.Model,
               train_data: Tuple[np.ndarray, np.ndarray],
               validation_data: Tuple[np.ndarray, np.ndarray],
               batch_size: int = 32,
               epochs: int = 50,
               model_save_path: str = 'best_model.h5') -> Dict[str, Any]:
    """
    Train the model with the provided data.
    
    Args:
        model: Compiled Keras model
        train_data: Tuple of (x_train, y_train)
        validation_data: Tuple of (x_val, y_val)
        batch_size: Batch size for training
        epochs: Number of epochs to train
        model_save_path: Path to save the best model
        
    Returns:
        Training history
    """
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    x_train, y_train = train_data
    x_val, y_val = validation_data
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history.history


def evaluate_model(model: tf.keras.Model,
                  test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained Keras model
        test_data: Tuple of (x_test, y_test)
        
    Returns:
        Dictionary of evaluation metrics
    """
    x_test, y_test = test_data
    
    # Evaluate the model
    results = model.evaluate(x_test, y_test, verbose=1)
    
    # Create a dictionary of metrics
    metrics = {}
    for i, metric_name in enumerate(model.metrics_names):
        metrics[metric_name] = results[i]
    
    return metrics


def predict(model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained Keras model
        image: Preprocessed image as numpy array
        
    Returns:
        Model predictions
    """
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Make prediction
    predictions = model.predict(image)
    
    return predictions


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create a simple test model
    input_shape = (224, 224, 3)
    base_model = create_base_model(input_shape)
    model = build_feature_extraction_model(base_model)
    
    # Print model summary
    model.summary()
    
    # Example of model visualization
    tf.keras.utils.plot_model(
        model,
        to_file='model_architecture.png',
        show_shapes=True,
        show_layer_names=True
    )
    
    print("Model created successfully!")
