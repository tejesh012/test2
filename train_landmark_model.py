#!/usr/bin/env python3
"""
Train TensorFlow Landmark-based Emotion Recognition Model

This script trains a 1D-CNN model on sequences of facial landmarks for emotion classification.
The model uses:
- Input: sequences of (T=20, F=1872) where F = 468 landmarks * 2 (x,y) + deltas
- Architecture: Conv1D layers with GlobalAveragePooling
- Output: 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise)

Usage:
    python train_landmark_model.py --data_dir data/landmarks --epochs 50
    python train_landmark_model.py --synthetic  # Generate synthetic data and train
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.model_selection import train_test_split

# Constants
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTION_LABELS)
SEQUENCE_LENGTH = 20  # T = 20 frames
FEATURE_DIM = 1872    # 468 * 2 (coords) + 468 * 2 (deltas)
LANDMARK_COUNT = 468


def build_model(input_shape=(SEQUENCE_LENGTH, FEATURE_DIM), num_classes=NUM_CLASSES):
    """
    Build the 1D-CNN model for landmark-based emotion recognition.

    Architecture:
    - Conv1D(64, kernel=3, relu)
    - Conv1D(128, kernel=3, relu)
    - GlobalAveragePooling1D
    - Dense(128, relu) + Dropout(0.3)
    - Dense(num_classes, softmax)
    """
    inputs = keras.Input(shape=input_shape, name="landmark_sequence")

    # Convolutional layers
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu', name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D(name="global_pool")(x)

    # Dense layers
    x = layers.Dense(128, activation='relu', name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = keras.Model(inputs, outputs, name="LandmarkEmotionModel")
    return model


def generate_synthetic_data(num_samples=5000, save_dir=None):
    """
    Generate synthetic landmark sequences for testing.

    This creates fake but structured data where each emotion class has
    characteristic patterns in the landmark movements.
    """
    print(f"[Synthetic] Generating {num_samples} synthetic samples...")

    X = []
    y = []

    for i in range(num_samples):
        emotion_idx = i % NUM_CLASSES

        # Base landmarks (normalized to [-1, 1])
        base_landmarks = np.random.randn(LANDMARK_COUNT, 2) * 0.3

        # Add emotion-specific patterns
        if EMOTION_LABELS[emotion_idx] == 'happy':
            # Smile: raise mouth corners
            base_landmarks[61:68, 1] -= 0.2  # Lower lip moves up
            base_landmarks[48:55, 1] -= 0.1  # Upper lip slight
        elif EMOTION_LABELS[emotion_idx] == 'sad':
            # Frown: lower mouth corners
            base_landmarks[61:68, 1] += 0.2
            base_landmarks[17:22, 1] += 0.1  # Eyebrows down
        elif EMOTION_LABELS[emotion_idx] == 'angry':
            # Furrowed brows
            base_landmarks[17:27, 1] += 0.15
            base_landmarks[17:27, 0] += 0.1
        elif EMOTION_LABELS[emotion_idx] == 'surprise':
            # Wide eyes, open mouth
            base_landmarks[36:48, 1] -= 0.15  # Eyes wider
            base_landmarks[61:68, 1] += 0.2  # Mouth open
        elif EMOTION_LABELS[emotion_idx] == 'fear':
            # Similar to surprise but more tense
            base_landmarks[36:48, 1] -= 0.1
            base_landmarks[17:22, 1] -= 0.1  # Raised eyebrows
        elif EMOTION_LABELS[emotion_idx] == 'disgust':
            # Wrinkled nose
            base_landmarks[27:36, 1] += 0.1
            base_landmarks[48:55, 1] -= 0.05

        # Generate sequence with temporal variation
        sequence = []
        prev_coords = None

        for t in range(SEQUENCE_LENGTH):
            # Add temporal noise
            noise = np.random.randn(LANDMARK_COUNT, 2) * 0.02
            frame_landmarks = base_landmarks + noise

            # Flatten coordinates
            coords = frame_landmarks.flatten()  # 468 * 2 = 936

            # Compute deltas
            if prev_coords is not None:
                deltas = coords - prev_coords
            else:
                deltas = np.zeros_like(coords)

            prev_coords = coords.copy()

            # Combine coords + deltas
            features = np.concatenate([coords, deltas])  # 936 + 936 = 1872
            sequence.append(features)

        X.append(sequence)
        y.append(emotion_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"[Synthetic] Generated X shape: {X.shape}, y shape: {y.shape}")

    # Save if directory provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_dir / "synthetic_landmarks.npz",
            X=X, y=y, labels=EMOTION_LABELS
        )
        print(f"[Synthetic] Saved to {save_dir / 'synthetic_landmarks.npz'}")

    return X, y


def load_data(data_dir):
    """
    Load landmark sequences from directory.

    Expected format:
    - data_dir/train.npz with keys 'X' (N, T, F) and 'y' (N,)
    - Or individual .npz files per sample
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"[Data] Directory {data_dir} not found!")
        print("[Data] Run with --synthetic flag to generate synthetic data.")
        sys.exit(1)

    # Try loading single file
    train_file = data_dir / "train.npz"
    if train_file.exists():
        data = np.load(train_file)
        X = data['X']
        y = data['y']
        print(f"[Data] Loaded {len(X)} samples from {train_file}")
        return X, y

    # Try loading from synthetic
    synthetic_file = data_dir / "synthetic_landmarks.npz"
    if synthetic_file.exists():
        data = np.load(synthetic_file)
        X = data['X']
        y = data['y']
        print(f"[Data] Loaded {len(X)} samples from {synthetic_file}")
        return X, y

    print(f"[Data] No data files found in {data_dir}")
    print("[Data] Expected: train.npz or synthetic_landmarks.npz")
    sys.exit(1)


def train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, output_dir="models"):
    """Train the model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = build_model()
    model.summary()

    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    cb_list = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            str(output_dir / "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=str(output_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
        )
    ]

    # Train
    print(f"\n[Training] Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1
    )

    # Save final model
    saved_model_dir = output_dir / "saved_model"
    model.export(str(saved_model_dir))
    print(f"\n[Training] Model saved to {saved_model_dir}")

    # Save training history
    with open(output_dir / "history.json", 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    return model, history


def evaluate(model, X_test, y_test):
    """Evaluate model on test set."""
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n[Evaluation] Running on test set...")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Accuracy
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=EMOTION_LABELS))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)

    return acc, y_pred_classes


def main():
    parser = argparse.ArgumentParser(description="Train Landmark Emotion Model")
    parser.add_argument('--data_dir', type=str, default='data/landmarks',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='models/landmark_model',
                        help='Output directory for model and logs')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate and use synthetic data')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Fraction of data for testing')

    args = parser.parse_args()

    print("=" * 60)
    print("Landmark-based Emotion Recognition Training")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print()

    # Load or generate data
    if args.synthetic:
        X, y = generate_synthetic_data(args.num_samples, args.data_dir)
    else:
        X, y = load_data(args.data_dir)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    print(f"\n[Data] Training set: {len(X_train)} samples")
    print(f"[Data] Validation set: {len(X_val)} samples")
    print(f"[Data] Test set: {len(X_test)} samples")

    # Train
    model, history = train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    # Evaluate
    acc, _ = evaluate(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Test Accuracy: {acc:.4f}")
    print(f"Model saved to: {args.output_dir}/saved_model")
    print(f"\nNext step: Run convert_to_tfjs.sh to convert for browser use")


if __name__ == "__main__":
    main()
