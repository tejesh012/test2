#!/usr/bin/env python3
"""
PyTorch Training Pipeline for EfficientNet-B4 + TemporalConv Emotion Recognition

This trains a high-accuracy server-side model using:
- EfficientNet-B4 backbone (pretrained ImageNet)
- TemporalConv1D head for video sequences
- Mixed precision training
- Extensive data augmentation

Usage:
    python train.py --data_dir data/affectnet --epochs 60
    python train.py --synthetic  # Use synthetic data for testing
"""

import argparse
import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image

try:
    import timm
except ImportError:
    print("Please install timm: pip install timm")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Constants
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTION_LABELS)
SEQUENCE_LENGTH = 32  # T = 32 frames for video
IMAGE_SIZE = 224
EMBEDDING_DIM = 512


class TemporalConvBlock(nn.Module):
    """Single Temporal Convolution Block with BatchNorm and Dropout."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class EmotionRecognitionModel(nn.Module):
    """
    EfficientNet-B4 + TemporalConv for video-based emotion recognition.

    Architecture:
    - EfficientNet-B4 backbone (frozen or fine-tuned)
    - Linear projection to EMBEDDING_DIM
    - 3x TemporalConv1D layers
    - Global pooling + Classification head
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()

        # EfficientNet-B4 backbone
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        backbone_dim = self.backbone.num_features  # 1792 for B4

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, EMBEDDING_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Temporal convolution head (3 layers)
        self.temporal_conv = nn.Sequential(
            TemporalConvBlock(EMBEDDING_DIM, 512, kernel_size=3, dropout=0.3),
            TemporalConvBlock(512, 512, kernel_size=3, dropout=0.3),
            TemporalConvBlock(512, 512, kernel_size=3, dropout=0.3),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-frame features using backbone."""
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Reshape to process all frames at once
        x = x.view(B * T, C, H, W)

        # Extract features
        features = self.backbone(x)  # (B*T, backbone_dim)
        features = self.projection(features)  # (B*T, EMBEDDING_DIM)

        # Reshape back
        features = features.view(B, T, -1)  # (B, T, EMBEDDING_DIM)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, C, H, W) for video frames
               or (B, C, H, W) for single images (T=1 assumed)
        """
        # Handle single images
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add time dimension

        # Extract per-frame features
        features = self.extract_features(x)  # (B, T, EMBEDDING_DIM)

        # Transpose for Conv1D: (B, EMBEDDING_DIM, T)
        features = features.transpose(1, 2)

        # Temporal convolutions
        features = self.temporal_conv(features)  # (B, 512, T)

        # Classification
        logits = self.classifier(features)  # (B, num_classes)

        return logits


class VideoEmotionDataset(Dataset):
    """Dataset for video-based emotion recognition."""

    def __init__(self, samples: List[Tuple[List[str], int]], transform=None,
                 sequence_length: int = SEQUENCE_LENGTH):
        """
        Args:
            samples: List of (frame_paths, label) tuples
            transform: Image transforms
            sequence_length: Number of frames per video
        """
        self.samples = samples
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        # Sample or pad frames to sequence_length
        if len(frame_paths) >= self.sequence_length:
            # Uniform sampling
            indices = np.linspace(0, len(frame_paths) - 1, self.sequence_length, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]
        else:
            # Pad by repeating last frame
            padding = [frame_paths[-1]] * (self.sequence_length - len(frame_paths))
            frame_paths = frame_paths + padding

        # Load frames
        frames = []
        for path in frame_paths:
            if isinstance(path, np.ndarray):
                # Synthetic data
                img = Image.fromarray(path.astype(np.uint8))
            else:
                img = Image.open(path).convert('RGB')

            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Stack frames: (T, C, H, W)
        frames = torch.stack(frames)

        return frames, label


def get_transforms(train: bool = True):
    """Get data transforms."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def generate_synthetic_data(num_samples: int = 500) -> List[Tuple[List[np.ndarray], int]]:
    """Generate synthetic face images for testing."""
    print(f"[Synthetic] Generating {num_samples} synthetic video samples...")

    samples = []

    for i in range(num_samples):
        label = i % NUM_CLASSES

        # Generate random face-like images (simple colored squares with patterns)
        frames = []
        base_color = [
            [200, 100, 100],  # angry - reddish
            [100, 150, 100],  # disgust - greenish
            [150, 100, 200],  # fear - purplish
            [100, 200, 100],  # happy - green
            [150, 150, 150],  # neutral - gray
            [100, 100, 200],  # sad - blue
            [200, 200, 100],  # surprise - yellow
        ][label]

        for t in range(SEQUENCE_LENGTH):
            # Create simple face-like image
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

            # Background
            noise = np.random.randint(-20, 20, (IMAGE_SIZE, IMAGE_SIZE, 3))
            img = np.clip(np.array(base_color) + noise, 0, 255).astype(np.uint8)

            # Add "face" oval
            center = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)
            for y in range(IMAGE_SIZE):
                for x in range(IMAGE_SIZE):
                    dy = (y - center[1]) / 80
                    dx = (x - center[0]) / 60
                    if dx*dx + dy*dy < 1:
                        img[y, x] = np.clip(np.array([200, 180, 160]) + np.random.randint(-10, 10, 3), 0, 255)

            # Add temporal variation
            variation = int(10 * np.sin(t * 0.5))
            img = np.clip(img.astype(np.int32) + variation, 0, 255).astype(np.uint8)

            frames.append(img)

        samples.append((frames, label))

    print(f"[Synthetic] Generated {len(samples)} samples")
    return samples


def load_dataset(data_dir: str) -> List[Tuple[List[str], int]]:
    """
    Load dataset from directory.

    Expected structure:
    data_dir/
        angry/
            video1/
                frame001.jpg
                frame002.jpg
                ...
            video2/
                ...
        happy/
            ...
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"[Data] Directory {data_dir} not found!")
        return []

    samples = []

    for label_idx, label in enumerate(EMOTION_LABELS):
        label_dir = data_dir / label
        if not label_dir.exists():
            continue

        for video_dir in label_dir.iterdir():
            if video_dir.is_dir():
                frames = sorted(video_dir.glob("*.jpg")) + sorted(video_dir.glob("*.png"))
                if len(frames) > 0:
                    samples.append(([str(f) for f in frames], label_idx))

    print(f"[Data] Loaded {len(samples)} videos from {data_dir}")
    return samples


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (frames, labels) in enumerate(loader):
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(frames)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return total_loss / len(loader), accuracy, f1, all_preds, all_labels


def export_model(model, output_dir: Path, device):
    """Export model to TorchScript and ONNX."""
    model.eval()

    # Create dummy input: (B=1, T=32, C=3, H=224, W=224)
    dummy_input = torch.randn(1, SEQUENCE_LENGTH, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    # TorchScript
    print("[Export] Exporting to TorchScript...")
    try:
        traced = torch.jit.trace(model, dummy_input)
        traced.save(str(output_dir / "model.pt"))
        print(f"  Saved: {output_dir / 'model.pt'}")
    except Exception as e:
        print(f"  TorchScript export failed: {e}")

    # ONNX
    print("[Export] Exporting to ONNX...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_dir / "model.onnx"),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=14
        )
        print(f"  Saved: {output_dir / 'model.onnx'}")
    except Exception as e:
        print(f"  ONNX export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 + TemporalConv")
    parser.add_argument('--data_dir', type=str, default='data/affectnet')
    parser.add_argument('--output_dir', type=str, default='server_model/checkpoints')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--freeze_backbone', action='store_true')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EfficientNet-B4 + TemporalConv Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load data
    if args.synthetic:
        samples = generate_synthetic_data(args.num_samples)
        # Convert to format expected by dataset
        train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)
    else:
        samples = load_dataset(args.data_dir)
        if len(samples) == 0:
            print("No data found. Use --synthetic for testing.")
            sys.exit(1)
        train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)

    print(f"[Data] Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Create datasets
    train_dataset = VideoEmotionDataset(train_samples, transform=get_transforms(True))
    val_dataset = VideoEmotionDataset(val_samples, transform=get_transforms(False))

    # Balanced sampler
    labels = [s[1] for s in train_samples]
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = EmotionRecognitionModel(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    print(f"\n[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Model] Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = GradScaler()

    # Training loop
    best_f1 = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)

        scheduler.step(val_f1)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
            }, output_dir / "best_model.pth")
            print(f"  Saved best model (F1: {val_f1:.4f})")

    # Load best model
    checkpoint = torch.load(output_dir / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    _, val_acc, val_f1, preds, labels = validate(model, val_loader, criterion, device)
    print(f"\nBest Val F1: {best_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=EMOTION_LABELS))

    # Export
    export_model(model, output_dir, device)

    # Save history
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved to: {output_dir / 'best_model.pth'}")
    print(f"TorchScript model: {output_dir / 'model.pt'}")
    print(f"ONNX model: {output_dir / 'model.onnx'}")


if __name__ == "__main__":
    main()
