#!/usr/bin/env python3
"""
Evaluation Script for Emotion Recognition Models

Computes:
- Accuracy
- Per-class F1 scores
- Confusion matrix
- ROC curves (if applicable)

Usage:
    python eval.py --model_path checkpoints/best_model.pth --data_dir data/test
    python eval.py --synthetic  # Evaluate on synthetic data
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize

from train import (
    EmotionRecognitionModel,
    VideoEmotionDataset,
    generate_synthetic_data,
    load_dataset,
    get_transforms,
    EMOTION_LABELS,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
    IMAGE_SIZE
)

def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path
) -> dict:
    """
    Comprehensive model evaluation.

    Returns dict with all metrics.
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("[Eval] Running inference...")
    with torch.no_grad():
        for batch_idx, (frames, labels) in enumerate(dataloader):
            frames = frames.to(device)

            logits = model(frames)
            probs = F.softmax(logits, dim=1)

            _, preds = logits.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Basic metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")

    # Per-class metrics
    print("\nPer-class F1 Scores:")
    for label, f1 in zip(EMOTION_LABELS, f1_per_class):
        print(f"  {label:12s}: {f1:.4f}")

    # Classification report
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=EMOTION_LABELS)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix to {output_dir / 'confusion_matrix.png'}")

    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS
    )
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=150)
    plt.close()

    # ROC curves (one-vs-rest)
    print("\nComputing ROC curves...")
    all_labels_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, NUM_CLASSES))

    roc_auc = {}
    for i, (label, color) in enumerate(zip(EMOTION_LABELS, colors)):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc[label] = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{label} (AUC = {roc_auc[label]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=150)
    plt.close()
    print(f"Saved ROC curves to {output_dir / 'roc_curves.png'}")

    print("\nAUC Scores:")
    for label, score in roc_auc.items():
        print(f"  {label:12s}: {score:.4f}")

    # Precision-Recall curves
    print("\nComputing Precision-Recall curves...")
    plt.figure(figsize=(12, 8))

    pr_auc = {}
    for i, (label, color) in enumerate(zip(EMOTION_LABELS, colors)):
        precision, recall, _ = precision_recall_curve(all_labels_bin[:, i], all_probs[:, i])
        pr_auc[label] = auc(recall, precision)
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{label} (AUC = {pr_auc[label]:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curves.png', dpi=150)
    plt.close()
    print(f"Saved PR curves to {output_dir / 'pr_curves.png'}")

    # Compile results
    results = {
        'accuracy': float(accuracy),
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'f1_per_class': {l: float(f) for l, f in zip(EMOTION_LABELS, f1_per_class)},
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(all_labels),
        'class_distribution': {
            l: int((all_labels == i).sum())
            for i, l in enumerate(EMOTION_LABELS)
        }
    }

    # Save results
    with open(output_dir / 'eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_dir / 'eval_results.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Emotion Recognition Model")
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='data/test')
    parser.add_argument('--output_dir', type=str, default='eval_results')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Emotion Recognition Model Evaluation")
    print("=" * 60)
    print(f"Device: {device}")

    # Load model
    model_path = Path(args.model_path)
    if model_path.exists():
        print(f"\nLoading model from {model_path}")
        model = EmotionRecognitionModel(NUM_CLASSES, pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation F1 at checkpoint: {checkpoint.get('val_f1', 'unknown')}")
    else:
        print(f"\nModel not found at {model_path}")
        print("Creating untrained model for demo...")
        model = EmotionRecognitionModel(NUM_CLASSES, pretrained=True)
        model = model.to(device)

    # Load data
    if args.synthetic:
        print(f"\nGenerating {args.num_samples} synthetic samples...")
        samples = generate_synthetic_data(args.num_samples)
    else:
        samples = load_dataset(args.data_dir)
        if len(samples) == 0:
            print(f"No data found in {args.data_dir}. Using synthetic data.")
            samples = generate_synthetic_data(args.num_samples)

    print(f"Evaluating on {len(samples)} samples")

    # Create dataset and loader
    dataset = VideoEmotionDataset(samples, transform=get_transforms(False))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Evaluate
    results = evaluate_model(model, dataloader, device, output_dir)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Weighted F1: {results['f1_weighted']:.4f}")
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
