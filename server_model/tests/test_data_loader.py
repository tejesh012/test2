#!/usr/bin/env python3
"""
Unit tests for server model data loading and shapes.
"""

import sys
import unittest
import numpy as np

sys.path.insert(0, '..')

from train import (
    EMOTION_LABELS,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
    IMAGE_SIZE,
    EMBEDDING_DIM,
    generate_synthetic_data,
    VideoEmotionDataset,
    get_transforms,
)


class TestDataShapes(unittest.TestCase):
    """Test data loading and tensor shapes."""

    def setUp(self):
        """Generate synthetic data for testing."""
        self.samples = generate_synthetic_data(num_samples=10)

    def test_synthetic_data_structure(self):
        """Test synthetic data has correct structure."""
        self.assertEqual(len(self.samples), 10)

        for frames, label in self.samples:
            self.assertIsInstance(frames, list)
            self.assertEqual(len(frames), SEQUENCE_LENGTH)
            self.assertIsInstance(label, int)
            self.assertTrue(0 <= label < NUM_CLASSES)

    def test_synthetic_frame_shape(self):
        """Test synthetic frames have correct shape."""
        frames, _ = self.samples[0]

        for frame in frames:
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.shape, (IMAGE_SIZE, IMAGE_SIZE, 3))
            self.assertEqual(frame.dtype, np.uint8)

    def test_dataset_output_shape(self):
        """Test dataset outputs correct tensor shapes."""
        dataset = VideoEmotionDataset(
            self.samples,
            transform=get_transforms(train=False)
        )

        frames, label = dataset[0]

        # Frames: (T, C, H, W)
        self.assertEqual(frames.shape, (SEQUENCE_LENGTH, 3, IMAGE_SIZE, IMAGE_SIZE))

        # Label: scalar
        self.assertIsInstance(label, int)

    def test_dataset_length(self):
        """Test dataset length matches samples."""
        dataset = VideoEmotionDataset(
            self.samples,
            transform=get_transforms(train=False)
        )

        self.assertEqual(len(dataset), len(self.samples))

    def test_label_distribution(self):
        """Test labels cover expected range."""
        labels = [s[1] for s in self.samples]

        self.assertTrue(all(0 <= l < NUM_CLASSES for l in labels))

    def test_emotion_labels_count(self):
        """Test we have 7 emotion labels."""
        self.assertEqual(len(EMOTION_LABELS), 7)
        self.assertEqual(NUM_CLASSES, 7)

    def test_constants(self):
        """Test model constants are as expected."""
        self.assertEqual(SEQUENCE_LENGTH, 32)
        self.assertEqual(IMAGE_SIZE, 224)
        self.assertEqual(EMBEDDING_DIM, 512)


class TestTransforms(unittest.TestCase):
    """Test data augmentation transforms."""

    def test_train_transform_output(self):
        """Test train transform produces correct output."""
        from PIL import Image
        import torch

        transform = get_transforms(train=True)
        img = Image.new('RGB', (256, 256), color='red')

        output = transform(img)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (3, IMAGE_SIZE, IMAGE_SIZE))

    def test_val_transform_output(self):
        """Test validation transform produces correct output."""
        from PIL import Image
        import torch

        transform = get_transforms(train=False)
        img = Image.new('RGB', (256, 256), color='blue')

        output = transform(img)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (3, IMAGE_SIZE, IMAGE_SIZE))

    def test_normalization(self):
        """Test images are normalized to expected range."""
        from PIL import Image
        import torch

        transform = get_transforms(train=False)
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        output = transform(img)

        # Normalized values should be around 0 for gray image
        # Given ImageNet normalization
        self.assertTrue(output.min() > -3)
        self.assertTrue(output.max() < 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
