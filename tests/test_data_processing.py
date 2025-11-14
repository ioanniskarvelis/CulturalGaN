"""
Unit tests for data processing modules.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.preprocess import GreekMotifPreprocessor
from src.data_processing.motif_dataset import GreekMotifDataset


class TestGreekMotifPreprocessor(unittest.TestCase):
    """Tests for GreekMotifPreprocessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir()

        # Create test image
        self.test_image_path = self.input_dir / "test_motif.jpg"
        test_img = Image.new('RGB', (512, 512), color='red')
        test_img.save(self.test_image_path)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = GreekMotifPreprocessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir)
        )

        self.assertEqual(preprocessor.input_dir, self.input_dir)
        self.assertEqual(preprocessor.output_dir, self.output_dir)
        self.assertEqual(preprocessor.target_size, (512, 512))

    def test_process_single_image(self):
        """Test processing a single image."""
        preprocessor = GreekMotifPreprocessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir)
        )

        # Process the test image
        result = preprocessor._process_image(
            self.test_image_path,
            region="Test_Region"
        )

        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('filename', result)
        self.assertIn('region', result)
        self.assertIn('vertical_symmetry', result)
        self.assertIn('horizontal_symmetry', result)
        self.assertIn('edge_density', result)

        # Check values
        self.assertEqual(result['region'], "Test_Region")
        self.assertIsInstance(result['vertical_symmetry'], (float, np.floating))
        self.assertIsInstance(result['horizontal_symmetry'], (float, np.floating))
        self.assertIsInstance(result['edge_density'], (float, np.floating))

    def test_output_directory_creation(self):
        """Test that output directory is created."""
        preprocessor = GreekMotifPreprocessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir)
        )

        # Output dir should be created
        self.assertTrue(self.output_dir.exists())


class TestGreekMotifDataset(unittest.TestCase):
    """Tests for GreekMotifDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "processed"
        self.data_dir.mkdir()

        # Create dummy metadata
        metadata = pd.DataFrame({
            'filename': ['test1.jpg', 'test2.jpg'],
            'image_path': [
                str(self.data_dir / 'test1.jpg'),
                str(self.data_dir / 'test2.jpg')
            ],
            'region': ['Cyclades', 'Rhodes'],
            'vertical_symmetry': [0.8, 0.7],
            'horizontal_symmetry': [0.6, 0.5],
            'edge_density': [0.3, 0.4]
        })

        metadata.to_csv(self.data_dir / 'metadata.csv', index=False)

        # Create dummy images
        for img_name in ['test1.jpg', 'test2.jpg']:
            img = Image.new('RGB', (512, 512), color='blue')
            img.save(self.data_dir / img_name)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = GreekMotifDataset(
            data_dir=str(self.data_dir),
            use_embeddings=False
        )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset.regions), 2)

    def test_dataset_getitem(self):
        """Test getting an item from dataset."""
        dataset = GreekMotifDataset(
            data_dir=str(self.data_dir),
            use_embeddings=False
        )

        sample = dataset[0]

        # Check sample structure
        self.assertIsInstance(sample, dict)
        self.assertIn('image', sample)
        self.assertIn('region_label', sample)
        self.assertIn('region_onehot', sample)
        self.assertIn('geometric_features', sample)

        # Check types
        self.assertIsInstance(sample['image'], torch.Tensor)
        self.assertIsInstance(sample['region_label'], torch.Tensor)
        self.assertIsInstance(sample['region_onehot'], torch.Tensor)
        self.assertIsInstance(sample['geometric_features'], torch.Tensor)

        # Check shapes
        self.assertEqual(sample['image'].shape[0], 3)  # RGB
        self.assertEqual(sample['region_onehot'].shape[0], 2)  # 2 regions
        self.assertEqual(sample['geometric_features'].shape[0], 3)  # 3 features

    def test_region_filtering(self):
        """Test filtering by region."""
        dataset = GreekMotifDataset(
            data_dir=str(self.data_dir),
            regions=['Cyclades'],
            use_embeddings=False
        )

        self.assertEqual(len(dataset), 1)

    def test_region_distribution(self):
        """Test get_region_distribution method."""
        dataset = GreekMotifDataset(
            data_dir=str(self.data_dir),
            use_embeddings=False
        )

        distribution = dataset.get_region_distribution()

        self.assertIsInstance(distribution, dict)
        self.assertEqual(distribution['Cyclades'], 1)
        self.assertEqual(distribution['Rhodes'], 1)


class TestDataPipelineIntegration(unittest.TestCase):
    """Integration tests for data pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "raw"
        self.output_dir = Path(self.temp_dir) / "processed"

        # Create input directory structure
        region_dir = self.input_dir / "TestRegion"
        region_dir.mkdir(parents=True)

        # Create test images
        for i in range(3):
            img = Image.new('RGB', (600, 400), color=(255, i * 50, i * 30))
            img.save(region_dir / f"motif_{i}.jpg")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_processing(self):
        """Test end-to-end preprocessing pipeline."""
        # Preprocess
        preprocessor = GreekMotifPreprocessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir)
        )

        metadata = preprocessor.process_all()

        # Check preprocessing results
        self.assertEqual(len(metadata), 3)
        self.assertTrue((self.output_dir / 'metadata.csv').exists())

        # Load dataset
        dataset = GreekMotifDataset(
            data_dir=str(self.output_dir),
            use_embeddings=False
        )

        self.assertEqual(len(dataset), 3)

        # Check samples
        for i in range(3):
            sample = dataset[i]
            self.assertEqual(sample['image'].shape, (3, 512, 512))


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
