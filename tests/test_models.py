"""
Unit tests for model architectures.
"""

import unittest
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stylegan3_model import (
    MappingNetwork,
    ModulatedConv2d,
    SynthesisBlock,
    StyleGAN3Generator,
    StyleGAN3Discriminator,
    build_stylegan3
)


class TestMappingNetwork(unittest.TestCase):
    """Tests for MappingNetwork."""

    def test_forward_no_condition(self):
        """Test forward pass without conditioning."""
        model = MappingNetwork(z_dim=512, w_dim=512, condition_dim=0)
        z = torch.randn(4, 512)

        w = model(z)

        self.assertEqual(w.shape, (4, 512))

    def test_forward_with_condition(self):
        """Test forward pass with conditioning."""
        model = MappingNetwork(z_dim=512, w_dim=512, condition_dim=11)
        z = torch.randn(4, 512)
        condition = torch.randn(4, 11)

        w = model(z, condition)

        self.assertEqual(w.shape, (4, 512))


class TestModulatedConv2d(unittest.TestCase):
    """Tests for ModulatedConv2d."""

    def test_forward(self):
        """Test forward pass."""
        model = ModulatedConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            style_dim=512
        )

        x = torch.randn(2, 64, 32, 32)
        w = torch.randn(2, 512)

        out = model(x, w)

        self.assertEqual(out.shape, (2, 128, 32, 32))

    def test_demodulation(self):
        """Test with and without demodulation."""
        model_with_demod = ModulatedConv2d(64, 128, 3, 512, demodulate=True)
        model_without_demod = ModulatedConv2d(64, 128, 3, 512, demodulate=False)

        x = torch.randn(2, 64, 16, 16)
        w = torch.randn(2, 512)

        out1 = model_with_demod(x, w)
        out2 = model_without_demod(x, w)

        self.assertEqual(out1.shape, out2.shape)
        # Results should be different
        self.assertFalse(torch.allclose(out1, out2))


class TestSynthesisBlock(unittest.TestCase):
    """Tests for SynthesisBlock."""

    def test_forward(self):
        """Test forward pass."""
        block = SynthesisBlock(
            in_channels=64,
            out_channels=128,
            w_dim=512,
            use_noise=True
        )

        x = torch.randn(2, 64, 32, 32)
        w = torch.randn(2, 512)

        out = block(x, w)

        self.assertEqual(out.shape, (2, 128, 32, 32))

    def test_without_noise(self):
        """Test forward pass without noise."""
        block = SynthesisBlock(
            in_channels=64,
            out_channels=128,
            w_dim=512,
            use_noise=False
        )

        x = torch.randn(2, 64, 32, 32)
        w = torch.randn(2, 512)

        out = block(x, w)

        self.assertEqual(out.shape, (2, 128, 32, 32))


class TestStyleGAN3Generator(unittest.TestCase):
    """Tests for StyleGAN3Generator."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = StyleGAN3Generator(
            z_dim=512,
            w_dim=512,
            condition_dim=11,
            img_resolution=512,
            img_channels=3
        )

        self.assertEqual(generator.z_dim, 512)
        self.assertEqual(generator.w_dim, 512)
        self.assertEqual(generator.img_resolution, 512)

    def test_forward(self):
        """Test generator forward pass."""
        generator = StyleGAN3Generator(
            z_dim=512,
            w_dim=512,
            condition_dim=0,
            img_resolution=512,
            img_channels=3
        )

        batch_size = 2
        z = torch.randn(batch_size, 512)

        images = generator(z)

        self.assertEqual(images.shape, (batch_size, 3, 512, 512))
        # Check output range (should be tanh: [-1, 1])
        self.assertTrue(images.min() >= -1.0)
        self.assertTrue(images.max() <= 1.0)

    def test_forward_with_conditioning(self):
        """Test generator with conditioning."""
        generator = StyleGAN3Generator(
            z_dim=512,
            w_dim=512,
            condition_dim=11,
            img_resolution=512,
            img_channels=3
        )

        batch_size = 2
        z = torch.randn(batch_size, 512)
        condition = torch.randn(batch_size, 11)

        images = generator(z, condition)

        self.assertEqual(images.shape, (batch_size, 3, 512, 512))

    def test_truncation(self):
        """Test truncation trick."""
        generator = StyleGAN3Generator(
            z_dim=512,
            w_dim=512,
            condition_dim=0,
            img_resolution=512
        )

        z = torch.randn(2, 512)

        # Generate with different truncation values
        img_no_trunc = generator(z, truncation_psi=1.0)
        img_trunc = generator(z, truncation_psi=0.5)

        # Shapes should match
        self.assertEqual(img_no_trunc.shape, img_trunc.shape)

        # Results should be different
        self.assertFalse(torch.allclose(img_no_trunc, img_trunc))


class TestStyleGAN3Discriminator(unittest.TestCase):
    """Tests for StyleGAN3Discriminator."""

    def test_initialization(self):
        """Test discriminator initialization."""
        discriminator = StyleGAN3Discriminator(
            img_resolution=512,
            img_channels=3,
            condition_dim=11
        )

        self.assertEqual(discriminator.img_resolution, 512)
        self.assertEqual(discriminator.condition_dim, 11)

    def test_forward(self):
        """Test discriminator forward pass."""
        discriminator = StyleGAN3Discriminator(
            img_resolution=512,
            img_channels=3,
            condition_dim=0
        )

        batch_size = 2
        images = torch.randn(batch_size, 3, 512, 512)

        scores = discriminator(images)

        self.assertEqual(scores.shape, (batch_size, 1))

    def test_forward_with_conditioning(self):
        """Test discriminator with conditioning."""
        discriminator = StyleGAN3Discriminator(
            img_resolution=512,
            img_channels=3,
            condition_dim=11
        )

        batch_size = 2
        images = torch.randn(batch_size, 3, 512, 512)
        condition = torch.randn(batch_size, 11)

        scores = discriminator(images, condition)

        self.assertEqual(scores.shape, (batch_size, 1))


class TestBuildStyleGAN3(unittest.TestCase):
    """Tests for build_stylegan3 function."""

    def test_build_default(self):
        """Test building with default parameters."""
        generator, discriminator = build_stylegan3()

        self.assertIsInstance(generator, StyleGAN3Generator)
        self.assertIsInstance(discriminator, StyleGAN3Discriminator)

    def test_build_with_params(self):
        """Test building with custom parameters."""
        generator, discriminator = build_stylegan3(
            img_resolution=512,
            z_dim=512,
            w_dim=512,
            condition_dim=11,
            img_channels=3
        )

        self.assertEqual(generator.z_dim, 512)
        self.assertEqual(discriminator.img_resolution, 512)

    def test_parameter_count(self):
        """Test that models have reasonable parameter counts."""
        generator, discriminator = build_stylegan3(img_resolution=512)

        g_params = sum(p.numel() for p in generator.parameters())
        d_params = sum(p.numel() for p in discriminator.parameters())

        # Should have millions of parameters
        self.assertGreater(g_params, 1_000_000)
        self.assertGreater(d_params, 1_000_000)

        print(f"\nGenerator parameters: {g_params:,}")
        print(f"Discriminator parameters: {d_params:,}")


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
