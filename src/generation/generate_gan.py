"""
GAN-based generation pipeline for Greek motifs
Generates authentic traditional patterns without modernization
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Tuple
import argparse
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.stylegan3_model import StyleGAN3Generator


class GreekMotifGenerator:
    """
    Generator for authentic Greek motifs using trained StyleGAN3.
    Preserves traditional characteristics without modern adaptation.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda"
    ):
        """
        Initialize generator.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run on (cuda/cpu)
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Load model
        self.generator = self._load_generator(checkpoint_path)
        self.generator.eval()
        
        print(f"Loaded generator from {checkpoint_path}")
    
    def _load_generator(self, checkpoint_path: str):
        """
        Load generator model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded generator model
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract config from checkpoint
        config = checkpoint.get('config', {})

        # Get model parameters (with defaults matching stylegan3_greek_simple.yaml)
        img_resolution = config.get('img_resolution', 512)
        z_dim = config.get('z_dim', 512)
        w_dim = config.get('w_dim', 512)
        condition_dim = config.get('condition_dim', 11)
        img_channels = config.get('img_channels', 3)

        # Build generator architecture
        generator = StyleGAN3Generator(
            z_dim=z_dim,
            w_dim=w_dim,
            condition_dim=condition_dim,
            img_resolution=img_resolution,
            img_channels=img_channels
        )

        # Load trained weights
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator = generator.to(self.device)
        generator.eval()

        print(f"✓ Loaded generator (z_dim={z_dim}, resolution={img_resolution}x{img_resolution})")

        return generator
    
    def generate(
        self,
        num_samples: int = 1,
        region: Optional[str] = None,
        seed: Optional[int] = None,
        latent_vectors: Optional[torch.Tensor] = None,
        truncation_psi: float = 0.7
    ) -> List[np.ndarray]:
        """
        Generate Greek motifs.

        Args:
            num_samples: Number of motifs to generate
            region: Optional region conditioning (e.g., 'Cyclades')
            seed: Random seed for reproducibility
            latent_vectors: Optional pre-defined latent vectors
            truncation_psi: Truncation trick parameter (0.0-1.0, lower = more typical samples)

        Returns:
            List of generated images as numpy arrays
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate or use provided latent vectors
        if latent_vectors is None:
            latent_dim = self.generator.z_dim
            latent_vectors = torch.randn(
                num_samples, latent_dim
            ).to(self.device)

        # Region conditioning (if model supports it)
        region_condition = None
        if region is not None:
            region_condition = self._encode_region(region)
            # Expand to match batch size
            if region_condition.shape[0] == 1 and num_samples > 1:
                region_condition = region_condition.repeat(num_samples, 1)

        # Generate images
        with torch.no_grad():
            generated_images = self.generator(
                latent_vectors,
                condition=region_condition,
                truncation_psi=truncation_psi
            )

        # Convert to numpy arrays
        images = []
        for i in range(num_samples):
            img = generated_images[i].cpu()
            img = (img * 0.5 + 0.5).clamp(0, 1)  # Denormalize from [-1, 1] to [0, 1]
            img = img.permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            images.append(img)

        return images
    
    def _encode_region(self, region: str) -> torch.Tensor:
        """
        Encode region name to one-hot conditional vector.

        Args:
            region: Region name (e.g., 'Cyclades')

        Returns:
            Region encoding tensor [1, num_regions]
        """
        # Region mapping (matching training data)
        region_map = {
            'Aegean_Islands': 0,
            'Cyclades': 1,
            'Dodecanese': 2,
            'Epirus': 3,
            'Greece': 4,
            'Lesvos': 5,
            'North_Aegean': 6,
            'Rhodes': 7,
            'Thessaly': 8,
            'Thrace': 9,
            'Turkey': 10
        }

        num_regions = 11
        region_id = region_map.get(region, 0)

        # Create one-hot encoding
        one_hot = torch.zeros(1, num_regions).to(self.device)
        one_hot[0, region_id] = 1.0

        return one_hot
    
    def save_images(
        self,
        images: List[np.ndarray],
        output_dir: str,
        prefix: str = "generated"
    ):
        """
        Save generated images to disk.
        
        Args:
            images: List of images to save
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(images):
            filename = output_path / f"{prefix}_{i:04d}.png"
            Image.fromarray(img).save(filename)
            print(f"Saved: {filename}")
    
    def interpolate(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        steps: int = 10
    ) -> List[np.ndarray]:
        """
        Interpolate between two latent vectors.
        Creates smooth transitions between motifs.
        
        Args:
            start_latent: Starting latent vector
            end_latent: Ending latent vector
            steps: Number of interpolation steps
            
        Returns:
            List of interpolated images
        """
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        interpolated_images = []
        for alpha in alphas:
            # Linear interpolation
            latent = (1 - alpha) * start_latent + alpha * end_latent
            
            # Generate
            images = self.generate(num_samples=1, latent_vectors=latent)
            interpolated_images.extend(images)
        
        return interpolated_images
    
    def generate_variations(
        self,
        base_latent: torch.Tensor,
        num_variations: int = 5,
        variation_strength: float = 0.1
    ) -> List[np.ndarray]:
        """
        Generate variations of a base motif.
        
        Args:
            base_latent: Base latent vector
            num_variations: Number of variations
            variation_strength: Strength of variation (0 to 1)
            
        Returns:
            List of variation images
        """
        variations = []
        
        for _ in range(num_variations):
            # Add controlled noise
            noise = torch.randn_like(base_latent) * variation_strength
            varied_latent = base_latent + noise
            
            # Generate
            images = self.generate(num_samples=1, latent_vectors=varied_latent)
            variations.extend(images)
        
        return variations


def main():
    """Command-line interface for generation."""
    parser = argparse.ArgumentParser(
        description="Generate authentic Greek motifs using trained GAN"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of motifs to generate"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region conditioning (e.g., Cyclades)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/generated",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--truncation_psi",
        type=float,
        default=0.7,
        help="Truncation trick parameter (0.0-1.0, lower = more typical samples)"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = GreekMotifGenerator(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # Generate motifs
    print(f"Generating {args.num_samples} motifs...")
    if args.region:
        print(f"Region conditioning: {args.region}")
    print(f"Truncation: {args.truncation_psi}")

    images = generator.generate(
        num_samples=args.num_samples,
        region=args.region,
        seed=args.seed,
        truncation_psi=args.truncation_psi
    )
    
    # Save images
    generator.save_images(
        images=images,
        output_dir=args.output_dir,
        prefix=f"{args.region or 'greek'}_motif"
    )
    
    print(f"\nGenerated {len(images)} motifs in {args.output_dir}")


if __name__ == "__main__":
    main()

