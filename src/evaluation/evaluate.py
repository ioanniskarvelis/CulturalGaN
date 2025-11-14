"""
Evaluation script for trained StyleGAN3 models.
Computes FID, IS, Precision/Recall, and cultural authenticity metrics.
"""

import argparse
import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import GAN_Evaluator, print_evaluation_results
from src.data_processing.motif_dataset import GreekMotifDataset
from src.generation.generate_gan import GreekMotifGenerator
from torchvision import transforms


class GeneratedDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for generated images.
    """

    def __init__(self, images: list):
        """
        Args:
            images: List of numpy arrays [H, W, 3] in range [0, 255]
        """
        self.images = images
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Convert to tensor [3, H, W] in range [0, 1]
        img_tensor = self.transform(img)
        return img_tensor


def generate_samples(
    checkpoint_path: str,
    num_samples: int,
    batch_size: int = 16,
    device: str = "cuda"
):
    """
    Generate samples from a trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device to use

    Returns:
        List of generated images
    """
    print(f"Loading generator from {checkpoint_path}...")
    generator = GreekMotifGenerator(checkpoint_path, device)

    print(f"Generating {num_samples} samples...")
    all_images = []

    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)

        images = generator.generate(
            num_samples=current_batch_size,
            truncation_psi=0.7
        )

        all_images.extend(images)

        if (i + current_batch_size) % 100 == 0:
            print(f"  Generated {i + current_batch_size}/{num_samples}")

    return all_images


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained StyleGAN3 model on Greek motifs"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory with real images"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to generate for evaluation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for generation and evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--skip_is",
        action="store_true",
        help="Skip Inception Score calculation (faster)"
    )
    parser.add_argument(
        "--skip_pr",
        action="store_true",
        help="Skip Precision/Recall calculation"
    )
    parser.add_argument(
        "--skip_authenticity",
        action="store_true",
        help="Skip cultural authenticity metrics"
    )

    args = parser.parse_args()

    # Generate samples
    print("\n" + "=" * 60)
    print("STEP 1: Generating Samples")
    print("=" * 60)

    fake_images = generate_samples(
        args.checkpoint,
        args.num_samples,
        args.batch_size,
        args.device
    )

    # Create fake dataset and dataloader
    fake_dataset = GeneratedDataset(fake_images)
    fake_dataloader = DataLoader(
        fake_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Load real dataset
    print("\n" + "=" * 60)
    print("STEP 2: Loading Real Dataset")
    print("=" * 60)

    real_dataset = GreekMotifDataset(
        data_dir=args.data_dir,
        transform=transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    )

    real_dataloader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Loaded {len(real_dataset)} real images")

    # Run evaluation
    print("\n" + "=" * 60)
    print("STEP 3: Computing Metrics")
    print("=" * 60)

    evaluator = GAN_Evaluator(device=args.device)

    results = evaluator.evaluate(
        real_dataloader=real_dataloader,
        fake_dataloader=fake_dataloader,
        max_samples=args.num_samples,
        calculate_is=not args.skip_is,
        calculate_pr=not args.skip_pr,
        calculate_authenticity=not args.skip_authenticity
    )

    # Print results
    print_evaluation_results(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
