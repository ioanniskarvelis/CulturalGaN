"""
Evaluation metrics for Greek motif generation.
Includes standard GAN metrics and cultural authenticity measures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights
import lpips


class InceptionFeatureExtractor(nn.Module):
    """
    InceptionV3 feature extractor for FID and IS calculations.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

        # Load pre-trained InceptionV3
        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.inception = inception_v3(weights=weights, transform_input=False)
        self.inception.eval()
        self.inception = self.inception.to(device)

        # Remove final FC layer to get features
        self.inception.fc = nn.Identity()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract InceptionV3 features.

        Args:
            x: Images [batch, 3, H, W] in range [0, 1]

        Returns:
            Features [batch, 2048]
        """
        # Resize to 299x299 for InceptionV3
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Normalize to [-1, 1] for Inception
        x = x * 2 - 1

        # Extract features
        features = self.inception(x)

        return features


def calculate_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Calculate Fréchet Inception Distance (FID).

    Args:
        real_features: Real image features [N, 2048]
        fake_features: Generated image features [M, 2048]
        eps: Small constant for numerical stability

    Returns:
        FID score (lower is better)
    """
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Calculate squared difference of means
    diff = mu_real - mu_fake

    # Calculate sqrt of product of covariances
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

    # Check for imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)

    return float(fid)


def calculate_inception_score(
    features: torch.Tensor,
    inception_model: nn.Module,
    images: torch.Tensor,
    splits: int = 10,
    eps: float = 1e-16
) -> Tuple[float, float]:
    """
    Calculate Inception Score (IS).

    Args:
        features: Pre-computed features (not used, kept for API compatibility)
        inception_model: Full inception model with classifier
        images: Generated images [N, 3, H, W] in range [0, 1]
        splits: Number of splits for computing mean/std
        eps: Small constant for numerical stability

    Returns:
        Tuple of (mean_is, std_is)
    """
    device = images.device

    # Load full InceptionV3 with classifier
    weights = Inception_V3_Weights.IMAGENET1K_V1
    inception_full = inception_v3(weights=weights, transform_input=False)
    inception_full.eval()
    inception_full = inception_full.to(device)

    # Get predictions
    with torch.no_grad():
        # Resize to 299x299
        if images.shape[-1] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        # Normalize to [-1, 1]
        images = images * 2 - 1

        # Get predictions
        preds = F.softmax(inception_full(images), dim=1).cpu().numpy()

    # Calculate IS for each split
    n = preds.shape[0]
    split_scores = []

    for i in range(splits):
        start_idx = i * n // splits
        end_idx = (i + 1) * n // splits
        split_preds = preds[start_idx:end_idx]

        # KL divergence
        kl = split_preds * (np.log(split_preds + eps) - np.log(np.mean(split_preds, axis=0, keepdims=True) + eps))
        kl = np.mean(np.sum(kl, axis=1))

        split_scores.append(np.exp(kl))

    mean_is = float(np.mean(split_scores))
    std_is = float(np.std(split_scores))

    return mean_is, std_is


def calculate_precision_recall(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    k: int = 3
) -> Tuple[float, float]:
    """
    Calculate Precision and Recall using k-NN.

    Improved Precision and Recall metric from:
    "Improved Precision and Recall Metric for Assessing Generative Models"

    Args:
        real_features: Real image features [N, D]
        fake_features: Generated image features [M, D]
        k: Number of nearest neighbors

    Returns:
        Tuple of (precision, recall)
    """
    # Compute pairwise distances
    def compute_manifold(features_a, features_b, k):
        """
        Check if points in features_a are within the manifold of features_b.
        """
        # Compute distances from each point in A to all points in B
        distances = np.sqrt(np.sum((features_a[:, None, :] - features_b[None, :, :]) ** 2, axis=2))

        # For each point in A, find k-th nearest neighbor in B
        kth_distances_b = np.partition(
            np.sqrt(np.sum((features_b[:, None, :] - features_b[None, :, :]) ** 2, axis=2)),
            k, axis=1
        )[:, k]

        # Check if points in A are within k-NN radius of B
        manifold_distances = np.min(distances, axis=1)
        kth_distances_expanded = kth_distances_b[np.argmin(distances, axis=1)]

        in_manifold = manifold_distances <= kth_distances_expanded

        return np.mean(in_manifold)

    # Precision: fraction of generated samples that fall within real manifold
    precision = compute_manifold(fake_features, real_features, k)

    # Recall: fraction of real samples that are covered by generated manifold
    recall = compute_manifold(real_features, fake_features, k)

    return float(precision), float(recall)


class LPIPSMetric:
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric.
    Measures perceptual similarity between images.
    """

    def __init__(self, device: str = "cuda", net: str = "alex"):
        """
        Initialize LPIPS metric.

        Args:
            device: Device to run on
            net: Network to use ('alex', 'vgg', 'squeeze')
        """
        self.device = device
        self.lpips_model = lpips.LPIPS(net=net).to(device)
        self.lpips_model.eval()

    @torch.no_grad()
    def calculate(self, images_a: torch.Tensor, images_b: torch.Tensor) -> float:
        """
        Calculate LPIPS distance between two sets of images.

        Args:
            images_a: First set of images [N, 3, H, W] in range [0, 1]
            images_b: Second set of images [N, 3, H, W] in range [0, 1]

        Returns:
            Mean LPIPS distance
        """
        # Convert to [-1, 1] for LPIPS
        images_a = images_a * 2 - 1
        images_b = images_b * 2 - 1

        # Calculate LPIPS
        distances = self.lpips_model(images_a, images_b)

        return float(distances.mean())


class AuthenticityMetrics:
    """
    Custom metrics for evaluating cultural authenticity of generated motifs.
    Measures preservation of traditional Greek design characteristics.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def calculate_color_preservation(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> float:
        """
        Measure how well generated images preserve traditional color palettes.

        Args:
            real_images: Real motifs [N, 3, H, W] in range [0, 1]
            fake_images: Generated motifs [M, 3, H, W] in range [0, 1]

        Returns:
            Color preservation score (0-1, higher is better)
        """
        # Extract color histograms
        def get_color_histogram(images, bins=64):
            """Get RGB color histogram."""
            images_np = images.cpu().numpy()
            histograms = []

            for img in images_np:
                # Flatten spatial dimensions
                img_flat = img.reshape(3, -1)  # [3, H*W]

                # Calculate histogram for each channel
                hist_r = np.histogram(img_flat[0], bins=bins, range=(0, 1))[0]
                hist_g = np.histogram(img_flat[1], bins=bins, range=(0, 1))[0]
                hist_b = np.histogram(img_flat[2], bins=bins, range=(0, 1))[0]

                # Concatenate and normalize
                hist = np.concatenate([hist_r, hist_g, hist_b])
                hist = hist / (hist.sum() + 1e-8)

                histograms.append(hist)

            return np.array(histograms)

        real_hists = get_color_histogram(real_images)
        fake_hists = get_color_histogram(fake_images)

        # Calculate mean histogram for each set
        real_mean_hist = np.mean(real_hists, axis=0)
        fake_mean_hist = np.mean(fake_hists, axis=0)

        # Calculate histogram intersection (similarity)
        intersection = np.minimum(real_mean_hist, fake_mean_hist).sum()

        return float(intersection)

    def calculate_symmetry_preservation(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Measure preservation of symmetry (key characteristic of Greek motifs).

        Args:
            real_images: Real motifs [N, 3, H, W]
            fake_images: Generated motifs [M, 3, H, W]

        Returns:
            Tuple of (vertical_symmetry_score, horizontal_symmetry_score)
        """
        def calculate_symmetry_score(images):
            """Calculate average symmetry score."""
            # Vertical symmetry (left-right)
            flipped_v = torch.flip(images, dims=[3])
            v_diff = torch.abs(images - flipped_v).mean()
            v_symmetry = 1.0 - v_diff.item()

            # Horizontal symmetry (top-bottom)
            flipped_h = torch.flip(images, dims=[2])
            h_diff = torch.abs(images - flipped_h).mean()
            h_symmetry = 1.0 - h_diff.item()

            return v_symmetry, h_symmetry

        real_v, real_h = calculate_symmetry_score(real_images)
        fake_v, fake_h = calculate_symmetry_score(fake_images)

        # Score is how close fake symmetry is to real symmetry
        v_score = 1.0 - abs(real_v - fake_v)
        h_score = 1.0 - abs(real_h - fake_h)

        return float(v_score), float(h_score)

    def calculate_edge_density_preservation(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> float:
        """
        Measure preservation of geometric complexity (edge density).

        Args:
            real_images: Real motifs [N, 3, H, W]
            fake_images: Generated motifs [M, 3, H, W]

        Returns:
            Edge density preservation score (0-1, higher is better)
        """
        def get_edge_density(images):
            """Calculate mean edge density using Sobel filter."""
            # Convert to grayscale
            gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
            gray = gray.unsqueeze(1)  # [N, 1, H, W]

            # Sobel filters
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

            sobel_x = sobel_x.view(1, 1, 3, 3).to(images.device)
            sobel_y = sobel_y.view(1, 1, 3, 3).to(images.device)

            # Compute gradients
            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)

            # Gradient magnitude
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

            # Mean edge density
            return grad_mag.mean().item()

        real_density = get_edge_density(real_images)
        fake_density = get_edge_density(fake_images)

        # Score is inverse of relative difference
        score = 1.0 - abs(real_density - fake_density) / (real_density + 1e-8)

        return float(max(0.0, score))


class GAN_Evaluator:
    """
    Complete evaluation suite for GAN-generated Greek motifs.
    Combines standard GAN metrics with cultural authenticity measures.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize evaluator.

        Args:
            device: Device to run evaluation on
        """
        self.device = device
        self.inception_extractor = InceptionFeatureExtractor(device)
        self.lpips_metric = LPIPSMetric(device)
        self.authenticity_metrics = AuthenticityMetrics(device)

    @torch.no_grad()
    def extract_features(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract InceptionV3 features from a dataset.

        Args:
            dataloader: DataLoader with images
            max_samples: Maximum number of samples to process

        Returns:
            Features array [N, 2048]
        """
        features_list = []
        n_samples = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)

            # Extract features
            features = self.inception_extractor(images)
            features_list.append(features.cpu().numpy())

            n_samples += images.shape[0]
            if max_samples and n_samples >= max_samples:
                break

        features = np.concatenate(features_list, axis=0)

        if max_samples:
            features = features[:max_samples]

        return features

    def evaluate(
        self,
        real_dataloader: DataLoader,
        fake_dataloader: DataLoader,
        max_samples: int = 10000,
        calculate_is: bool = True,
        calculate_pr: bool = True,
        calculate_lpips: bool = False,
        calculate_authenticity: bool = True
    ) -> dict:
        """
        Run complete evaluation.

        Args:
            real_dataloader: DataLoader with real images
            fake_dataloader: DataLoader with generated images
            max_samples: Maximum samples to use for evaluation
            calculate_is: Whether to calculate Inception Score
            calculate_pr: Whether to calculate Precision/Recall
            calculate_lpips: Whether to calculate LPIPS (expensive)
            calculate_authenticity: Whether to calculate cultural metrics

        Returns:
            Dictionary with all metrics
        """
        print("Extracting features from real images...")
        real_features = self.extract_features(real_dataloader, max_samples)

        print("Extracting features from generated images...")
        fake_features = self.extract_features(fake_dataloader, max_samples)

        results = {}

        # FID (always calculated)
        print("Calculating FID...")
        fid = calculate_fid(real_features, fake_features)
        results['fid'] = fid

        # Inception Score
        if calculate_is:
            print("Calculating Inception Score...")
            # Need to reload images for IS
            fake_images_list = []
            for i, batch in enumerate(fake_dataloader):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                fake_images_list.append(images.to(self.device))
                if len(fake_images_list) * images.shape[0] >= min(max_samples, 5000):
                    break

            fake_images_tensor = torch.cat(fake_images_list, dim=0)[:min(max_samples, 5000)]
            is_mean, is_std = calculate_inception_score(
                None, None, fake_images_tensor
            )
            results['is_mean'] = is_mean
            results['is_std'] = is_std

        # Precision & Recall
        if calculate_pr:
            print("Calculating Precision & Recall...")
            precision, recall = calculate_precision_recall(real_features, fake_features)
            results['precision'] = precision
            results['recall'] = recall

        # Cultural authenticity metrics
        if calculate_authenticity:
            print("Calculating cultural authenticity metrics...")

            # Load real and fake images
            real_images_list = []
            fake_images_list = []

            for batch in real_dataloader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                real_images_list.append(images.to(self.device))
                if len(real_images_list) * images.shape[0] >= 1000:
                    break

            for batch in fake_dataloader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                fake_images_list.append(images.to(self.device))
                if len(fake_images_list) * images.shape[0] >= 1000:
                    break

            real_images = torch.cat(real_images_list, dim=0)[:1000]
            fake_images = torch.cat(fake_images_list, dim=0)[:1000]

            # Color preservation
            color_score = self.authenticity_metrics.calculate_color_preservation(
                real_images, fake_images
            )
            results['color_preservation'] = color_score

            # Symmetry preservation
            v_sym, h_sym = self.authenticity_metrics.calculate_symmetry_preservation(
                real_images, fake_images
            )
            results['vertical_symmetry'] = v_sym
            results['horizontal_symmetry'] = h_sym

            # Edge density preservation
            edge_score = self.authenticity_metrics.calculate_edge_density_preservation(
                real_images, fake_images
            )
            results['edge_density_preservation'] = edge_score

            # Overall authenticity score (weighted average)
            authenticity_score = (
                0.3 * color_score +
                0.25 * v_sym +
                0.25 * h_sym +
                0.2 * edge_score
            )
            results['authenticity_score'] = authenticity_score

        return results


def print_evaluation_results(results: dict):
    """
    Pretty-print evaluation results.

    Args:
        results: Dictionary of evaluation metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n📊 Standard GAN Metrics:")
    print(f"  FID Score:                {results.get('fid', 'N/A'):.3f}")

    if 'is_mean' in results:
        print(f"  Inception Score:          {results['is_mean']:.3f} ± {results['is_std']:.3f}")

    if 'precision' in results:
        print(f"  Precision:                {results['precision']:.3f}")
        print(f"  Recall:                   {results['recall']:.3f}")

    if 'authenticity_score' in results:
        print("\n🏛️  Cultural Authenticity Metrics:")
        print(f"  Overall Authenticity:     {results['authenticity_score']:.3f}")
        print(f"  Color Preservation:       {results['color_preservation']:.3f}")
        print(f"  Vertical Symmetry:        {results['vertical_symmetry']:.3f}")
        print(f"  Horizontal Symmetry:      {results['horizontal_symmetry']:.3f}")
        print(f"  Edge Density:             {results['edge_density_preservation']:.3f}")

    print("\n" + "=" * 60 + "\n")
