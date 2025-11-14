"""
Cultural authenticity loss functions for Greek motif generation.
Uses multi-modal embeddings to preserve traditional characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer


class AuthenticityLoss(nn.Module):
    """
    Combined authenticity loss using multi-modal embeddings.
    Preserves cultural characteristics through:
    1. Visual features (CLIP)
    2. Geometric features
    3. Color distributions
    4. Symmetry patterns
    """

    def __init__(
        self,
        device: str = "cuda",
        use_clip: bool = True,
        color_weight: float = 0.3,
        geometric_weight: float = 0.3,
        visual_weight: float = 0.2,
        symmetry_weight: float = 0.2
    ):
        super().__init__()
        self.device = device
        self.use_clip = use_clip

        # Loss weights
        self.color_weight = color_weight
        self.geometric_weight = geometric_weight
        self.visual_weight = visual_weight
        self.symmetry_weight = symmetry_weight

        # CLIP model for visual feature extraction
        if use_clip:
            print("Loading CLIP model for authenticity loss...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(device)
            self.clip_model.eval()

            # Freeze CLIP
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            self.clip_model = None

        # Sobel filters for edge detection
        self.register_buffer(
            'sobel_x',
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )

    @torch.no_grad()
    def extract_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP visual features from images.

        Args:
            images: Images [batch, 3, H, W] in range [-1, 1]

        Returns:
            CLIP features [batch, 512]
        """
        if self.clip_model is None:
            return None

        # Convert to [0, 1]
        images = (images + 1) / 2

        # Convert to PIL format for CLIP processor
        # CLIP expects [0, 255] uint8
        images_np = (images * 255).clamp(0, 255).byte().cpu()

        # Process images
        inputs = self.clip_processor(
            images=[img.permute(1, 2, 0).numpy() for img in images_np],
            return_tensors="pt"
        ).to(self.device)

        # Extract features
        features = self.clip_model.get_image_features(**inputs)
        features = F.normalize(features, dim=-1)

        return features

    def color_distribution_loss(
        self,
        fake_images: torch.Tensor,
        real_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Match color distributions between real and fake images.
        Uses both mean and variance matching.

        Args:
            fake_images: Generated images [batch, 3, H, W]
            real_images: Real images [batch, 3, H, W]

        Returns:
            Color distribution loss
        """
        # Compute color statistics per channel
        fake_mean = fake_images.mean(dim=[2, 3])  # [batch, 3]
        real_mean = real_images.mean(dim=[2, 3])

        fake_std = fake_images.std(dim=[2, 3])
        real_std = real_images.std(dim=[2, 3])

        # Mean matching
        mean_loss = F.mse_loss(fake_mean, real_mean)

        # Variance matching
        std_loss = F.mse_loss(fake_std, real_std)

        # Color histogram matching (simplified)
        # Flatten spatial dimensions
        fake_flat = fake_images.reshape(fake_images.size(0), 3, -1)  # [batch, 3, H*W]
        real_flat = real_images.reshape(real_images.size(0), 3, -1)

        # Compute percentiles as proxy for histogram
        percentiles = torch.tensor([0.25, 0.5, 0.75], device=self.device)
        fake_percentiles = torch.quantile(fake_flat, percentiles.unsqueeze(1).unsqueeze(2), dim=2)
        real_percentiles = torch.quantile(real_flat, percentiles.unsqueeze(1).unsqueeze(2), dim=2)

        percentile_loss = F.mse_loss(fake_percentiles, real_percentiles)

        return mean_loss + 0.5 * std_loss + 0.3 * percentile_loss

    def geometric_feature_loss(
        self,
        fake_images: torch.Tensor,
        real_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Match geometric features (edges, patterns).

        Args:
            fake_images: Generated images [batch, 3, H, W]
            real_images: Real images [batch, 3, H, W]

        Returns:
            Geometric feature loss
        """
        # Convert to grayscale
        fake_gray = 0.299 * fake_images[:, 0:1] + 0.587 * fake_images[:, 1:2] + 0.114 * fake_images[:, 2:3]
        real_gray = 0.299 * real_images[:, 0:1] + 0.587 * real_images[:, 1:2] + 0.114 * real_images[:, 2:3]

        # Edge detection
        fake_edges_x = F.conv2d(fake_gray, self.sobel_x, padding=1)
        fake_edges_y = F.conv2d(fake_gray, self.sobel_y, padding=1)
        fake_edges = torch.sqrt(fake_edges_x ** 2 + fake_edges_y ** 2)

        real_edges_x = F.conv2d(real_gray, self.sobel_x, padding=1)
        real_edges_y = F.conv2d(real_gray, self.sobel_y, padding=1)
        real_edges = torch.sqrt(real_edges_x ** 2 + real_edges_y ** 2)

        # Edge density matching
        fake_density = fake_edges.mean()
        real_density = real_edges.mean()
        density_loss = F.mse_loss(fake_density, real_density)

        # Edge distribution matching
        edge_dist_loss = F.mse_loss(
            fake_edges.flatten(1).mean(dim=0),
            real_edges.flatten(1).mean(dim=0)
        )

        return density_loss + 0.5 * edge_dist_loss

    def symmetry_loss(
        self,
        fake_images: torch.Tensor,
        real_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Match symmetry patterns (key characteristic of Greek motifs).

        Args:
            fake_images: Generated images [batch, 3, H, W]
            real_images: Real images [batch, 3, H, W]

        Returns:
            Symmetry loss
        """
        # Vertical symmetry (left-right)
        fake_flipped_v = torch.flip(fake_images, dims=[3])
        real_flipped_v = torch.flip(real_images, dims=[3])

        fake_v_sym = F.mse_loss(fake_images, fake_flipped_v)
        real_v_sym = F.mse_loss(real_images, real_flipped_v)
        v_loss = F.mse_loss(fake_v_sym, real_v_sym)

        # Horizontal symmetry (top-bottom)
        fake_flipped_h = torch.flip(fake_images, dims=[2])
        real_flipped_h = torch.flip(real_images, dims=[2])

        fake_h_sym = F.mse_loss(fake_images, fake_flipped_h)
        real_h_sym = F.mse_loss(real_images, real_flipped_h)
        h_loss = F.mse_loss(fake_h_sym, real_h_sym)

        return v_loss + h_loss

    def visual_feature_loss(
        self,
        fake_images: torch.Tensor,
        real_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Match visual features using CLIP embeddings.

        Args:
            fake_images: Generated images [batch, 3, H, W] in range [-1, 1]
            real_images: Real images [batch, 3, H, W] in range [-1, 1]

        Returns:
            Visual feature loss
        """
        if self.clip_model is None:
            return torch.tensor(0.0, device=self.device)

        # Extract CLIP features
        with torch.no_grad():
            fake_features = self.extract_clip_features(fake_images)
            real_features = self.extract_clip_features(real_images)

        # Match feature distributions
        fake_mean = fake_features.mean(dim=0)
        real_mean = real_features.mean(dim=0)

        feature_loss = F.mse_loss(fake_mean, real_mean)

        # Also match pairwise similarities
        fake_sim = torch.mm(fake_features, fake_features.t())
        real_sim = torch.mm(real_features, real_features.t())

        similarity_loss = F.mse_loss(fake_sim, real_sim)

        return feature_loss + 0.3 * similarity_loss

    def forward(
        self,
        fake_images: torch.Tensor,
        real_images: torch.Tensor,
        fake_embeddings: Optional[torch.Tensor] = None,
        real_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined authenticity loss.

        Args:
            fake_images: Generated images [batch, 3, H, W] in range [-1, 1]
            real_images: Real images [batch, 3, H, W] in range [-1, 1]
            fake_embeddings: Optional pre-computed embeddings for fake images [batch, dim]
            real_embeddings: Optional embeddings for real images [batch, dim]

        Returns:
            Dictionary with individual losses and total
        """
        losses = {}

        # Color distribution loss
        losses['color'] = self.color_distribution_loss(fake_images, real_images)

        # Geometric feature loss
        losses['geometric'] = self.geometric_feature_loss(fake_images, real_images)

        # Symmetry loss
        losses['symmetry'] = self.symmetry_loss(fake_images, real_images)

        # Visual feature loss (CLIP)
        if self.use_clip:
            losses['visual'] = self.visual_feature_loss(fake_images, real_images)
        else:
            losses['visual'] = torch.tensor(0.0, device=self.device)

        # Embedding matching loss (if provided)
        if fake_embeddings is not None and real_embeddings is not None:
            # Match embedding distributions
            fake_emb_mean = fake_embeddings.mean(dim=0)
            real_emb_mean = real_embeddings.mean(dim=0)
            losses['embedding'] = F.mse_loss(fake_emb_mean, real_emb_mean)
        else:
            losses['embedding'] = torch.tensor(0.0, device=self.device)

        # Combined weighted loss
        total_loss = (
            self.color_weight * losses['color'] +
            self.geometric_weight * losses['geometric'] +
            self.visual_weight * losses['visual'] +
            self.symmetry_weight * losses['symmetry']
        )

        losses['total'] = total_loss

        return losses


class EmbeddingMatcher(nn.Module):
    """
    Matches embeddings between real and generated images.
    Used when pre-computed embeddings are available.
    """

    def __init__(self, embedding_dim: int = 912):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(
        self,
        fake_embeddings: torch.Tensor,
        real_embeddings: torch.Tensor,
        margin: float = 0.2
    ) -> torch.Tensor:
        """
        Compute embedding matching loss.

        Args:
            fake_embeddings: Embeddings for generated images [batch, dim]
            real_embeddings: Embeddings for real images [batch, dim]
            margin: Margin for contrastive loss

        Returns:
            Embedding loss
        """
        # Normalize embeddings
        fake_norm = F.normalize(fake_embeddings, dim=1)
        real_norm = F.normalize(real_embeddings, dim=1)

        # Distribution matching
        fake_mean = fake_norm.mean(dim=0)
        real_mean = real_norm.mean(dim=0)
        mean_loss = F.mse_loss(fake_mean, real_mean)

        # Covariance matching (simplified)
        fake_cov = torch.mm(fake_norm.t(), fake_norm) / fake_norm.size(0)
        real_cov = torch.mm(real_norm.t(), real_norm) / real_norm.size(0)
        cov_loss = F.mse_loss(fake_cov, real_cov)

        # Pairwise similarity matching
        fake_sim = torch.mm(fake_norm, fake_norm.t())
        real_sim = torch.mm(real_norm, real_norm.t())
        sim_loss = F.mse_loss(fake_sim, real_sim)

        return mean_loss + 0.5 * cov_loss + 0.3 * sim_loss
