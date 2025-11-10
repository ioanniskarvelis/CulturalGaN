"""
Training script for StyleGAN3 on Greek motifs.
Implements CDGFD methodology for cultural preservation.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.stylegan3_model import build_stylegan3
from src.data_processing.motif_dataset import create_dataloaders


class StyleGAN3Trainer:
    """
    Trainer for StyleGAN3 on Greek motifs.
    Includes custom losses for cultural authenticity preservation.
    """
    
    def __init__(
        self,
        config: Dict,
        device: str = "cuda"
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            device: Device to train on
        """
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Build models
        print("Building models...")
        self.generator, self.discriminator = build_stylegan3(
            img_resolution=config['img_resolution'],
            z_dim=config['z_dim'],
            w_dim=config['w_dim'],
            condition_dim=config['condition_dim'],
            img_channels=config['img_channels']
        )
        
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        # Count parameters
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"  Generator: {g_params:,} parameters")
        print(f"  Discriminator: {d_params:,} parameters")
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config['g_lr'],
            betas=(config['beta1'], config['beta2'])
        )
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config['d_lr'],
            betas=(config['beta1'], config['beta2'])
        )
        
        # Loss weights
        self.lambda_gp = config.get('lambda_gp', 10.0)  # Gradient penalty
        self.lambda_auth = config.get('lambda_auth', 0.5)  # Authenticity loss
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Create output directories
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'models/checkpoints'))
        self.sample_dir = Path(config.get('sample_dir', 'outputs/samples'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_gradient_penalty(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.
        """
        batch_size = real_images.size(0)
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
        
        # Discriminator output
        d_interpolates = self.discriminator(interpolates, condition)
        
        # Gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def compute_authenticity_loss(
        self,
        fake_images: torch.Tensor,
        real_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Custom authenticity loss to preserve cultural features.
        Matches color distributions and geometric properties.
        """
        # Color distribution matching (histogram matching)
        fake_hist = fake_images.mean(dim=[2, 3])  # [batch, 3]
        real_hist = real_images.mean(dim=[2, 3])
        color_loss = nn.functional.mse_loss(fake_hist, real_hist)
        
        # Edge consistency (geometric preservation)
        # Simple sobel filter approximation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().to(self.device)
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        fake_edges = nn.functional.conv2d(
            fake_images,
            sobel_x,
            padding=1,
            groups=3
        ).abs().mean()
        
        real_edges = nn.functional.conv2d(
            real_images,
            sobel_x,
            padding=1,
            groups=3
        ).abs().mean()
        
        edge_loss = nn.functional.mse_loss(fake_edges, real_edges)
        
        # Combined authenticity loss
        auth_loss = color_loss + 0.5 * edge_loss
        
        return auth_loss
    
    def train_step(
        self,
        real_images: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Single training step.
        """
        batch_size = real_images.size(0)
        
        # ==================
        # Train Discriminator
        # ==================
        self.discriminator.train()
        self.generator.eval()
        
        # Real images
        real_output = self.discriminator(real_images, condition)
        d_loss_real = -real_output.mean()  # WGAN loss
        
        # Fake images
        z = torch.randn(batch_size, self.config['z_dim']).to(self.device)
        with torch.no_grad():
            fake_images = self.generator(z, condition)
        
        fake_output = self.discriminator(fake_images.detach(), condition)
        d_loss_fake = fake_output.mean()
        
        # Gradient penalty
        gp = self.compute_gradient_penalty(real_images, fake_images, condition)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake + self.lambda_gp * gp
        
        # Update discriminator
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        # ==================
        # Train Generator
        # ==================
        self.generator.train()
        
        # Generate fake images
        z = torch.randn(batch_size, self.config['z_dim']).to(self.device)
        fake_images = self.generator(z, condition)
        
        # Generator loss (WGAN)
        fake_output = self.discriminator(fake_images, condition)
        g_loss_adv = -fake_output.mean()
        
        # Authenticity loss
        auth_loss = self.compute_authenticity_loss(fake_images, real_images)
        
        # Total generator loss
        g_loss = g_loss_adv + self.lambda_auth * auth_loss
        
        # Update generator
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        # Return losses
        return {
            'd_loss': d_loss.item(),
            'd_loss_real': d_loss_real.item(),
            'd_loss_fake': d_loss_fake.item(),
            'gp': gp.item(),
            'g_loss': g_loss.item(),
            'g_loss_adv': g_loss_adv.item(),
            'auth_loss': auth_loss.item()
        }
    
    @torch.no_grad()
    def generate_samples(
        self,
        n_samples: int = 16,
        condition: Optional[torch.Tensor] = None,
        truncation: float = 0.7
    ) -> torch.Tensor:
        """Generate sample images."""
        self.generator.eval()
        
        z = torch.randn(n_samples, self.config['z_dim']).to(self.device)
        
        if condition is None and self.config['condition_dim'] > 0:
            # Random regions
            condition = torch.randn(n_samples, self.config['condition_dim']).to(self.device)
        
        samples = self.generator(z, condition, truncation_psi=truncation)
        
        return samples
    
    def save_samples(self, epoch: int, n_samples: int = 16):
        """Save sample images."""
        samples = self.generate_samples(n_samples)
        
        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # Create grid
        grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)
        
        # Save
        save_path = self.sample_dir / f"samples_epoch_{epoch:04d}.png"
        torchvision.utils.save_image(grid, save_path)
        
        print(f"  Saved samples to {save_path}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"  Resumed from epoch {self.current_epoch}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        save_interval: int = 10,
        sample_interval: int = 5
    ):
        """
        Main training loop.
        """
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        for epoch in range(self.current_epoch, n_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            epoch_losses = {
                'd_loss': [],
                'g_loss': [],
                'auth_loss': []
            }
            
            pbar = tqdm(train_loader, desc=f"Training")
            for batch in pbar:
                # Move to device
                real_images = batch['image'].to(self.device)
                condition = batch['region_onehot'].to(self.device)
                
                # Training step
                losses = self.train_step(real_images, condition)
                
                # Track losses
                epoch_losses['d_loss'].append(losses['d_loss'])
                epoch_losses['g_loss'].append(losses['g_loss'])
                epoch_losses['auth_loss'].append(losses['auth_loss'])
                
                # Update progress bar
                pbar.set_postfix({
                    'D': f"{losses['d_loss']:.3f}",
                    'G': f"{losses['g_loss']:.3f}",
                    'Auth': f"{losses['auth_loss']:.3f}"
                })
                
                self.global_step += 1
            
            # Epoch summary
            print(f"  D Loss: {np.mean(epoch_losses['d_loss']):.4f}")
            print(f"  G Loss: {np.mean(epoch_losses['g_loss']):.4f}")
            print(f"  Auth Loss: {np.mean(epoch_losses['auth_loss']):.4f}")
            
            # Generate samples
            if (epoch + 1) % sample_interval == 0:
                self.save_samples(epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Train StyleGAN3 on Greek motifs")
    parser.add_argument(
        "--config",
        default="configs/stylegan3_greek.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.epochs:
        config['n_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    
    # Fix for Windows: set num_workers=0 to avoid multiprocessing issues
    import platform
    num_workers = 0 if platform.system() == 'Windows' else config.get('num_workers', 4)
    if num_workers == 0:
        print("  Note: Using single-process data loading (num_workers=0) for Windows compatibility")
    
    train_loader, val_loader = create_dataloaders(
        data_dir=config.get('data_dir', 'data/processed'),
        embeddings_dir=config.get('embeddings_dir', 'data/embeddings'),
        batch_size=config['batch_size'],
        img_size=config['img_resolution'],
        use_embeddings=config.get('use_embeddings', True),
        num_workers=num_workers
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = StyleGAN3Trainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config['n_epochs'],
        save_interval=config.get('save_interval', 10),
        sample_interval=config.get('sample_interval', 5)
    )


if __name__ == "__main__":
    main()

