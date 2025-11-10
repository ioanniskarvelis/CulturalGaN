"""
StyleGAN3-inspired architecture for Greek motif generation.
Adapted for cultural preservation with conditional generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class MappingNetwork(nn.Module):
    """
    Mapping network: Maps latent code z to intermediate latent w.
    With conditioning on region and semantic embeddings.
    """
    
    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        condition_dim: int = 0,  # For region/semantic conditioning
        num_layers: int = 8,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.condition_dim = condition_dim
        
        # Build mapping layers
        layers = []
        in_dim = z_dim + condition_dim
        
        for i in range(num_layers):
            out_dim = w_dim if i == num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(0.2) if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        
        self.mapping = nn.Sequential(*layers)
        
    def forward(
        self,
        z: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            z: Latent code [batch, z_dim]
            condition: Optional conditioning vector [batch, condition_dim]
        Returns:
            w: Intermediate latent [batch, w_dim]
        """
        # Normalize input
        z = F.normalize(z, dim=1)
        
        # Concatenate with condition if provided
        if condition is not None and self.condition_dim > 0:
            x = torch.cat([z, condition], dim=1)
        else:
            x = z
        
        # Map to w space
        w = self.mapping(x)
        
        return w


class ModulatedConv2d(nn.Module):
    """
    Modulated convolution layer (core of StyleGAN).
    Modulates conv weights based on style vector w.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        style_dim: int = 512,
        demodulate: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        
        # Conv weight
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        
        # Style modulation
        self.style = nn.Linear(style_dim, in_channels, bias=True)
        self.style.bias.data.fill_(1.0)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch, in_channels, height, width]
            w: Style vector [batch, style_dim]
        Returns:
            Modulated convolution output
        """
        batch_size = x.shape[0]
        
        # Get modulation factors
        style = self.style(w)  # [batch, in_channels]
        
        # Modulate weights
        weight = self.weight.unsqueeze(0)  # [1, out, in, k, k]
        weight = weight * style.view(batch_size, 1, -1, 1, 1)  # [batch, out, in, k, k]
        
        # Demodulation
        if self.demodulate:
            d = torch.rsqrt(weight.pow(2).sum([2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * d
        
        # Reshape for group convolution
        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        weight = weight.reshape(-1, self.in_channels, self.kernel_size, self.kernel_size)
        
        # Perform convolution
        padding = self.kernel_size // 2
        out = F.conv2d(x, weight, padding=padding, groups=batch_size)
        
        # Reshape back
        out = out.reshape(batch_size, self.out_channels, out.shape[2], out.shape[3])
        
        # Add bias
        out = out + self.bias.view(1, -1, 1, 1)
        
        return out


class SynthesisBlock(nn.Module):
    """
    Single synthesis block with modulated convolutions and noise injection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int = 512,
        use_noise: bool = True
    ):
        super().__init__()
        
        self.use_noise = use_noise
        
        # Main path
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, w_dim)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, w_dim)
        
        # Noise
        if use_noise:
            self.noise_strength1 = nn.Parameter(torch.zeros(1))
            self.noise_strength2 = nn.Parameter(torch.zeros(1))
        
        # Activation
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [batch, in_channels, height, width]
            w: Style vector [batch, w_dim]
            noise: Optional noise tensors
        Returns:
            Output features
        """
        # First conv
        x = self.conv1(x, w)
        
        # Add noise
        if self.use_noise and noise is not None:
            x = x + noise[0] * self.noise_strength1
        
        x = self.activation(x)
        
        # Second conv
        x = self.conv2(x, w)
        
        # Add noise
        if self.use_noise and noise is not None:
            x = x + noise[1] * self.noise_strength2
        
        x = self.activation(x)
        
        return x


class StyleGAN3Generator(nn.Module):
    """
    StyleGAN3-inspired generator for Greek motifs.
    Progressive synthesis with style modulation.
    """
    
    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        condition_dim: int = 0,
        img_resolution: int = 512,
        img_channels: int = 3,
        channel_base: int = 32768,
        channel_max: int = 512
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim, condition_dim)
        
        # Calculate number of blocks
        self.num_blocks = int(np.log2(img_resolution)) - 1  # e.g., 512 -> 8 blocks
        
        # Initial constant
        self.const = nn.Parameter(torch.randn(1, self.get_num_channels(4), 4, 4))
        
        # Synthesis blocks
        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        for res in [2**i for i in range(3, 3 + self.num_blocks)]:
            in_ch = self.get_num_channels(res // 2)
            out_ch = self.get_num_channels(res)
            
            self.blocks.append(SynthesisBlock(in_ch, out_ch, w_dim))
            self.to_rgb.append(nn.Conv2d(out_ch, img_channels, 1))
        
    def get_num_channels(self, resolution: int) -> int:
        """Calculate number of channels for given resolution."""
        # Decreasing channels as resolution increases
        base = 32768  # channel_base
        max_ch = 512  # channel_max
        
        channels = min(max_ch, base // resolution)
        return channels
    
    def forward(
        self,
        z: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        truncation_psi: float = 1.0
    ) -> torch.Tensor:
        """
        Args:
            z: Latent code [batch, z_dim]
            condition: Optional conditioning [batch, condition_dim]
            truncation_psi: Truncation trick parameter
        Returns:
            Generated images [batch, img_channels, resolution, resolution]
        """
        batch_size = z.shape[0]
        
        # Map to w space
        w = self.mapping(z, condition)
        
        # Truncation trick
        if truncation_psi < 1.0:
            w = w * truncation_psi
        
        # Start from constant
        x = self.const.repeat(batch_size, 1, 1, 1)
        
        # Progressive synthesis
        for i, (block, to_rgb) in enumerate(zip(self.blocks, self.to_rgb)):
            # Upsample (except first block)
            if i > 0:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Apply synthesis block
            x = block(x, w)
            
            # Convert to RGB (use last layer's output)
            if i == len(self.blocks) - 1:
                img = to_rgb(x)
                img = torch.tanh(img)  # [-1, 1]
        
        return img


class StyleGAN3Discriminator(nn.Module):
    """
    Discriminator for StyleGAN3.
    Progressive discrimination with gradient penalty support.
    """
    
    def __init__(
        self,
        img_resolution: int = 512,
        img_channels: int = 3,
        channel_base: int = 32768,
        channel_max: int = 512,
        condition_dim: int = 0
    ):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.condition_dim = condition_dim
        
        # From RGB layers
        self.from_rgb = nn.ModuleList()
        
        # Discrimination blocks
        self.blocks = nn.ModuleList()
        
        # Build blocks (reverse of generator)
        resolutions = [2**i for i in range(int(np.log2(img_resolution)), 2, -1)]
        
        for i, res in enumerate(resolutions):
            in_ch = self.get_num_channels(res)
            out_ch = self.get_num_channels(res // 2) if res > 4 else in_ch
            
            # From RGB
            if i == 0:
                self.from_rgb.append(nn.Conv2d(img_channels, in_ch, 1))
            
            # Discrimination block
            block = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            )
            self.blocks.append(block)
        
        # Final layers
        final_ch = self.get_num_channels(4)
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_ch * 4 * 4 + condition_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )
        
    def get_num_channels(self, resolution: int) -> int:
        """Calculate number of channels for given resolution."""
        base = 32768
        max_ch = 512
        return min(max_ch, base // resolution)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input images [batch, channels, height, width]
            condition: Optional conditioning [batch, condition_dim]
        Returns:
            Discrimination scores [batch, 1]
        """
        # From RGB
        x = self.from_rgb[0](x)
        
        # Progressive discrimination
        for block in self.blocks:
            x = block(x)
        
        # Flatten
        x = x.reshape(x.shape[0], -1)
        
        # Concatenate condition
        if condition is not None and self.condition_dim > 0:
            x = torch.cat([x, condition], dim=1)
        
        # Final classification
        out = self.final(x)
        
        return out


def build_stylegan3(
    img_resolution: int = 512,
    z_dim: int = 512,
    w_dim: int = 512,
    condition_dim: int = 0,
    img_channels: int = 3
) -> Tuple[StyleGAN3Generator, StyleGAN3Discriminator]:
    """
    Build StyleGAN3 generator and discriminator.
    
    Args:
        img_resolution: Output image resolution
        z_dim: Latent code dimension
        w_dim: Intermediate latent dimension
        condition_dim: Conditioning vector dimension
        img_channels: Number of image channels
        
    Returns:
        Tuple of (generator, discriminator)
    """
    generator = StyleGAN3Generator(
        z_dim=z_dim,
        w_dim=w_dim,
        condition_dim=condition_dim,
        img_resolution=img_resolution,
        img_channels=img_channels
    )
    
    discriminator = StyleGAN3Discriminator(
        img_resolution=img_resolution,
        img_channels=img_channels,
        condition_dim=condition_dim
    )
    
    return generator, discriminator


if __name__ == "__main__":
    # Test model
    print("Testing StyleGAN3 model...")
    
    batch_size = 4
    z_dim = 512
    condition_dim = 11  # for regions
    
    # Build models
    G, D = build_stylegan3(
        img_resolution=512,
        z_dim=z_dim,
        condition_dim=condition_dim
    )
    
    # Test generator
    z = torch.randn(batch_size, z_dim)
    condition = torch.randn(batch_size, condition_dim)
    
    fake_img = G(z, condition)
    print(f"Generated image shape: {fake_img.shape}")
    
    # Test discriminator
    score = D(fake_img, condition)
    print(f"Discriminator score shape: {score.shape}")
    
    # Count parameters
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\nGenerator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    print("\n✓ Model test passed!")

