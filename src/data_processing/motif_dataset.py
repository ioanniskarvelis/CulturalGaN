"""
PyTorch Dataset for Greek motifs with embeddings.
Loads processed images, geometric features, and semantic embeddings.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Optional, Dict, Tuple, List


class GreekMotifDataset(Dataset):
    """
    Dataset for Greek traditional motifs.
    Loads images with corresponding embeddings and metadata.
    """
    
    def __init__(
        self,
        data_dir: str = "data/processed",
        embeddings_dir: str = "data/embeddings",
        img_size: int = 512,
        use_embeddings: bool = True,
        regions: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory with processed images
            embeddings_dir: Directory with embeddings
            img_size: Image size
            use_embeddings: Whether to load embeddings
            regions: Optional list of regions to include
            transform: Optional image transform
        """
        self.data_dir = Path(data_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.img_size = img_size
        self.use_embeddings = use_embeddings
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.csv"
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter by regions if specified
        if regions:
            self.metadata = self.metadata[self.metadata['region'].isin(regions)]
        
        # Load embeddings if available
        if use_embeddings:
            embeddings_path = self.embeddings_dir / "embeddings.npz"
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)
                
                # Load embedding metadata to match indices
                emb_metadata_path = self.embeddings_dir / "embeddings_metadata.csv"
                self.emb_metadata = pd.read_csv(emb_metadata_path)
            else:
                print(f"Warning: Embeddings not found at {embeddings_path}")
                self.use_embeddings = False
                self.embeddings = None
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
            ])
        else:
            self.transform = transform
        
        # Region mapping
        self.regions = sorted(self.metadata['region'].unique().tolist())
        self.region_to_idx = {region: idx for idx, region in enumerate(self.regions)}
        
        print(f"✓ Loaded {len(self.metadata)} motifs from {len(self.regions)} regions")
        if self.use_embeddings:
            print(f"✓ Embeddings loaded: {list(self.embeddings.keys())}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - image: Image tensor [3, H, W]
            - region_label: Region index
            - region_onehot: One-hot region encoding
            - geometric_features: Geometric features
            - embedding: Semantic embedding (if available)
        """
        row = self.metadata.iloc[idx]
        
        # Load image
        img_path = Path(row['image_path'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Region label
        region = row['region']
        region_idx = self.region_to_idx[region]
        region_onehot = torch.zeros(len(self.regions))
        region_onehot[region_idx] = 1.0
        
        # Geometric features
        geometric_features = torch.tensor([
            row.get('vertical_symmetry', 0.0),
            row.get('horizontal_symmetry', 0.0),
            row.get('edge_density', 0.0)
        ], dtype=torch.float32)
        
        sample = {
            'image': image,
            'region_label': torch.tensor(region_idx, dtype=torch.long),
            'region_onehot': region_onehot,
            'geometric_features': geometric_features,
            'filename': row['filename']
        }
        
        # Add embeddings if available
        if self.use_embeddings and self.embeddings is not None:
            # Find corresponding embedding
            emb_idx = self._find_embedding_index(row['image_path'])
            
            if emb_idx is not None:
                sample['embedding_visual'] = torch.from_numpy(
                    self.embeddings['visual'][emb_idx]
                ).float()
                sample['embedding_text'] = torch.from_numpy(
                    self.embeddings['text'][emb_idx]
                ).float()
                sample['embedding_geometric'] = torch.from_numpy(
                    self.embeddings['geometric'][emb_idx]
                ).float()
                sample['embedding_combined'] = torch.from_numpy(
                    self.embeddings['combined'][emb_idx]
                ).float()
        
        return sample
    
    def _find_embedding_index(self, image_path: str) -> Optional[int]:
        """Find embedding index for given image path."""
        if not hasattr(self, 'emb_metadata'):
            return None
        
        # Normalize paths
        image_path = str(image_path).replace('\\', '/')
        
        # Search in embedding metadata
        matches = self.emb_metadata[
            self.emb_metadata['image_path'].str.replace('\\', '/') == image_path
        ]
        
        if len(matches) > 0:
            return matches.iloc[0]['index']
        
        return None
    
    def get_region_distribution(self) -> Dict[str, int]:
        """Get distribution of regions in dataset."""
        return self.metadata['region'].value_counts().to_dict()
    
    def get_sample_by_region(self, region: str, n: int = 5) -> List[Dict]:
        """Get n samples from specific region."""
        region_data = self.metadata[self.metadata['region'] == region]
        samples = []
        
        for idx in region_data.index[:n]:
            samples.append(self[self.metadata.index.get_loc(idx)])
        
        return samples


def create_dataloaders(
    data_dir: str = "data/processed",
    embeddings_dir: str = "data/embeddings",
    batch_size: int = 16,
    img_size: int = 512,
    train_split: float = 0.9,
    use_embeddings: bool = True,
    num_workers: int = 4,
    regions: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory with processed images
        embeddings_dir: Directory with embeddings
        batch_size: Batch size
        img_size: Image size
        train_split: Fraction for training
        use_embeddings: Whether to use embeddings
        num_workers: Number of data loading workers
        regions: Optional list of regions to include
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = GreekMotifDataset(
        data_dir=data_dir,
        embeddings_dir=embeddings_dir,
        img_size=img_size,
        use_embeddings=use_embeddings,
        regions=regions
    )
    
    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"✓ Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


if __name__ == "__main__":
    """Test dataset"""
    print("Testing Greek Motif Dataset...")
    
    # Create dataset
    dataset = GreekMotifDataset(
        data_dir="data/processed",
        embeddings_dir="data/embeddings",
        use_embeddings=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Regions: {dataset.regions}")
    print(f"\nRegion distribution:")
    for region, count in dataset.get_region_distribution().items():
        print(f"  {region}: {count}")
    
    # Test loading
    print(f"\nTesting sample loading...")
    sample = dataset[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Region: {dataset.regions[sample['region_label'].item()]}")
    print(f"  Geometric features: {sample['geometric_features']}")
    
    if 'embedding_combined' in sample:
        print(f"  Combined embedding shape: {sample['embedding_combined'].shape}")
    
    # Test dataloader
    print(f"\nTesting dataloader...")
    train_loader, val_loader = create_dataloaders(
        batch_size=4,
        num_workers=0  # 0 for testing
    )
    
    batch = next(iter(train_loader))
    print(f"  Batch image shape: {batch['image'].shape}")
    print(f"  Batch regions: {batch['region_label']}")
    
    print("\n✓ Dataset test passed!")

