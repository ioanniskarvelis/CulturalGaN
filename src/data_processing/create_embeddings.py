"""
Create semantic embeddings for Greek motifs.
Combines visual and textual features for GAN conditioning.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm


class MotifEmbeddingCreator:
    """
    Creates semantic embeddings for Greek motifs combining:
    - Visual features (CLIP)
    - Textual features (symbolic descriptions)
    - Geometric features (from preprocessing)
    """
    
    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        text_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize embedding creator.
        
        Args:
            clip_model: CLIP model variant
            text_model: Sentence transformer model for text
            device: Device to use (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize CLIP
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
            self.clip_available = True
            print(f"✓ Loaded CLIP model: {clip_model}")
        except Exception as e:
            print(f"⚠️  CLIP not available: {e}")
            self.clip_available = False
        
        # Initialize sentence transformer for text embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer(text_model, device=self.device)
            self.text_available = True
            print(f"✓ Loaded text encoder: {text_model}")
        except Exception as e:
            print(f"⚠️  Sentence transformers not available: {e}")
            self.text_available = False
    
    def create_visual_embedding(self, image_path: str) -> np.ndarray:
        """
        Create visual embedding using CLIP.
        
        Args:
            image_path: Path to image
            
        Returns:
            Visual embedding vector
        """
        if not self.clip_available:
            return np.zeros(512)  # Default dimension
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                visual_features = self.clip_model.encode_image(image)
                visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
            
            return visual_features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"Error creating visual embedding for {image_path}: {e}")
            return np.zeros(512)
    
    def create_text_embedding(self, text: str) -> np.ndarray:
        """
        Create text embedding from symbolic description.
        
        Args:
            text: Textual description
            
        Returns:
            Text embedding vector
        """
        if not self.text_available or not text:
            return np.zeros(384)  # Default dimension for MiniLM
        
        try:
            embedding = self.text_encoder.encode(text, convert_to_numpy=True)
            return embedding.flatten()
        except Exception as e:
            print(f"Error creating text embedding: {e}")
            return np.zeros(384)
    
    def create_geometric_embedding(self, geometric_features: Dict) -> np.ndarray:
        """
        Create embedding from geometric features.
        
        Args:
            geometric_features: Dictionary of geometric properties
            
        Returns:
            Geometric feature vector
        """
        features = []
        
        # Symmetry features
        features.append(geometric_features.get('vertical_symmetry', 0.0))
        features.append(geometric_features.get('horizontal_symmetry', 0.0))
        
        # Edge density
        features.append(geometric_features.get('edge_density', 0.0))
        
        # Pad to fixed size (e.g., 16 dimensions)
        while len(features) < 16:
            features.append(0.0)
        
        return np.array(features[:16], dtype=np.float32)
    
    def create_combined_embedding(
        self,
        image_path: str,
        symbolic_analysis: Dict,
        geometric_features: Dict,
        region: str
    ) -> Dict[str, np.ndarray]:
        """
        Create combined multi-modal embedding.
        
        Args:
            image_path: Path to motif image
            symbolic_analysis: Symbolic analysis results
            geometric_features: Geometric features
            region: Regional label
            
        Returns:
            Dictionary of different embedding types
        """
        embeddings = {}
        
        # Visual embedding
        embeddings['visual'] = self.create_visual_embedding(image_path)
        
        # Text embedding from symbolic description
        text_description = self._create_text_description(
            symbolic_analysis, region
        )
        embeddings['text'] = self.create_text_embedding(text_description)
        embeddings['text_description'] = text_description
        
        # Geometric embedding
        embeddings['geometric'] = self.create_geometric_embedding(geometric_features)
        
        # Combined embedding (concatenate all)
        embeddings['combined'] = np.concatenate([
            embeddings['visual'],
            embeddings['text'],
            embeddings['geometric']
        ])
        
        # Region one-hot encoding (for 11 regions)
        region_embedding = self._create_region_embedding(region)
        embeddings['region'] = region_embedding
        
        return embeddings
    
    def _create_text_description(
        self,
        symbolic_analysis: Dict,
        region: str
    ) -> str:
        """
        Create comprehensive text description from symbolic analysis.
        
        Args:
            symbolic_analysis: Analysis results
            region: Region name
            
        Returns:
            Formatted text description
        """
        parts = []
        
        # Region
        parts.append(f"Traditional Greek motif from {region} region.")
        
        # Pattern type
        pattern_type = symbolic_analysis.get('pattern_type', 'traditional')
        parts.append(f"Pattern type: {pattern_type}.")
        
        # Geometric structure
        geometric = symbolic_analysis.get('geometric_structure', '')
        if geometric:
            parts.append(geometric)
        
        # Cultural symbolism
        symbolism = symbolic_analysis.get('cultural_symbolism', '')
        if symbolism:
            parts.append(symbolism)
        
        # Key features
        key_features = symbolic_analysis.get('key_features', [])
        if key_features:
            features_str = ', '.join(key_features)
            parts.append(f"Key features: {features_str}.")
        
        return ' '.join(parts)
    
    def _create_region_embedding(self, region: str) -> np.ndarray:
        """
        Create one-hot encoding for region.
        
        Args:
            region: Region name
            
        Returns:
            One-hot vector
        """
        regions = [
            'Aegean_Islands', 'Cyclades', 'Dodecanese', 'Epirus',
            'Greece', 'Lesvos', 'North_Aegean', 'Rhodes',
            'Thessaly', 'Thrace', 'Turkey'
        ]
        
        embedding = np.zeros(len(regions), dtype=np.float32)
        
        if region in regions:
            embedding[regions.index(region)] = 1.0
        
        return embedding
    
    def process_dataset(
        self,
        annotations_path: str,
        output_path: str,
        save_format: str = 'npz'
    ) -> Dict:
        """
        Process entire dataset and create embeddings.
        
        Args:
            annotations_path: Path to annotations.json
            output_path: Output directory for embeddings
            save_format: Format to save ('npz' or 'pt')
            
        Returns:
            Dictionary with embedding statistics
        """
        print("\n" + "="*70)
        print("CREATING SEMANTIC EMBEDDINGS")
        print("="*70)
        
        # Load annotations
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"\n✓ Loaded {len(annotations)} annotations")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Process each motif
        all_embeddings = []
        metadata = []
        
        for idx, (image_path, data) in enumerate(tqdm(annotations.items(), desc="Creating embeddings")):
            try:
                # Extract data
                symbolic_analysis = data.get('symbolic_analysis', {})
                geometric_features = data.get('geometric_features', {})
                region = data.get('region', 'Unknown')
                filename = data.get('filename', '')
                
                # Create embeddings
                embeddings = self.create_combined_embedding(
                    image_path=image_path,
                    symbolic_analysis=symbolic_analysis,
                    geometric_features=geometric_features,
                    region=region
                )
                
                # Store
                all_embeddings.append(embeddings)
                metadata.append({
                    'index': idx,
                    'filename': filename,
                    'region': region,
                    'image_path': image_path,
                    'text_description': embeddings['text_description']
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Convert to arrays
        embedding_arrays = {
            'visual': np.stack([e['visual'] for e in all_embeddings]),
            'text': np.stack([e['text'] for e in all_embeddings]),
            'geometric': np.stack([e['geometric'] for e in all_embeddings]),
            'combined': np.stack([e['combined'] for e in all_embeddings]),
            'region': np.stack([e['region'] for e in all_embeddings])
        }
        
        # Save embeddings
        if save_format == 'npz':
            embeddings_file = Path(output_path) / "embeddings.npz"
            np.savez_compressed(embeddings_file, **embedding_arrays)
            print(f"\n✓ Saved embeddings to: {embeddings_file}")
        elif save_format == 'pt':
            embeddings_file = Path(output_path) / "embeddings.pt"
            torch_embeddings = {k: torch.from_numpy(v) for k, v in embedding_arrays.items()}
            torch.save(torch_embeddings, embeddings_file)
            print(f"\n✓ Saved embeddings to: {embeddings_file}")
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_file = Path(output_path) / "embeddings_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        print(f"✓ Saved metadata to: {metadata_file}")
        
        # Statistics
        stats = {
            'total_embeddings': len(all_embeddings),
            'visual_dim': embedding_arrays['visual'].shape[1],
            'text_dim': embedding_arrays['text'].shape[1],
            'geometric_dim': embedding_arrays['geometric'].shape[1],
            'combined_dim': embedding_arrays['combined'].shape[1],
            'region_dim': embedding_arrays['region'].shape[1]
        }
        
        print(f"\n✓ Embedding Statistics:")
        print(f"  Total embeddings: {stats['total_embeddings']}")
        print(f"  Visual dimension: {stats['visual_dim']}")
        print(f"  Text dimension: {stats['text_dim']}")
        print(f"  Geometric dimension: {stats['geometric_dim']}")
        print(f"  Combined dimension: {stats['combined_dim']}")
        print(f"  Region dimension: {stats['region_dim']}")
        
        # Save statistics
        stats_file = Path(output_path) / "embedding_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats


if __name__ == "__main__":
    """
    Example usage:
        python src/data_processing/create_embeddings.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Create semantic embeddings for Greek motifs")
    parser.add_argument(
        "--annotations",
        default="data/annotations/annotations.json",
        help="Path to annotations file"
    )
    parser.add_argument(
        "--output",
        default="data/embeddings",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--format",
        default="npz",
        choices=["npz", "pt"],
        help="Save format (npz or pytorch)"
    )
    
    args = parser.parse_args()
    
    # Create embeddings
    creator = MotifEmbeddingCreator()
    
    stats = creator.process_dataset(
        annotations_path=args.annotations,
        output_path=args.output,
        save_format=args.format
    )
    
    print("\n" + "="*70)
    print("✓ EMBEDDING CREATION COMPLETE!")
    print("="*70)
    print(f"\nNext step: Train StyleGAN3 model")
    print(f"  python src/models/stylegan3_trainer.py")

