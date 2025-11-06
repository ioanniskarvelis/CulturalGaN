"""
Prepare Greek motifs dataset for StyleGAN3 training
Creates the required dataset.json file for StyleGAN3
"""

import json
import pandas as pd
from pathlib import Path
from PIL import Image

def prepare_stylegan3_dataset():
    """
    Prepare dataset in StyleGAN3 format.
    Creates dataset.json with image paths and labels (regions).
    """
    
    # Read metadata
    metadata_path = Path("data/processed/metadata.csv")
    df = pd.read_csv(metadata_path)
    
    print("=" * 60)
    print("PREPARING STYLEGAN3 DATASET")
    print("=" * 60)
    
    # Map regions to numerical labels
    regions = sorted(df['region'].unique())
    region_to_label = {region: idx for idx, region in enumerate(regions)}
    
    print(f"\nRegions ({len(regions)}):")
    for region, label in region_to_label.items():
        count = len(df[df['region'] == region])
        print(f"  {label}: {region} ({count} images)")
    
    # Prepare dataset structure for StyleGAN3
    dataset = {
        "labels": []
    }
    
    processed_dir = Path("data/processed")
    
    for idx, row in df.iterrows():
        # Get image path (relative to processed dir)
        img_path = Path(row['image_path'])
        relative_path = img_path.relative_to(processed_dir)
        
        # Get region label
        region = row['region']
        label = region_to_label[region]
        
        # Add to dataset
        # Format: [["path/to/image.png", label]]
        dataset["labels"].append([str(relative_path).replace('\\', '/'), label])
    
    # Save dataset.json in processed directory
    dataset_json_path = processed_dir / "dataset.json"
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✅ Dataset prepared: {dataset_json_path}")
    print(f"   Total images: {len(dataset['labels'])}")
    
    # Also save region mapping for reference
    mapping_path = processed_dir / "region_labels.json"
    with open(mapping_path, 'w') as f:
        json.dump(region_to_label, f, indent=2)
    
    print(f"✅ Region mapping saved: {mapping_path}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. Copy your processed dataset to StyleGAN3 directory:")
    print("   (Or create a symlink)")
    print()
    print("2. Train StyleGAN3 on your motifs:")
    print("   cd ../stylegan3")
    print("   python train.py \\")
    print("     --outdir=../CulturalGaN/models/checkpoints \\")
    print("     --cfg=stylegan3-t \\")
    print("     --data=../CulturalGaN/data/processed \\")
    print("     --gpus=1 \\")
    print("     --batch=16 \\")
    print("     --gamma=8.2 \\")
    print("     --cond=1 \\")
    print("     --mirror=1")
    print()
    print("3. Monitor training with TensorBoard:")
    print("   tensorboard --logdir=models/checkpoints")
    print()
    print("=" * 60)


if __name__ == "__main__":
    prepare_stylegan3_dataset()

