"""
Check preprocessing results
"""
import pandas as pd
from pathlib import Path

# Read metadata
metadata_path = Path("data/processed/metadata.csv")
df = pd.read_csv(metadata_path)

print("=" * 60)
print("PREPROCESSING RESULTS")
print("=" * 60)

print(f"\nTotal processed images: {len(df)}")

print("\n" + "=" * 60)
print("REGIONAL DISTRIBUTION")
print("=" * 60)
print(df['region'].value_counts())

print("\n" + "=" * 60)
print("AVERAGE FEATURE SCORES")
print("=" * 60)
print(f"Vertical Symmetry:   {df['vertical_symmetry'].mean():.4f}")
print(f"Horizontal Symmetry: {df['horizontal_symmetry'].mean():.4f}")
print(f"Edge Density:        {df['edge_density'].mean():.4f}")

print("\n" + "=" * 60)
print("SAMPLE DATA (First 3 images)")
print("=" * 60)
print(df[['filename', 'region', 'vertical_symmetry', 'horizontal_symmetry', 'edge_density']].head(3))

print("\n" + "=" * 60)
print("✅ Preprocessing completed successfully!")
print("=" * 60)
print("\nNext steps:")
print("1. Review the processed images in data/processed/")
print("2. Check metadata.csv for extracted features")
print("3. Proceed to Phase 2: Model Implementation")

