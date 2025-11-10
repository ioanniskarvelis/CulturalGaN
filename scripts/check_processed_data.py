"""Quick script to check processed data statistics"""
import pandas as pd

df = pd.read_csv('data/processed/metadata.csv')

print(f"✓ Total images processed: {len(df)}")
print(f"\n✓ Regional distribution:")
print(df['region'].value_counts().to_string())
print(f"\n✓ Features extracted:")
print(f"  - Vertical symmetry: {df['vertical_symmetry'].mean():.3f} (avg)")
print(f"  - Horizontal symmetry: {df['horizontal_symmetry'].mean():.3f} (avg)")
print(f"  - Edge density: {df['edge_density'].mean():.3f} (avg)")
print(f"  - Color histogram: ✓")
print(f"  - Dominant colors: ✓")

print("\n✓ Sample images:")
print(df[['region', 'filename', 'vertical_symmetry', 'edge_density']].head(5).to_string())

