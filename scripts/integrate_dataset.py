"""Integrate the dataset.xlsx and images folder into the project pipeline.

This script:
1. Reads the Excel file with annotations
2. Converts each row to a JSON annotation file
3. Organizes images into data/raw/ by region
4. Creates a dataset summary
"""

import sys
from pathlib import Path
import json
import pandas as pd
import shutil
from collections import defaultdict
import io

# Set UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.utils.config import get_paths
from normalize_regions import normalize_region

def clean_value(val):
    """Clean up string values (remove tabs, extra spaces)."""
    if pd.isna(val):
        return None
    if isinstance(val, str):
        return val.strip().replace('\t', '')
    return val

def create_annotation_from_row(row, image_path):
    """Convert an Excel row to our annotation JSON format."""
    # Generate motif ID from image name
    image_name = row['Image Name']
    motif_id = image_name.replace('.png', '').replace('image', 'MTF_')
    
    # Normalize region
    region_raw = clean_value(row.get('Region', 'Unknown'))
    region_normalized = normalize_region(region_raw)
    
    # Map Excel columns to our template structure
    annotation = {
        "motif_id": motif_id,
        "name": clean_value(row.get('Description', image_name)),
        "region": region_normalized,
        "region_original": region_raw,  # Keep original for reference
        "subregion": clean_value(row.get('Subregion')),
        "period": clean_value(row.get('Period', 'Unknown')),
        "type": clean_value(row.get('Type', 'Unknown')),
        "style": clean_value(row.get('Style')),
        "original_medium": clean_value(row.get('Craft Type', 'Unknown')),
        "dominant_shapes": clean_value(row.get('Dominant Shapes')),
        "symmetry_type": clean_value(row.get('Symmetry Type')),
        "symbolic_meaning": clean_value(row.get('Symbolic Meaning')),
        "motif_elements": clean_value(row.get('Motif Elements')),
        "cultural_significance": clean_value(row.get('Cultural Significance')),
        "source": "Collected Dataset",
        "source_file": "dataset.xlsx",
        "image_name": image_name,
        "file_path": str(image_path),
    }
    
    return annotation

def organize_images_by_region(df, images_dir, target_dir):
    """Copy images to data/raw organized by region."""
    print("\nOrganizing images by region...")
    
    images_dir = Path(images_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    region_counts = defaultdict(int)
    copied_count = 0
    
    for idx, row in df.iterrows():
        image_name = row['Image Name']
        source_path = images_dir / image_name
        
        if not source_path.exists():
            print(f"  [WARNING] Image not found: {image_name}")
            continue
        
        # Get region and normalize it
        region_raw = clean_value(row.get('Region', 'Unknown'))
        region = normalize_region(region_raw)
        if not region or region == 'Unknown':
            region = 'Unknown'
        
        # Create region directory (already has underscores from normalization)
        region_dir = target_dir / region
        region_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy image
        target_path = region_dir / image_name
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
            copied_count += 1
        
        region_counts[region] += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} images...")
    
    print(f"\n[SUCCESS] Copied {copied_count} images to {target_dir}")
    print(f"\nImages by region:")
    for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {region}: {count}")
    
    return region_counts

def create_annotations(df, images_dir, annotations_dir):
    """Create JSON annotation files for each image."""
    print("\nCreating JSON annotations...")
    
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    created_count = 0
    
    for idx, row in df.iterrows():
        image_name = row['Image Name']
        source_path = images_dir / image_name
        
        if not source_path.exists():
            continue
        
        # Create annotation
        annotation = create_annotation_from_row(row, source_path)
        
        # Save to JSON
        json_filename = image_name.replace('.png', '.json')
        json_path = annotations_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
        
        created_count += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Created {idx + 1}/{len(df)} annotations...")
    
    print(f"\n[SUCCESS] Created {created_count} JSON annotations in {annotations_dir}")
    
    return created_count

def create_summary_report(df, output_path):
    """Create a summary report of the dataset."""
    print("\nCreating summary report...")
    
    summary = {
        "total_motifs": len(df),
        "dataset_file": "dataset.xlsx",
        "regions": df['Region'].value_counts().head(15).to_dict(),
        "types": df['Type'].value_counts().head(15).to_dict(),
        "periods": df['Period'].value_counts().head(10).to_dict() if 'Period' in df.columns else {},
        "craft_types": df['Craft Type'].value_counts().head(10).to_dict() if 'Craft Type' in df.columns else {},
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Summary saved to {output_path}")
    
    return summary

def main():
    """Main integration workflow."""
    print("=" * 70)
    print("INTEGRATING DATASET INTO PIPELINE")
    print("=" * 70)
    
    # Get paths
    paths = get_paths()
    
    # Locate files
    excel_path = Path('dataset.xlsx')
    images_dir = Path('images')
    
    if not excel_path.exists():
        print(f"[ERROR] dataset.xlsx not found at {excel_path.absolute()}")
        return
    
    if not images_dir.exists():
        print(f"[ERROR] images/ directory not found at {images_dir.absolute()}")
        return
    
    print(f"\n[SUCCESS] Found dataset.xlsx")
    print(f"[SUCCESS] Found images/ directory")
    
    # Load Excel
    print(f"\nLoading Excel file...")
    df = pd.read_excel(excel_path)
    print(f"[SUCCESS] Loaded {len(df)} motif records")
    
    # Organize images by region
    region_counts = organize_images_by_region(df, images_dir, paths.data_raw)
    
    # Create JSON annotations
    annotation_count = create_annotations(df, images_dir, paths.data_annotations)
    
    # Create summary report
    summary = create_summary_report(df, paths.outputs_eval / 'dataset_summary.json')
    
    print("\n" + "=" * 70)
    print("INTEGRATION COMPLETE!")
    print("=" * 70)
    print(f"\nDataset Statistics:")
    print(f"  Total motifs: {len(df)}")
    print(f"  Images organized: {sum(region_counts.values())}")
    print(f"  Annotations created: {annotation_count}")
    print(f"  Unique regions: {len(region_counts)}")
    print(f"\nNext steps:")
    print(f"  1. Run: python scripts/check_setup.py")
    print(f"  2. Open: notebooks/01_data_exploration.ipynb")
    print(f"  3. Test generation: python scripts/smoke_test.py --input \"data/raw/{list(region_counts.keys())[0]}/image1.png\"")
    print("=" * 70)

if __name__ == "__main__":
    main()

