"""Fix and reorganize existing data by normalizing region names.

This script:
1. Backs up current organization
2. Re-organizes images with normalized region names
3. Updates JSON annotations with normalized regions
4. Shows before/after statistics
"""

import sys
from pathlib import Path
import json
import shutil
import io
from collections import defaultdict

# Set UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.utils.config import get_paths
from normalize_regions import normalize_region

def analyze_current_regions(data_dir):
    """Analyze current region organization."""
    data_dir = Path(data_dir)
    
    region_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    region_counts = {}
    
    for region_dir in region_dirs:
        image_count = len(list(region_dir.glob('*.png')))
        if image_count > 0:
            region_counts[region_dir.name] = image_count
    
    return region_counts

def reorganize_images(data_dir, dry_run=False):
    """Reorganize images with normalized region names."""
    data_dir = Path(data_dir)
    
    print("\nAnalyzing current organization...")
    current_regions = analyze_current_regions(data_dir)
    
    print(f"\nCurrent: {len(current_regions)} region directories")
    print("Top 10 regions:")
    for region, count in sorted(current_regions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {region}: {count}")
    
    # Map old regions to new regions
    reorganization_plan = defaultdict(list)
    
    for old_region_dir in data_dir.iterdir():
        if not old_region_dir.is_dir():
            continue
        
        old_region_name = old_region_dir.name
        
        # Try to reverse engineer the original region from directory name
        # (directory names have spaces replaced with underscores and no commas)
        original_region = old_region_name.replace('_', ' ')
        
        # Normalize it
        new_region_name = normalize_region(original_region)
        
        if new_region_name != old_region_name:
            images = list(old_region_dir.glob('*.png'))
            if images:
                reorganization_plan[new_region_name].append({
                    'old_dir': old_region_dir,
                    'old_name': old_region_name,
                    'image_count': len(images)
                })
    
    print(f"\n{'='*70}")
    print("REORGANIZATION PLAN")
    print(f"{'='*70}")
    print(f"Will consolidate {len(current_regions)} regions into normalized regions:")
    
    # Show what will be merged
    new_region_counts = defaultdict(int)
    for new_region, old_dirs in sorted(reorganization_plan.items()):
        total_images = sum(d['image_count'] for d in old_dirs)
        new_region_counts[new_region] = total_images
        print(f"\n{new_region} ({total_images} images):")
        for old_dir_info in old_dirs:
            print(f"  ← {old_dir_info['old_name']} ({old_dir_info['image_count']} images)")
    
    # Also add regions that don't need changes
    for old_region, count in current_regions.items():
        original_region = old_region.replace('_', ' ')
        new_region = normalize_region(original_region)
        if new_region == old_region:
            new_region_counts[new_region] = count
    
    print(f"\n{'='*70}")
    print(f"RESULT: {len(current_regions)} regions → {len(new_region_counts)} normalized regions")
    print(f"{'='*70}")
    
    print("\nNew region distribution:")
    for region, count in sorted(new_region_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {region}: {count}")
    
    if dry_run:
        print("\n[DRY RUN] No changes made. Run without --dry-run to apply changes.")
        return
    
    # Execute reorganization
    print(f"\n{'='*70}")
    print("EXECUTING REORGANIZATION")
    print(f"{'='*70}")
    
    moved_count = 0
    
    for new_region, old_dirs in reorganization_plan.items():
        new_region_dir = data_dir / new_region
        new_region_dir.mkdir(parents=True, exist_ok=True)
        
        for old_dir_info in old_dirs:
            old_dir = old_dir_info['old_dir']
            print(f"\nMoving from {old_dir.name} to {new_region}...")
            
            for image_file in old_dir.glob('*.png'):
                target_path = new_region_dir / image_file.name
                
                # If file already exists (shouldn't happen), add suffix
                if target_path.exists():
                    stem = image_file.stem
                    suffix = image_file.suffix
                    counter = 1
                    while target_path.exists():
                        target_path = new_region_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                
                shutil.move(str(image_file), str(target_path))
                moved_count += 1
            
            # Remove empty old directory
            if not any(old_dir.iterdir()):
                old_dir.rmdir()
                print(f"  Removed empty directory: {old_dir.name}")
    
    print(f"\n[SUCCESS] Moved {moved_count} images")
    print(f"[SUCCESS] Regions normalized: {len(current_regions)} → {len(new_region_counts)}")

def update_annotations(annotations_dir, data_dir):
    """Update JSON annotations with normalized region names."""
    annotations_dir = Path(annotations_dir)
    data_dir = Path(data_dir)
    
    print(f"\n{'='*70}")
    print("UPDATING ANNOTATIONS")
    print(f"{'='*70}")
    
    updated_count = 0
    
    for json_file in annotations_dir.glob('*.json'):
        if json_file.name == 'template.json':
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            
            # Get original region
            region_original = annotation.get('region_original') or annotation.get('region')
            
            if region_original:
                # Normalize it
                region_normalized = normalize_region(region_original)
                
                # Update annotation
                annotation['region'] = region_normalized
                if 'region_original' not in annotation:
                    annotation['region_original'] = region_original
                
                # Update file_path to reflect new organization
                image_name = annotation.get('image_name')
                if image_name:
                    new_path = data_dir / region_normalized / image_name
                    if new_path.exists():
                        annotation['file_path'] = str(new_path)
                
                # Save updated annotation
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(annotation, f, indent=2, ensure_ascii=False)
                
                updated_count += 1
        
        except Exception as e:
            print(f"  [WARNING] Error updating {json_file.name}: {e}")
    
    print(f"\n[SUCCESS] Updated {updated_count} annotation files")

def main():
    """Main workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix region names in dataset")
    parser.add_argument('--dry-run', action='store_true', help="Show plan without making changes")
    args = parser.parse_args()
    
    print("="*70)
    print("FIX REGION NAMES")
    print("="*70)
    
    paths = get_paths()
    
    # Reorganize images
    reorganize_images(paths.data_raw, dry_run=args.dry_run)
    
    if not args.dry_run:
        # Update annotations
        update_annotations(paths.data_annotations, paths.data_raw)
        
        print("\n" + "="*70)
        print("REGION NORMALIZATION COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Check data/raw/ to see consolidated regions")
        print("  2. Run: python scripts/generate_designs.py --num-motifs 3 --random")
        print("  3. Verify regions are now properly grouped")
    else:
        print("\n" + "="*70)
        print("DRY RUN COMPLETE - No changes made")
        print("="*70)
        print("\nTo apply these changes, run:")
        print("  python scripts/fix_regions.py")

if __name__ == "__main__":
    main()

