"""Inspect the dataset.xlsx file to understand its structure.

This helps us figure out how to integrate it with the pipeline.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

def inspect_excel(filepath='dataset.xlsx'):
    """Inspect the Excel file structure."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        print(f"   Looking in: {filepath.absolute()}")
        return
    
    print("=" * 70)
    print(f"INSPECTING: {filepath.name}")
    print(f"Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 70)
    
    # Load Excel file - check sheet names first
    try:
        excel_file = pd.ExcelFile(filepath)
        print(f"\nSheet names: {excel_file.sheet_names}")
        print(f"   Number of sheets: {len(excel_file.sheet_names)}")
        
        # Load the first sheet (or main sheet)
        sheet_name = excel_file.sheet_names[0]
        print(f"\nLoading sheet: '{sheet_name}'")
        
        df = pd.read_excel(filepath, sheet_name=sheet_name, nrows=5)
        
        print(f"\nDataset shape (first 5 rows): {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        print("\nColumn data types:")
        for col in df.columns:
            print(f"   {col}: {df[col].dtype}")
        
        print("\nFirst few rows:")
        print(df.head())
        
        # Try to load full dataset to get count
        print("\nLoading full dataset to count rows...")
        df_full = pd.read_excel(filepath, sheet_name=sheet_name)
        print(f"[SUCCESS] Total rows: {len(df_full)}")
        
        # Look for image-related columns
        print("\nImage-related columns:")
        image_cols = [col for col in df_full.columns if any(keyword in col.lower() for keyword in ['image', 'file', 'path', 'img', 'photo'])]
        if image_cols:
            for col in image_cols:
                print(f"   - {col}")
                print(f"     Sample values: {df_full[col].head(3).tolist()}")
        else:
            print("   [WARNING] No obvious image-related columns found")
            print("   All columns:", list(df_full.columns))
        
        # Check for location/region columns
        print("\nLocation-related columns:")
        loc_cols = [col for col in df_full.columns if any(keyword in col.lower() for keyword in ['region', 'location', 'place', 'area', 'city'])]
        if loc_cols:
            for col in loc_cols:
                unique_vals = df_full[col].unique()
                print(f"   - {col}: {len(unique_vals)} unique values")
                print(f"     Examples: {list(unique_vals[:5])}")
        
        # Check for motif type columns
        print("\nType-related columns:")
        type_cols = [col for col in df_full.columns if any(keyword in col.lower() for keyword in ['type', 'category', 'class', 'kind', 'motif'])]
        if type_cols:
            for col in type_cols:
                unique_vals = df_full[col].unique()
                print(f"   - {col}: {len(unique_vals)} unique values")
                print(f"     Examples: {list(unique_vals[:5])}")
        
        return df_full
        
    except Exception as e:
        print(f"\n[ERROR] Error loading Excel file: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run inspection."""
    # Try different possible locations
    locations = [
        Path('dataset.xlsx'),
        Path('data/dataset.xlsx'),
        Path('data/raw/dataset.xlsx'),
    ]
    
    for loc in locations:
        if loc.exists():
            print(f"[SUCCESS] Found dataset at: {loc}")
            df = inspect_excel(loc)
            break
    else:
        print("[ERROR] dataset.xlsx not found in expected locations:")
        for loc in locations:
            print(f"   - {loc.absolute()}")

if __name__ == "__main__":
    main()

