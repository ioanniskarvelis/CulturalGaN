"""Quick script to show motif details."""
import json
import sys
from pathlib import Path

motif_ids = sys.argv[1:] if len(sys.argv) > 1 else ['19', '361', '467']

for mid in motif_ids:
    ann_file = Path(f'data/annotations/image{mid}.json')
    if ann_file.exists():
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"MOTIF MTF_{mid}")
        print(f"{'='*60}")
        print(f"Name: {data.get('name', 'N/A')}")
        print(f"Region: {data.get('region', 'N/A')}")
        print(f"Original Region: {data.get('region_original', 'N/A')}")
        print(f"Type: {data.get('type', 'N/A')}")
        print(f"Period: {data.get('period', 'N/A')}")
        print(f"Craft: {data.get('original_medium', 'N/A')}")
        print(f"Style: {data.get('style', 'N/A')}")
        if data.get('cultural_significance'):
            print(f"\nCultural Significance:")
            print(f"  {data['cultural_significance'][:200]}...")
        if data.get('symbolic_meaning'):
            print(f"\nSymbolic Meaning:")
            print(f"  {data['symbolic_meaning'][:200]}...")

