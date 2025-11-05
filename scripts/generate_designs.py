"""Generate fashion designs from Greek motifs using annotations.

This script demonstrates the full pipeline with your integrated dataset.
"""

import sys
from pathlib import Path
import json
import argparse
import random
import io

# Set UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.generation.pipeline import GenerationPipeline
from src.utils.config import get_paths

def load_annotation(json_path):
    """Load annotation JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_all_motifs(annotations_dir):
    """Get list of all annotated motifs."""
    annotations_dir = Path(annotations_dir)
    motifs = []
    
    for json_file in annotations_dir.glob('*.json'):
        if json_file.name == 'template.json':
            continue
        try:
            annotation = load_annotation(json_file)
            motifs.append(annotation)
        except Exception as e:
            print(f"[WARNING] Error loading {json_file.name}: {e}")
    
    return motifs

def display_motif_info(annotation):
    """Display information about a motif."""
    print("\n" + "="*70)
    print(f"MOTIF: {annotation['motif_id']}")
    print("="*70)
    print(f"Name: {annotation.get('name', 'N/A')}")
    print(f"Region: {annotation.get('region', 'N/A')}")
    print(f"Type: {annotation.get('type', 'N/A')}")
    print(f"Period: {annotation.get('period', 'N/A')}")
    print(f"Craft: {annotation.get('original_medium', 'N/A')}")
    
    if annotation.get('cultural_significance'):
        print(f"\nCultural Significance:")
        print(f"  {annotation['cultural_significance'][:200]}...")
    
    if annotation.get('symbolic_meaning'):
        print(f"\nSymbolic Meaning:")
        print(f"  {annotation['symbolic_meaning'][:200]}...")
    
    print("="*70)

def generate_from_motif(motif_annotation, output_dir, pipeline, num_variations=3):
    """Generate fashion designs from a single motif."""
    motif_id = motif_annotation['motif_id']
    image_name = motif_annotation['image_name']
    
    # Find the image file
    paths = get_paths()
    image_path = None
    
    # Search in data/raw subdirectories
    for img_file in paths.data_raw.rglob(image_name):
        image_path = img_file
        break
    
    if not image_path or not image_path.exists():
        print(f"[ERROR] Image not found: {image_name}")
        return []
    
    print(f"\nGenerating designs from: {image_path.name}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Different adaptation strategies
    configs = [
        {"adapt": 0.3, "garment": "dress", "color": "original", "desc": "literal_dress"},
        {"adapt": 0.5, "garment": "blouse", "color": "modernized", "desc": "moderate_blouse"},
        {"adapt": 0.7, "garment": "scarf", "color": "monochrome", "desc": "abstract_scarf"},
    ]
    
    generated = []
    
    for i, config in enumerate(configs[:num_variations]):
        output_name = f"{motif_id}_{config['desc']}.png"
        output_path = output_dir / output_name
        
        print(f"  [{i+1}/{num_variations}] {config['garment']} (adapt={config['adapt']}, color={config['color']})")
        
        try:
            image = pipeline.generate(
                motif_image_path=image_path,
                garment_type=config['garment'],
                adaptation_level=config['adapt'],
                color_strategy=config['color'],
                output_path=output_path,
            )
            generated.append({
                'motif_id': motif_id,
                'output_path': str(output_path),
                'config': config,
                'annotation': motif_annotation
            })
            print(f"     Saved: {output_path.name}")
        except Exception as e:
            print(f"     [ERROR] Generation failed: {e}")
    
    return generated

def main():
    """Main generation workflow."""
    parser = argparse.ArgumentParser(description="Generate fashion designs from Greek motifs")
    parser.add_argument('--motif-id', type=str, help="Specific motif ID to generate from")
    parser.add_argument('--region', type=str, help="Filter by region (e.g., 'Thessaly', 'Rhodes')")
    parser.add_argument('--type', type=str, help="Filter by type")
    parser.add_argument('--num-motifs', type=int, default=5, help="Number of motifs to process")
    parser.add_argument('--variations', type=int, default=3, help="Variations per motif")
    parser.add_argument('--random', action='store_true', help="Random selection")
    parser.add_argument('--output-dir', type=str, default='outputs/generated_designs', help="Output directory")
    args = parser.parse_args()
    
    print("="*70)
    print("GREEK MOTIF TO FASHION DESIGN GENERATION")
    print("="*70)
    
    # Get paths and load motifs
    paths = get_paths()
    print("\nLoading motif annotations...")
    all_motifs = get_all_motifs(paths.data_annotations)
    print(f"[SUCCESS] Loaded {len(all_motifs)} motifs")
    
    # Filter motifs
    motifs = all_motifs
    
    if args.motif_id:
        motifs = [m for m in motifs if m['motif_id'] == args.motif_id]
        if not motifs:
            print(f"[ERROR] Motif ID '{args.motif_id}' not found")
            return
    
    if args.region:
        motifs = [m for m in motifs if args.region.lower() in m.get('region', '').lower()]
        print(f"Filtered by region '{args.region}': {len(motifs)} motifs")
    
    if args.type:
        motifs = [m for m in motifs if args.type.lower() in m.get('type', '').lower()]
        print(f"Filtered by type '{args.type}': {len(motifs)} motifs")
    
    if not motifs:
        print("[ERROR] No motifs match the filters")
        return
    
    # Select motifs to process
    if args.random:
        motifs = random.sample(motifs, min(args.num_motifs, len(motifs)))
    else:
        motifs = motifs[:args.num_motifs]
    
    print(f"\nProcessing {len(motifs)} motifs...")
    
    # Initialize pipeline
    print("\nInitializing generation pipeline...")
    print("[INFO] First run will download Stable Diffusion (~4GB)")
    pipeline = GenerationPipeline()
    print("[SUCCESS] Pipeline ready")
    
    # Generate designs
    all_generated = []
    
    for idx, motif in enumerate(motifs):
        print(f"\n{'='*70}")
        print(f"MOTIF {idx+1}/{len(motifs)}")
        display_motif_info(motif)
        
        generated = generate_from_motif(
            motif, 
            args.output_dir, 
            pipeline, 
            num_variations=args.variations
        )
        all_generated.extend(generated)
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print(f"\nTotal designs generated: {len(all_generated)}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nGenerated designs:")
    for gen in all_generated[:10]:  # Show first 10
        print(f"  - {Path(gen['output_path']).name}")
    if len(all_generated) > 10:
        print(f"  ... and {len(all_generated) - 10} more")
    print("="*70)

if __name__ == "__main__":
    main()

