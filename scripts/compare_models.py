"""Compare different models and approaches systematically.

This script generates the same motif with different models/approaches
for direct comparison.
"""

import sys
from pathlib import Path
import json
import time
import io

# Set UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.generation.pipeline import GenerationPipeline
from src.generation.pipeline_sdxl import SDXLGenerationPipeline
from src.utils.config import get_paths


def compare_models_on_motif(motif_id, motif_path, output_dir):
    """Compare different models on the same motif."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test configurations
    configs = [
        {"adapt": 0.3, "garment": "dress", "desc": "literal_dress"},
        {"adapt": 0.5, "garment": "blouse", "desc": "moderate_blouse"},
    ]
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"COMPARING MODELS: Motif {motif_id}")
    print(f"{'='*70}")
    
    # Model 1: SD v1.5 (baseline)
    print("\n[1/2] Testing SD v1.5 (baseline)...")
    try:
        pipeline_v15 = GenerationPipeline()
        
        for config in configs:
            output_name = f"{motif_id}_v15_{config['desc']}.png"
            output_path = output_dir / output_name
            
            start_time = time.time()
            image = pipeline_v15.generate(
                motif_image_path=motif_path,
                garment_type=config['garment'],
                adaptation_level=config['adapt'],
                color_strategy="modernized",
                output_path=output_path,
            )
            gen_time = time.time() - start_time
            
            results.append({
                'model': 'SD_v1.5',
                'motif_id': motif_id,
                'config': config['desc'],
                'output': str(output_path),
                'time_seconds': gen_time,
                'resolution': image.size
            })
            
            print(f"  Generated: {output_name} ({gen_time:.1f}s)")
    
    except Exception as e:
        print(f"  [ERROR] SD v1.5 failed: {e}")
    
    # Model 2: SDXL
    print("\n[2/2] Testing SDXL...")
    try:
        pipeline_sdxl = SDXLGenerationPipeline()
        
        for config in configs:
            output_name = f"{motif_id}_sdxl_{config['desc']}.png"
            output_path = output_dir / output_name
            
            start_time = time.time()
            image = pipeline_sdxl.generate(
                motif_image_path=motif_path,
                garment_type=config['garment'],
                adaptation_level=config['adapt'],
                color_strategy="modernized",
                output_path=output_path,
            )
            gen_time = time.time() - start_time
            
            results.append({
                'model': 'SDXL',
                'motif_id': motif_id,
                'config': config['desc'],
                'output': str(output_path),
                'time_seconds': gen_time,
                'resolution': image.size
            })
            
            print(f"  Generated: {output_name} ({gen_time:.1f}s)")
    
    except Exception as e:
        print(f"  [ERROR] SDXL failed: {e}")
    
    return results


def main():
    """Main comparison workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare models on same motifs")
    parser.add_argument('--motif-ids', nargs='+', type=str, default=['19', '361', '467'],
                       help="Motif IDs to test (e.g., 19 361 467)")
    parser.add_argument('--output-dir', type=str, default='outputs/model_comparison',
                       help="Output directory")
    args = parser.parse_args()
    
    print("="*70)
    print("MODEL COMPARISON STUDY")
    print("="*70)
    print(f"\nMotifs to test: {', '.join(args.motif_ids)}")
    print(f"Output directory: {args.output_dir}")
    
    paths = get_paths()
    all_results = []
    
    for motif_id in args.motif_ids:
        # Find motif image
        image_name = f"image{motif_id}.png"
        motif_path = None
        
        for img_file in paths.data_raw.rglob(image_name):
            motif_path = img_file
            break
        
        if not motif_path or not motif_path.exists():
            print(f"\n[WARNING] Image not found: {image_name}")
            continue
        
        # Compare models
        results = compare_models_on_motif(
            f"MTF_{motif_id}",
            motif_path,
            args.output_dir
        )
        all_results.extend(results)
    
    # Save comparison results
    results_file = Path(args.output_dir) / "comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal designs generated: {len(all_results)}")
    print(f"Results saved to: {results_file}")
    print(f"\nGeneration times:")
    
    # Calculate average times per model
    v15_times = [r['time_seconds'] for r in all_results if r['model'] == 'SD_v1.5']
    sdxl_times = [r['time_seconds'] for r in all_results if r['model'] == 'SDXL']
    
    if v15_times:
        print(f"  SD v1.5:  {sum(v15_times)/len(v15_times):.1f}s average")
    if sdxl_times:
        print(f"  SDXL:     {sum(sdxl_times)/len(sdxl_times):.1f}s average")
    
    print(f"\nNext steps:")
    print(f"1. Review outputs in: {args.output_dir}")
    print(f"2. Fill in evaluation scores")
    print(f"3. Create comparison document")
    print("="*70)


if __name__ == "__main__":
    main()

