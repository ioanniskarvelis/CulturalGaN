"""
Quick script to create semantic embeddings for Greek motifs.

Usage:
    python scripts/create_embeddings.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.create_embeddings import MotifEmbeddingCreator
import argparse


def main():
    parser = argparse.ArgumentParser(description="Create semantic embeddings")
    parser.add_argument(
        "--annotations",
        default="data/annotations/annotations.json",
        help="Path to annotations file"
    )
    parser.add_argument(
        "--output",
        default="data/embeddings",
        help="Output directory"
    )
    parser.add_argument(
        "--format",
        default="npz",
        choices=["npz", "pt"],
        help="Save format"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE 2: CREATING SEMANTIC EMBEDDINGS")
    print("=" * 70)
    
    # Check if annotations exist
    if not os.path.exists(args.annotations):
        print(f"\n❌ Error: Annotations file not found: {args.annotations}")
        print(f"\nYou need to run symbolic analysis first:")
        print(f"  python scripts/run_symbolic_analysis.py")
        return 1
    
    # Create embeddings
    print(f"\n✓ Initializing Embedding Creator")
    print(f"  This will create multi-modal embeddings combining:")
    print(f"  - Visual features (CLIP)")
    print(f"  - Textual features (symbolic descriptions)")
    print(f"  - Geometric features")
    print()
    
    try:
        creator = MotifEmbeddingCreator()
        
        stats = creator.process_dataset(
            annotations_path=args.annotations,
            output_path=args.output,
            save_format=args.format
        )
        
        print("\n" + "=" * 70)
        print("✓ EMBEDDING CREATION COMPLETE!")
        print("=" * 70)
        print(f"\nEmbeddings saved to: {args.output}/")
        print(f"  - embeddings.{args.format}")
        print(f"  - embeddings_metadata.csv")
        print(f"  - embedding_stats.json")
        
        print(f"\nNext step: Train StyleGAN3")
        print(f"  python src/models/stylegan3_trainer.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

