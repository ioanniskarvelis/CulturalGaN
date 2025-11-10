"""
Complete Phase 2: Symbolic Analysis & Semantic Embeddings
Runs both steps sequentially.

Usage:
    # Without vision API (free, basic quality)
    python scripts/run_phase2.py

    # With OpenAI vision (best quality, requires API key)
    python scripts/run_phase2.py --use-vision

    # Test run (10 images only)
    python scripts/run_phase2.py --use-vision --limit 10
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from src.data_processing.symbolic_analysis import SymbolicAnalyzer
from src.data_processing.create_embeddings import MotifEmbeddingCreator


def run_symbolic_analysis(args):
    """Step 1: Symbolic Analysis"""
    print("\n" + "="*70)
    print("PHASE 2 - STEP 1: SYMBOLIC ANALYSIS")
    print("="*70)
    
    # Check API key if using vision
    if args.use_vision:
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n⚠️  WARNING: No API key found!")
            print("   Set ANTHROPIC_API_KEY environment variable to use Claude vision analysis")
            print("   Running in fallback mode instead...\n")
            args.use_vision = False
    
    print(f"\n✓ Initializing Symbolic Analyzer")
    print(f"  Model: {args.model}")
    print(f"  Vision: {'✓ Enabled' if args.use_vision else '✗ Disabled (fallback mode)'}")
    print(f"  Limit: {args.limit if args.limit else 'None (process all)'}")
    
    analyzer = SymbolicAnalyzer(
        model=args.model,
        use_vision=args.use_vision
    )
    
    print(f"\n✓ Running symbolic analysis...")
    annotations_df = analyzer.process_dataset(
        metadata_path="data/processed/metadata.csv",
        output_path="data/annotations",
        limit=args.limit,
        skip_existing=True
    )
    
    print(f"\n✓ Generating summary report...")
    analyzer.generate_summary_report(
        annotations_path="data/annotations/annotations.json",
        output_path="data/annotations"
    )
    
    return True


def run_embedding_creation(args):
    """Step 2: Create Embeddings"""
    print("\n" + "="*70)
    print("PHASE 2 - STEP 2: CREATE SEMANTIC EMBEDDINGS")
    print("="*70)
    
    # Check if annotations exist
    if not os.path.exists("data/annotations/annotations.json"):
        print("\n❌ Error: Annotations not found!")
        print("   Symbolic analysis must have failed. Please check errors above.")
        return False
    
    print(f"\n✓ Initializing Embedding Creator")
    creator = MotifEmbeddingCreator()
    
    print(f"\n✓ Creating multi-modal embeddings...")
    stats = creator.process_dataset(
        annotations_path="data/annotations/annotations.json",
        output_path="data/embeddings",
        save_format="npz"
    )
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run complete Phase 2 pipeline")
    parser.add_argument(
        "--use-vision",
        action="store_true",
        help="Use vision-language model (requires API key)"
    )
    parser.add_argument(
        "--model",
        default="claude-3-5-sonnet-20241022",
        help="Model to use: claude-3-5-sonnet-20241022 (default), gpt-4o"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images (for testing)"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip symbolic analysis (if already done)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding creation (if already done)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("RUNNING COMPLETE PHASE 2 PIPELINE")
    print("="*70)
    print("\nThis will:")
    print("  1. Analyze cultural symbolism of motifs")
    print("  2. Create semantic embeddings")
    print()
    
    try:
        # Step 1: Symbolic Analysis
        if not args.skip_analysis:
            success = run_symbolic_analysis(args)
            if not success:
                return 1
        else:
            print("\n⏭️  Skipping symbolic analysis (--skip-analysis)")
        
        # Step 2: Embedding Creation
        if not args.skip_embeddings:
            success = run_embedding_creation(args)
            if not success:
                return 1
        else:
            print("\n⏭️  Skipping embedding creation (--skip-embeddings)")
        
        # Success!
        print("\n" + "="*70)
        print("✅ PHASE 2 COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  📁 data/annotations/")
        print("     - annotations.json (cultural analysis)")
        print("     - symbolic_analysis_report.json (summary)")
        print("  📁 data/embeddings/")
        print("     - embeddings.npz (multi-modal features)")
        print("     - embeddings_metadata.csv")
        print("     - embedding_stats.json")
        
        if args.limit:
            print(f"\n💡 You processed {args.limit} images as a test.")
            print(f"   To process all images, run without --limit")
        
        print("\n📖 For detailed information, see: PHASE2_GUIDE.md")
        
        print("\n⏭️  Next Step: Phase 3 - Train StyleGAN3")
        print("   python src/models/stylegan3_trainer.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

