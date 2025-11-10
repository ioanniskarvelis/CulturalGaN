"""
Quick script to run symbolic analysis on processed motifs.

Usage:
    # Without vision (fallback mode - no API key needed)
    python scripts/run_symbolic_analysis.py

    # With OpenAI vision (requires API key)
    python scripts/run_symbolic_analysis.py --use-vision --limit 10

    # With Anthropic Claude (requires API key)
    python scripts/run_symbolic_analysis.py --use-vision --model claude-3-5-sonnet-20241022 --limit 10
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.symbolic_analysis import SymbolicAnalyzer
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run symbolic analysis on Greek motifs")
    parser.add_argument(
        "--use-vision",
        action="store_true",
        help="Use vision-language model (requires API key)"
    )
    parser.add_argument(
        "--model",
        default="claude-3-5-sonnet-20241022",
        help="Model to use: claude-3-5-sonnet-20241022 (default), gpt-4o, gpt-4o-mini"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (useful for testing)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE 2: SYMBOLIC ANALYSIS OF GREEK MOTIFS")
    print("=" * 70)
    
    # Check API key if using vision
    if args.use_vision:
        api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n⚠️  WARNING: No API key found!")
            print("   For vision analysis, you need to set:")
            print("   - ANTHROPIC_API_KEY environment variable (for Claude - recommended)")
            print("   - or OPENAI_API_KEY environment variable (for GPT models)")
            print("\n   Running in fallback mode instead...\n")
            args.use_vision = False
    
    # Initialize analyzer
    print(f"\n✓ Initializing Symbolic Analyzer")
    print(f"  Model: {args.model}")
    print(f"  Vision: {'✓ Enabled' if args.use_vision else '✗ Disabled (fallback mode)'}")
    print(f"  Limit: {args.limit if args.limit else 'None (process all)'}")
    
    analyzer = SymbolicAnalyzer(
        api_key=args.api_key,
        model=args.model,
        use_vision=args.use_vision
    )
    
    # Process dataset
    print(f"\n✓ Starting symbolic analysis...")
    print(f"  This will analyze cultural meanings and symbolism")
    print(f"  Progress will be saved every 10 images")
    print()
    
    try:
        annotations_df = analyzer.process_dataset(
            metadata_path="data/processed/metadata.csv",
            output_path="data/annotations",
            limit=args.limit,
            skip_existing=True
        )
        
        # Generate summary report
        print(f"\n✓ Generating summary report...")
        report = analyzer.generate_summary_report(
            annotations_path="data/annotations/annotations.json",
            output_path="data/annotations"
        )
        
        print("\n" + "=" * 70)
        print("✓ SYMBOLIC ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"\nResults saved to:")
        print(f"  - Annotations: data/annotations/annotations.json")
        print(f"  - Report: data/annotations/symbolic_analysis_report.json")
        
        if args.limit:
            print(f"\n💡 You processed {args.limit} images as a test.")
            print(f"   To process all images, run without --limit flag")
        
        if not args.use_vision:
            print(f"\n💡 You used fallback mode (no vision API).")
            print(f"   For better results, use --use-vision with an API key")
        
        print(f"\nNext step: Create semantic embeddings")
        print(f"  python scripts/create_embeddings.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

