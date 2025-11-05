"""Quick setup verification script.

Run this to ensure your environment is properly configured.
"""

import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

def check_imports():
    """Verify all required packages can be imported."""
    print("Checking Python packages...")
    packages = [
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
    ]
    
    all_good = True
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - NOT FOUND")
            all_good = False
    
    return all_good

def check_directories():
    """Verify required directories exist."""
    print("\nChecking directory structure...")
    from src.utils.config import get_paths
    
    paths = get_paths()
    required_dirs = [
        ("Data (raw)", paths.data_raw),
        ("Data (processed)", paths.data_processed),
        ("Data (annotations)", paths.data_annotations),
        ("Outputs", paths.outputs_generated),
        ("Models", paths.models_checkpoints),
    ]
    
    all_good = True
    for name, path in required_dirs:
        if path.exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} - NOT FOUND")
            all_good = False
    
    return all_good

def check_cuda():
    """Check GPU availability."""
    print("\nChecking GPU/CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✓ GPU available: {gpu_name}")
            print(f"    VRAM: {memory:.1f} GB")
            return True
        else:
            print("  ⚠ No GPU detected - will use CPU (slower but works)")
            return True
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False

def check_pipeline():
    """Verify generation pipeline can be imported."""
    print("\nChecking generation pipeline...")
    try:
        from src.generation.pipeline import GenerationPipeline
        print("  ✓ Pipeline import successful")
        return True
    except Exception as e:
        print(f"  ✗ Pipeline import failed: {e}")
        return False

def check_dataset():
    """Check if any data has been collected."""
    print("\nChecking dataset...")
    from src.utils.config import get_paths
    
    paths = get_paths()
    
    # Count images
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_count = 0
    
    if paths.data_raw.exists():
        for ext in image_extensions:
            image_count += len(list(paths.data_raw.rglob(f'*{ext}')))
    
    # Count annotations
    annotation_count = 0
    if paths.data_annotations.exists():
        annotations = [f for f in paths.data_annotations.rglob('*.json') 
                      if f.name != 'template.json']
        annotation_count = len(annotations)
    
    if image_count == 0:
        print(f"  ⚠ No images found in {paths.data_raw}")
        print("    → Start by collecting Greek motif images!")
    else:
        print(f"  ✓ Found {image_count} images")
        if annotation_count > 0:
            print(f"  ✓ Found {annotation_count} annotations ({annotation_count/image_count*100:.1f}% coverage)")
        else:
            print(f"  ⚠ No annotations yet - use template at {paths.data_annotations / 'template.json'}")
    
    return True

def main():
    """Run all checks."""
    print("=" * 60)
    print("CulturalGaN Setup Verification")
    print("=" * 60)
    
    checks = [
        check_imports,
        check_directories,
        check_cuda,
        check_pipeline,
        check_dataset,
    ]
    
    results = [check() for check in checks]
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL CHECKS PASSED!")
        print("\nYou're ready to start. Next steps:")
        print("1. Collect Greek motif images → data/raw/")
        print("2. Run smoke test: python scripts/smoke_test.py")
        print("3. See QUICKSTART.md for detailed guide")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nPlease resolve issues above before proceeding.")
        print("If you need help, check QUICKSTART.md troubleshooting section.")
    print("=" * 60)

if __name__ == "__main__":
    main()

