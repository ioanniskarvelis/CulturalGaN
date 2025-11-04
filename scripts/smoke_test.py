import argparse
import sys
from pathlib import Path
from pathlib import Path as _Path

from PIL import Image, ImageDraw, ImageFont

# Ensure repository root is on sys.path so `src` can be imported
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.generation.pipeline import GenerationPipeline
from src.utils.config import get_paths


def _ensure_dummy_input(paths: object) -> Path:
    inputs_dir = paths.outputs_generated
    inputs_dir.mkdir(parents=True, exist_ok=True)
    dummy_path = inputs_dir / "smoke_input.png"
    if not dummy_path.exists():
        img = Image.new("RGB", (512, 512), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        text = "CulturalGaN Test"
        # Basic font (system default)
        draw.text((40, 240), text, fill=(30, 30, 30))
        img.save(dummy_path)
    return dummy_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for CulturalGaN generation pipeline")
    parser.add_argument("--input", type=str, default=None, help="Path to motif image (PNG preferred)")
    parser.add_argument("--output", type=str, default=None, help="Where to save the generated result")
    parser.add_argument("--garment", type=str, default="dress", help="Garment type, e.g., dress")
    parser.add_argument("--adapt", type=float, default=0.5, help="Adaptation level (0.0 literal → 1.0 abstract)")
    parser.add_argument("--color", type=str, default="modernized", help="Color strategy")
    args = parser.parse_args()

    paths = get_paths()

    motif_path = Path(args.input) if args.input else _ensure_dummy_input(paths)
    output_path = Path(args.output) if args.output else (paths.outputs_generated / "smoke_output.png")

    pipeline = GenerationPipeline()
    img = pipeline.generate(
        motif_image_path=motif_path,
        garment_type=args.garment,
        adaptation_level=args.adapt,
        color_strategy=args.color,
        output_path=output_path,
    )

    print(f"Input:  {motif_path}")
    print(f"Output: {output_path}")
    print(f"Size:   {img.size}")


if __name__ == "__main__":
    main()


