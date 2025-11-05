"""Enhanced pipeline using Stable Diffusion XL for better quality.

SDXL provides:
- Higher resolution (1024x1024 native)
- Better image quality
- Improved prompt understanding
"""

from typing import Optional
from pathlib import Path
from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline


class SDXLGenerationPipeline:
    """High-quality generation with Stable Diffusion XL."""

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"Loading SDXL model... (this may take a few minutes)")
        print(f"Device: {self.device}, Dtype: {dtype}")

        # SDXL image-to-image pipeline
        self.img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        if self.device == "cuda":
            self.img2img = self.img2img.to(self.device)
        else:
            # CPU optimizations
            self.img2img.enable_attention_slicing()
            self.img2img.enable_vae_slicing()

        print(f"SDXL pipeline loaded successfully!")

    def generate(
        self,
        motif_image_path: str | Path,
        garment_type: str = "dress",
        adaptation_level: float = 0.5,
        color_strategy: str = "modernized",
        output_path: Optional[str | Path] = None,
    ) -> Image.Image:
        """Generate a design from a motif image using SDXL.

        Args:
            motif_image_path: Path to motif image.
            garment_type: Target garment type.
            adaptation_level: 0.0 (literal) to 1.0 (abstract).
            color_strategy: Color adaptation strategy.
            output_path: Optional path to save output.

        Returns:
            Generated fashion design image.
        """
        prompt = self._build_prompt(garment_type, adaptation_level, color_strategy)

        # Load and resize motif - SDXL native is 1024x1024
        motif = Image.open(motif_image_path).convert("RGB")
        motif = motif.resize((1024, 1024), Image.BICUBIC)

        strength = self._strength_from_adaptation_level(adaptation_level)

        # SDXL generation
        print(f"Generating with SDXL (strength={strength:.2f})...")
        image = self.img2img(
            prompt=prompt,
            image=motif,
            strength=strength,
            guidance_scale=7.5,  # SDXL works well with 7-8
            num_inference_steps=30,  # SDXL is efficient
            negative_prompt="low quality, blurry, distorted, deformed, ugly, bad anatomy",
        ).images[0]

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            print(f"Saved to: {output_path}")

        return image

    def _build_prompt(self, garment_type: str, adaptation_level: float, color_strategy: str) -> str:
        """Build detailed prompt for SDXL (benefits from longer prompts)."""
        
        # SDXL handles longer, more detailed prompts better
        if adaptation_level < 0.3:
            style_str = "exact traditional Greek motif pattern, authentic historical colors, precise cultural details"
        elif adaptation_level < 0.7:
            style_str = "modernized Greek traditional motif, contemporary updated palette, balanced cultural adaptation"
        else:
            style_str = "abstract artistic interpretation inspired by Greek traditional motifs, creative modern aesthetic"

        color_str = {
            "original": "traditional authentic color scheme, historical palette",
            "modernized": "contemporary fashion color palette, modern updated colors",
            "monochrome": "elegant monochromatic colors, sophisticated single-color scheme",
            "complementary": "harmonious color harmony, complementary color scheme",
            "seasonal": "seasonal fashion colors, trendy current palette",
        }.get(color_strategy, "contemporary color palette")

        # SDXL prompt (more detailed than v1.5)
        prompt = (
            f"professional high-quality fashion photography, "
            f"elegant {garment_type} featuring {style_str}, "
            f"{color_str}, "
            f"fashion editorial style, studio lighting, clean background, "
            f"wearable design, refined aesthetic, high resolution, detailed"
        )
        
        return prompt

    def _strength_from_adaptation_level(self, level: float) -> float:
        """Map adaptation level to img2img strength.
        
        SDXL is more powerful, so we can use slightly different ranges.
        """
        level = max(0.0, min(1.0, level))
        # SDXL: 0.15 to 0.75 (slightly lower than v1.5 for similar effect)
        return 0.15 + 0.60 * level

