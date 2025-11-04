"""Minimal diffusion-backed generation pipeline.

Current behavior:
- Builds a text prompt from inputs and generates an image with Stable Diffusion.
- Does not yet condition on the motif image; that will be added later.
"""

from typing import Optional
from pathlib import Path
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


class GenerationPipeline:
    """High-level orchestration for motif-to-fashion design generation."""

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Image-to-image pipeline to condition on the motif image
        self.img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )

        if self.device == "cuda":
            self.img2img = self.img2img.to(self.device)
        else:
            # CPU-friendly settings
            self.img2img.enable_attention_slicing()

    def generate(
        self,
        motif_image_path: str | Path,
        garment_type: str = "dress",
        adaptation_level: float = 0.5,
        color_strategy: str = "modernized",
        output_path: Optional[str | Path] = None,
    ) -> Image.Image:
        """Generate a design from a motif image.

        Args:
            motif_image_path: Path to motif image (PNG recommended).
            garment_type: Target garment type (e.g., "dress").
            adaptation_level: 0.0 (literal) to 1.0 (abstract).
            color_strategy: One of {"original", "modernized", "monochrome", "complementary", "seasonal"}.
            output_path: Optional path to save the generated image.

        Returns:
            A PIL Image representing the generated design.
        """
        prompt = self._build_prompt(garment_type, adaptation_level, color_strategy)

        # Load motif image and use as conditioning input (img2img)
        motif = Image.open(motif_image_path).convert("RGB")
        motif = motif.resize((512, 512), Image.BICUBIC)

        strength = self._strength_from_adaptation_level(adaptation_level)

        # Keep steps modest for speed; adjust as needed
        image = self.img2img(
            prompt=prompt,
            image=motif,
            strength=strength,
            guidance_scale=7.0,
            num_inference_steps=30,
            negative_prompt="low quality, blurry, distorted",
        ).images[0]

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)

        return image

    def _build_prompt(self, garment_type: str, adaptation_level: float, color_strategy: str) -> str:
        if adaptation_level < 0.3:
            style_str = "exact traditional Greek motif, authentic colors"
        elif adaptation_level < 0.7:
            style_str = "modernized Greek traditional motif, updated palette"
        else:
            style_str = "abstract interpretation inspired by Greek traditional motifs"

        color_str = {
            "original": "traditional color scheme",
            "modernized": "contemporary color palette",
            "monochrome": "monochromatic colors",
            "complementary": "color harmony, complementary scheme",
            "seasonal": "seasonal fashion colors",
        }.get(color_strategy, "contemporary color palette")

        base = f"high-quality {garment_type} featuring {style_str}, {color_str}, fashion editorial, elegant, wearable"
        return base

    def _strength_from_adaptation_level(self, level: float) -> float:
        """Map adaptation level (0..1) to img2img strength (0.2..0.85).

        Lower strength preserves more of the motif; higher strength allows more abstraction.
        """
        level = max(0.0, min(1.0, level))
        return 0.2 + 0.65 * level


