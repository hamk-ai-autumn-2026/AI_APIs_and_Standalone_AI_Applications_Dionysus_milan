"""Image -> Text -> Image: a two-model CLI pipeline.

Stage 1: GPT-5.6 Luna examines the source image and writes a description.
Stage 2: GPT Image 2 receives ONLY that description and creates a new image.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

VISION_PROMPT = """Describe the supplied image as a detailed, self-contained prompt for
an image-generation model. Describe what is VISUALLY PRESENT, not what might be
outside the frame. Include:
- the main subject(s), their appearance, colors, materials, pose, and actions;
- the setting, background objects, and foreground details;
- relative positions, composition, viewpoint, and depth;
- lighting, shadows, color palette, mood, and artistic/photographic style;
- visible text, if any, accurately (do not invent unreadable text).
Write 150-250 words in a single coherent descriptive passage. Prioritize
observable details; do not add an introduction, bullet list, commentary,
or a claim that the output is copied from the image. If a detail is unclear,
do not guess. The passage must be useful on its own as a generation prompt."""

SUPPORTED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def encode_source_image(image_path: Path) -> str:
    """Validate a local image and return a Base64 data URL for vision analysis."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot read a valid image: {image_path}") from exc

    if image_format not in SUPPORTED_FORMATS:
        raise ValueError("Supported image formats: JPG/JPEG, PNG, WEBP")

    # A modest input limit prevents inadvertently uploading unusually large files.
    if image_path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("The input image must be 20 MiB or smaller")

    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    mime_type = SUPPORTED_FORMATS[image_format]
    return f"data:{mime_type};base64,{image_base64}"


def describe_image(client: OpenAI, source_data_url: str, model: str) -> str:
    """Stage 1. Send the actual input image to the vision model."""
    result = client.responses.create(
        model=model,
        reasoning={"effort": "none"},
        max_output_tokens=1200,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": VISION_PROMPT},
                    {"type": "input_image", "image_url": source_data_url, "detail": "high"},
                ],
            }
        ],
    )
    description = (result.output_text or "").strip()
    if not description:
        raise RuntimeError("The vision model returned an empty description")
    return description


def generate_from_description(
    client: OpenAI, description: str, model: str, size: str, quality: str
) -> bytes:
    """Stage 2. Generate from TEXT ONLY: no source image is passed here."""
    result = client.images.generate(
        model=model,
        prompt=description,
        size=size,
        quality=quality,
        output_format="png",
        n=1,
    )
    if not result.data or not result.data[0].b64_json:
        raise RuntimeError("The image model returned no image data")
    return base64.b64decode(result.data[0].b64_json, validate=True)


def run_pipeline(
    client: OpenAI,
    input_path: Path,
    output_path: Path,
    description_path: Path,
    vision_model: str,
    image_model: str,
    size: str,
    quality: str,
) -> str:
    """Execute both model calls; print text and save both outputs locally."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    description_path = description_path.expanduser().resolve()

    if output_path.suffix.lower() != ".png":
        raise ValueError("--output must be a .png file")
    if len({input_path, output_path, description_path}) != 3:
        raise ValueError("Input image, output image and description must have different paths")

    source_data_url = encode_source_image(input_path)
    print(f"[1/3] Describing image with {vision_model}...", flush=True)
    description = describe_image(client, source_data_url, vision_model)

    print("\n========== GENERATED DESCRIPTION ==========\n")
    print(description)
    print("\n===========================================\n", flush=True)

    description_path.parent.mkdir(parents=True, exist_ok=True)
    description_path.write_text(description + "\n", encoding="utf-8")
    print(f"Description saved to: {description_path}", flush=True)

    print(f"[2/3] Generating new image with {image_model}...", flush=True)
    image_bytes = generate_from_description(
        client, description, image_model, size, quality
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    print(f"[3/3] Generated image saved to: {output_path}", flush=True)
    return description


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Image -> Text -> Image using OpenAI vision and image-generation APIs"
    )
    parser.add_argument("input", type=Path, help="Path to source image (.jpg/.png/.webp)")
    parser.add_argument("--output", type=Path, default=Path("output/generated.png"),
                        help="Where to save the generated PNG")
    parser.add_argument("--description-output", type=Path,
                        default=Path("output/description.txt"),
                        help="Where to save the generated description")
    parser.add_argument("--vision-model", default="gpt-5.6-luna",
                        help="Vision-capable model (default: gpt-5.6-luna)")
    parser.add_argument("--image-model", default="gpt-image-2",
                        help="Text-to-image model (default: gpt-image-2)")
    parser.add_argument("--size", choices=["1024x1024", "1536x1024", "1024x1536"],
                        default="1024x1024", help="Output image size")
    parser.add_argument("--quality", choices=["low", "medium", "high"],
                        default="medium", help="Output quality (and cost)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("put-your-own-"):
        print("ERROR: Set OPENAI_API_KEY in your environment or a .env file.",
              file=sys.stderr)
        return 1

    try:
        from openai import OpenAI, OpenAIError
    except ImportError:
        print("ERROR: OpenAI SDK is not installed. Run: pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    try:
        client = OpenAI(timeout=180.0, max_retries=2)
        run_pipeline(
            client=client,
            input_path=args.input,
            output_path=args.output,
            description_path=args.description_output,
            vision_model=args.vision_model,
            image_model=args.image_model,
            size=args.size,
            quality=args.quality,
        )
        return 0
    except (OpenAIError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
