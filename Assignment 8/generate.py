#!/usr/bin/env python3
"""Exercise #8: image generation CLI supporting OpenAI, fal and Replicate.

Usage:
    python generate.py --prompt "A red cottage beside a Finnish lake" \
        --aspect-ratio 16:9 --count 3 --output images

Only the selected provider receives your prompt. See README.md for the
provider-specific capabilities and configuration requirements.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import os
from pathlib import Path
import re
import sys
import unicodedata
from urllib.parse import urlsplit
from uuid import uuid4

# 16-pixel aligned dimensions supported by GPT Image 2; these also give
# fal custom image sizes. Replicate receives the aspect-ratio labels directly.
SIZES: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1536, 864),
    "9:16": (864, 1536),
    "4:3": (1280, 960),
    "3:4": (960, 1280),
    "3:2": (1536, 1024),
    "2:3": (1024, 1536),
}

PROVIDER_MODELS = {
    "openai": "gpt-image-2",
    "fal": "fal-ai/flux-general",
    "replicate": "black-forest-labs/flux-schnell",
}

# 10 images max per invocation. This guard prevents accidentally running a
# costly, unbounded generation, and each image is requested independently.
MAX_COUNT = 10
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_SEED = 2**32 - 1


class GenerationError(RuntimeError):
    """A readable generation or image-download error."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and locally save images with OpenAI, fal, or Replicate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", required=True, help="Text describing the image to make")
    parser.add_argument("--aspect-ratio", choices=tuple(SIZES), default="1:1")
    parser.add_argument("--count", type=int, default=1, help=f"Images to generate (1-{MAX_COUNT})")
    parser.add_argument("--output", type=Path, default=Path("images"), help="Output directory")
    parser.add_argument("--provider", choices=tuple(PROVIDER_MODELS), default="openai")
    parser.add_argument("--negative-prompt", help="fal FLUX only: features to avoid")
    parser.add_argument("--seed", type=int, help="fal or Replicate only: nonnegative random seed")
    parser.add_argument("--quality", choices=("low", "medium", "high"),
                        help="OpenAI only: model rendering quality")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.prompt.strip():
        parser.error("--prompt must not be empty or whitespace")
    if not 1 <= args.count <= MAX_COUNT:
        parser.error(f"--count must be between 1 and {MAX_COUNT}")
    if args.negative_prompt is not None:
        if not args.negative_prompt.strip():
            parser.error("--negative-prompt cannot be empty")
        if args.provider != "fal":
            parser.error("--negative-prompt is supported only by --provider fal")
    if args.seed is not None:
        if args.provider not in ("fal", "replicate"):
            parser.error("--seed is not supported by --provider openai")
        if args.seed < 0 or args.seed + args.count - 1 > MAX_SEED:
            parser.error(f"--seed must be between 0 and {MAX_SEED - args.count + 1} for this count")
    if args.quality is not None and args.provider != "openai":
        parser.error("--quality is supported only by --provider openai")
    if args.output.exists() and not args.output.is_dir():
        parser.error(f"--output must be a directory, not a file: {args.output}")


def check_key(provider: str) -> None:
    env_var = {"openai": "OPENAI_API_KEY", "fal": "FAL_KEY",
               "replicate": "REPLICATE_API_TOKEN"}[provider]
    if not os.getenv(env_var):
        raise GenerationError(
            f"Missing {env_var}. Set it as an environment variable or add it to a .env file."
        )


def slugify(prompt: str, max_length: int = 48) -> str:
    """Make a short, ASCII-only filename component; never use raw prompt paths."""
    ascii_text = unicodedata.normalize("NFKD", prompt).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "image"


def decode_base64_image(data: str) -> bytes:
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise GenerationError("Provider returned invalid Base64 image data") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise GenerationError("Image exceeds the configured 30 MB limit")
    return raw


def download_url(url: str) -> bytes:
    """Download temporary HTTP(S) image URLs immediately; never save just URLs."""
    import httpx

    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.netloc:
        raise GenerationError("Provider returned an unsafe or invalid image URL (HTTPS required)")
    try:
        content = bytearray()
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > MAX_IMAGE_BYTES:
                    raise GenerationError("Downloaded image exceeds the 30 MB limit")
        return bytes(content)
    except httpx.TimeoutException as exc:
        raise GenerationError("Timed out downloading the provider's image") from exc
    except httpx.HTTPError as exc:
        raise GenerationError(f"Image download failed: {exc}") from exc


def image_to_bytes(item: object) -> bytes:
    """Accept raw bytes, Base64 data URIs, temporary URLs and file objects."""
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    if isinstance(item, str):
        if item.startswith("data:"):
            header, sep, payload = item.partition(",")
            if not sep or ";base64" not in header.lower():
                raise GenerationError("Unsupported image data URI")
            return decode_base64_image(payload)
        return download_url(item)
    # Replicate's Python SDK returns FileOutput with a .read() method.
    read = getattr(item, "read", None)
    if callable(read):
        return image_to_bytes(read())
    raise GenerationError(f"Unsupported image response type: {type(item).__name__}")


def validate_image(raw: bytes) -> str:
    """Verify actual image data and select the extension from its file format."""
    from PIL import Image, UnidentifiedImageError

    if not raw:
        raise GenerationError("Provider returned an empty image")
    if len(raw) > MAX_IMAGE_BYTES:
        raise GenerationError("Image exceeds the 30 MB limit")
    try:
        with Image.open(io.BytesIO(raw)) as img:
            actual_format = img.format
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise GenerationError("Provider response is not a valid image file") from exc
    extensions = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
    if actual_format not in extensions:
        raise GenerationError(f"Unsupported image format returned: {actual_format}")
    return extensions[actual_format]


def save_image(raw: bytes, output: Path, prompt: str, provider: str, index: int) -> Path:
    """Save verified bytes without allowing a duplicate filename to overwrite anything."""
    ext = validate_image(raw)
    prefix = slugify(prompt)
    for _ in range(5):
        filename = f"{prefix}_{provider}_{index:02d}_{uuid4().hex[:12]}{ext}"
        target = output / filename
        try:
            with target.open("xb") as file:  # exclusive creation; never overwrite
                file.write(raw)
            return target
        except FileExistsError:
            continue
        except OSError:
            target.unlink(missing_ok=True)
            raise
    raise GenerationError("Could not create a unique filename after multiple attempts")


def generate_openai(args: argparse.Namespace, seed: int | None) -> bytes:
    """OpenAI Images API: GPT image models return Base64, not permanent URLs."""
    from openai import OpenAI

    width, height = SIZES[args.aspect_ratio]
    parameters = {
        "model": PROVIDER_MODELS["openai"],
        "prompt": args.prompt,
        "size": f"{width}x{height}",
        "n": 1,
        "output_format": "png",
    }
    if args.quality is not None:
        parameters["quality"] = args.quality
    # The SDK handles transient retryable API responses. Longer timeout is
    # deliberate because generating one image can require a couple of minutes.
    client = OpenAI(max_retries=2, timeout=180.0)
    response = client.images.generate(**parameters)
    images = response.data or []
    if len(images) != 1 or not images[0].b64_json:
        raise GenerationError("OpenAI returned no Base64 image for this request")
    return decode_base64_image(images[0].b64_json)


def generate_fal(args: argparse.Namespace, seed: int | None) -> bytes:
    """fal FLUX General API: returned image URL is fetched immediately."""
    import fal_client

    width, height = SIZES[args.aspect_ratio]
    parameters = {
        "prompt": args.prompt,
        "image_size": {"width": width, "height": height},
        "num_images": 1,
        "output_format": "png",
    }
    if args.negative_prompt is not None:
        parameters["negative_prompt"] = args.negative_prompt
    if seed is not None:
        parameters["seed"] = seed
    result = fal_client.subscribe(PROVIDER_MODELS["fal"], arguments=parameters)
    images = result.get("images", []) if isinstance(result, dict) else []
    if len(images) != 1 or not images[0].get("url"):
        raise GenerationError("fal returned no usable image URL")
    return image_to_bytes(images[0]["url"])


def generate_replicate(args: argparse.Namespace, seed: int | None) -> bytes:
    """Replicate FLUX Schnell API: output may be a FileOutput or temporary URL."""
    import replicate

    parameters = {
        "prompt": args.prompt,
        "aspect_ratio": args.aspect_ratio,
        "num_outputs": 1,
        "output_format": "webp",
    }
    if seed is not None:
        parameters["seed"] = seed
    result = replicate.run(PROVIDER_MODELS["replicate"], input=parameters)
    images = result if isinstance(result, (list, tuple)) else [result]
    if len(images) != 1 or images[0] is None:
        raise GenerationError("Replicate returned no usable image")
    return image_to_bytes(images[0])


def generate_image(args: argparse.Namespace, seed: int | None) -> bytes:
    generators = {
        "openai": generate_openai,
        "fal": generate_fal,
        "replicate": generate_replicate,
    }
    return generators[args.provider](args, seed)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    # The dotenv package is not required to use --help or validate arguments.
    try:
        from dotenv import load_dotenv
        load_dotenv()
        check_key(args.provider)
        args.output.mkdir(parents=True, exist_ok=True)
    except (OSError, GenerationError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Provider: {args.provider} ({PROVIDER_MODELS[args.provider]})", file=sys.stderr)
    print(f"Generating {args.count} image(s) at aspect ratio {args.aspect_ratio}...", file=sys.stderr)
    saved: list[Path] = []
    for index in range(1, args.count + 1):
        try:
            # Change the seed for each image to avoid identical outputs.
            seed = args.seed + index - 1 if args.seed is not None else None
            raw = generate_image(args, seed)
            path = save_image(raw, args.output, args.prompt, args.provider, index)
            saved.append(path)
            print(f"Saved {index}/{args.count}: {path}")
        except KeyboardInterrupt:
            print("\nInterrupted. Already saved images are kept.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"Error generating image {index}/{args.count}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            if saved:
                print(f"Already saved {len(saved)} image(s) in {args.output}.", file=sys.stderr)
            print("Check your API key, available credits, connectivity, and selected options.",
                  file=sys.stderr)
            return 1

    print(f"Success: saved all {len(saved)} image(s) in {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
