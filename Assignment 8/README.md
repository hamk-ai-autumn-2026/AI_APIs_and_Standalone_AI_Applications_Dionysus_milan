# Exercise #8 — Image Generator CLI

**Course:** AI APIs and Standalone AI Applications — ICT Robotics 2026 (IR00EU78-3002)

A Python command-line utility that generates images using **OpenAI**, **fal**, or **Replicate**, validates user input, saves all image bytes locally, and reports every saved filename.

## Project files

- `generate.py` — main, ready-to-run command-line program
- `requirements.txt` — required Python libraries
- `.env.example` — example key configuration (no real credentials)
- `tests/test_generate.py` — offline automated tests (no API charges)


## Setup

Use **Python 3.10 or newer** and a terminal in this folder. Create a virtual environment (optional but recommended):

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then add your **real key only for the provider you plan to use**. For example, for OpenAI:

```dotenv
OPENAI_API_KEY=sk-your-own-api-key
```

Get keys from the relevant provider's account dashboard. **API requests are billed by the provider**; a ChatGPT subscription does not automatically provide API credits. Never commit or submit `.env`.

## Example commands

**Basic example, OpenAI (default provider):**

```bash
python generate.py --prompt "A red wooden cottage beside a Finnish lake" --aspect-ratio 16:9 --count 3 --output images
```

**OpenAI image quality:**

```bash
python generate.py --provider openai --prompt "Agricultural robot working in a Finnish greenhouse" --aspect-ratio 4:3 --quality low --count 2 --output results
```

**fal FLUX General: negative prompt and seed:**

```bash
python generate.py --provider fal --prompt "A futuristic robot harvesting strawberries" --negative-prompt "blur, watermark, text" --seed 42 --aspect-ratio 16:9 --count 2 --output images
```

**Replicate FLUX Schnell: reproducible seed:**

```bash
python generate.py --provider replicate --prompt "A robotic arm picking apples" --seed 123 --aspect-ratio 1:1 --count 3 --output images
```

View options:

```bash
python generate.py --help
```

## Options and provider support

| CLI parameter | OpenAI (`gpt-image-2`) | fal (`fal-ai/flux-general`) | Replicate (`black-forest-labs/flux-schnell`) |
| --- | --- | --- | --- |
| `--prompt` | Yes | Yes | Yes |
| `--aspect-ratio` | Yes | Yes | Yes |
| `--count` (1–10) | Yes | Yes | Yes |
| `--output` | Yes | Yes | Yes |
| `--negative-prompt` | Not supported; error | Yes | Not supported; error |
| `--seed` | Not supported; error | Yes | Yes |
| `--quality` (`low`,`medium`,`high`) | Yes | Not supported; error | Not supported; error |

Supported ratios: `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3`.

The program generates **one image per provider call** so that successful images are saved immediately. For `--count 3 --seed 42`, seeds 42, 43 and 44 are used (for seed-supporting providers). Output filenames are derived from a safe prompt slug, provider name, index, and random UUID fragment; an existing file is **never overwritten**.

`--quality` refers to OpenAI's model **rendering quality**; Replicate's `output_quality` is an *encoding/compression* setting and therefore is not exposed under that CLI flag.

## How files are handled

- **OpenAI:** the SDK returns Base64 data, which the program decodes to bytes.
- **fal:** the SDK returns an image URL, which the program immediately downloads. It can also handle a Base64 data URI.
- **Replicate:** the SDK typically returns a `FileOutput` object (read as bytes); URL fallback is also handled.

Downloaded image URLs are assumed to be **temporary**. The image is fully downloaded and verified using Pillow before the CLI prints the saved local path. It can detect PNG, JPEG, and WebP formats and chooses the actual file extension. It restricts downloads to HTTPS and sets a 30 MB maximum image size.

## Tests (no paid API calls)

```bash
python -m unittest discover -s tests -v
```

Tests use mocks to check argument validation, handling of Base64 and file objects, unique filenames, API exceptions and successful multi-image saving. You can run these without provider credentials.

## Troubleshooting

- **Missing OPENAI_API_KEY, FAL_KEY or REPLICATE_API_TOKEN** — place the correct key in `.env` (copy from `.env.example`).
- **Authentication or insufficient quota** — make sure the provider account has usable API access and billing/credits.
- **Timeout or failed download** — verify internet connectivity; retry the command. URLs returned by image providers can expire.
- **Unsupported `--seed`, `--negative-prompt` or `--quality`** — select the compatible provider in the table above.
- **Directory errors** — `--output` must be a folder you can write to, not an existing file.
- **Partial success** — already generated images remain on disk even if a later API request fails.


