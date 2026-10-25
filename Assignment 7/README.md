# Image → Text → Image

Course: **AI APIs and Standalone AI Applications – ICT Robotics 2026 (IR00EU78-3002)**

A Python **command-line application** that:

1. Loads a local image (JPEG, PNG, or WebP).
2. Sends that image to the vision model `gpt-5.6-luna`.
3. Requests a detailed text description (150–250 words).
4. Prints the description and saves it to a `.txt` file.
5. Sends **only the description** as a text prompt to the image-generation model `gpt-image-2`.
6. Saves the generated PNG to a local folder.

**Important:** This is `image → text → image`, NOT direct image-to-image editing. The second model never receives the original image or its bytes.

## 1. Requirements

- Python 3.10+ (recommended Python 3.11/3.12)
- Internet connection
- OpenAI API key from <https://platform.openai.com/api-keys>
- Access and sufficient API credits for the two selected models. API use is **billed separately** from ChatGPT subscriptions. Image generation may require organization verification or enabling billing.

## 2. Install

Open a terminal in this project folder.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and replace the placeholder with **your own** key:

```dotenv
OPENAI_API_KEY=sk-...
```

**Do not send or upload your real key.** The `.gitignore` excludes `.env`.

## 3. Run

Try the included sample image:

```bash
python main.py examples/sample_input.png
```

Or your own photograph:

```bash
python main.py /path/to/your/photo.jpg
```

Example options:

```bash
python main.py examples/sample_input.png --output output/result.png --quality low --size 1536x1024
```

For Windows paths, use quotes when the path includes spaces:

```powershell
python main.py "C:\Users\You\Pictures\photo.jpg"
```

See all options:

```bash
python main.py --help
```

## 4. Outputs

By default the application creates:

```text
output/
├── description.txt    # Vision model's description, also printed in console
└── generated.png      # New image generated from that text alone
```

The program prints progress messages and a readable error if something goes wrong. It avoids overwriting the source image.

## 5. Pipeline diagram

```text
        local source image (.jpg/.png/.webp)
                       |
                       v
        [read, validate, Base64 encode]
                       |
                       v
       OpenAI Responses API / gpt-5.6-luna
                       |
                       v
          detailed plain-text description
              |                    |
              v                    v
       print in terminal      save description.txt
              |
              v
     OpenAI Images API / gpt-image-2
       (description as prompt ONLY)
              |
              v
          save generated.png
```

**API distinction:** Stage 1 uses `client.responses.create()` with an `input_image`; Stage 2 uses `client.images.generate()` with a text `prompt` and **no input image**.

## 6. Test the code without spending API credits

```bash
python -m unittest discover -s tests -v
```

The tests use fake responses instead of calling paid APIs. They confirm that Stage 1 receives the input image, Stage 2 receives text only, and a PNG can be saved. **These tests do not prove that your personal API key and model access work.**

For the actual assignment demonstration, run `python main.py examples/sample_input.png` with a valid key and inspect both files in `output/`.

## 7. Suggested submission evidence

- Screenshot of the terminal showing the printed description and saved file paths.
- Input image (`examples/sample_input.png` or your chosen image).
- Resulting `output/generated.png` created by **your own API run**.
- `output/description.txt` created by the same run.
- One paragraph explaining why the generated image may differ from the original.

**Before sharing your project, remove `.env` and any secrets.**

## 8. Limitations

- Details may be omitted, changed, or hallucinated during textual interpretation.
- The image-generation model creates a new visual and does not copy original pixels.
- Composition, identity, typography, or exact positions may differ.
- An internet connection, account access, and API billing are required for a live run.
- Input: JPEG/PNG/WebP, maximum 20 MiB. Output: PNG.

## 9. References

- OpenAI, [Images and vision guide](https://developers.openai.com/api/docs/guides/images-vision)
- OpenAI, [Image generation guide](https://developers.openai.com/api/docs/guides/image-generation)
- OpenAI, [GPT-5.6 Luna model](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- OpenAI, [GPT-Image-2 model](https://developers.openai.com/api/docs/models/gpt-image-2)
