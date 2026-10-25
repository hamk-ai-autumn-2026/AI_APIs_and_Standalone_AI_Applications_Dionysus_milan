# Assignment #6 — Grounded Article Generator and Structured PDF

A Python **command-line application** that asks for a scientific/technical topic and saves an **APA-cited PDF**. It also implements a controlled **no-web-search comparison**, rather than only describing one.

## Complete pipeline

```text
User topic (interactive or --topic)
    |
    v
OpenAI Responses API with REQUIRED web_search tool
    |
    v
DOIs taken from the actual search response/source URLs
    |
    v
Crossref /works/{doi} metadata verification
    |    (or additional real Crossref scholarly search results if needed)
    v
Reference objects (author, article title, journal, year, DOI URL)
    |
    v
LLM returns structured ReportDraft (Pydantic schema)
    |
    v
Code constructs final Report with Crossref reference metadata
    |
    v
Pydantic validates citation IDs, section count and reference provenance
    |
    v
ReportLab renders article.pdf with APA-style in-text citations and references
```

**Important design principle:** The report-writing LLM **cannot invent bibliography entries for the grounded report**. Reference metadata comes from independent Crossref lookups, not from the model's text. Citation IDs are validated against this fixed list. The program stops on validation errors rather than silently fabricating references.

## Installation (Windows PowerShell / VS Code)

Use Python 3.10+ (3.11 or newer recommended). Extract the ZIP, open the project folder in VS Code, then run:

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` (in VS Code file explorer or in PowerShell):

```powershell
Copy-Item .env.example .env
```

Open `.env` and enter **your own** OpenAI API key:

```dotenv
OPENAI_API_KEY=your_real_key_goes_here
```

Create a key at https://platform.openai.com/api-keys. OpenAI's **API is billed separately** from ChatGPT subscriptions. The project also makes HTTPS requests to the Crossref scholarly metadata API. You need internet access.

## Generate articles

**Interactive (asks for topic):**

```powershell
python generate_article.py
```

**Provide your topic in one PowerShell line (recommended):**

```powershell
python generate_article.py --topic "Robotics in precision agriculture" --output-dir output
```

Another example:

```powershell
python generate_article.py --topic "Applications of machine learning in medical image segmentation" --source-count 5
```

By default, it uses `gpt-5.5`, which supports OpenAI Responses web search and structured outputs. To choose a different compatible model, pass `--model MODEL_NAME`.

The `--source-count` option accepts **3-8**, default **4**. It may find more papers than the article cites; the saved `verified_sources.json` contains all selected search/metadata records, and `grounded_report.json` contains the references the article actually cites.

To make **only the grounded article** (fewer model API calls):

```powershell
python generate_article.py --topic "Robotics in precision agriculture" --skip-comparison
```

**PowerShell tip:** All commands above are on one line to avoid PowerShell parsing errors. Bash/Linux examples using a trailing backslash `\` are not correct PowerShell line-continuation syntax.

## Output files

Each run creates a separate timestamped folder under `output`:

```text
output/
  robotics-in-precision-agriculture-20261025-160000-a1b2c3/
    article.pdf                  <- GROUNDED main PDF
    grounded_report.json        <- validated Pydantic Report
    web_search_evidence.json    <- actual search response + result URLs
    verified_sources.json       <- DOI, authors, title, journal, year, Crossref verification URL
    no_web_article.pdf          <- same topic, NO web search during writing
    no_web_report.json          <- structured comparison article
    reference_comparison.md     <- how many references are genuine, incorrect, unverified
    reference_comparison.json
    reference_comparison.csv
```

The no-web files are omitted with `--skip-comparison`.

## Research integrity / APA citations

- Live searches happen through OpenAI's `web_search` API tool. The code requires a `web_search_call` to be present and extracts candidate DOIs from the actual response and source URLs.
- Each DOI is independently retrieved from `https://api.crossref.org/works/{doi}`. The author list, article title, journal/publication title, publication year and canonical DOI URL are populated from that record, not invented by the LLM.
- If general web search returns too few usable DOIs, the app searches the public Crossref scholarly index, then **re-fetches each DOI's individual metadata record**. Sources from this fallback are marked as such.
- The writing model sees the **verified bibliographic records and available abstracts**. If an abstract is missing, it is instructed **not** to make specific unsupported claims about that study's results. This is metadata/abstract grounding, **not** a full-text systematic evidence review.
- The generated `ReportDraft` is transformed into a `Report`. Pydantic rejects missing sections, uncited/unknown reference IDs, unverified sources for grounded mode, invalid structure, and too-short paragraphs.
- In-text citations are generated from metadata as `(Smith & Lee, 2022)` or `(Smith et al., 2022)`. The bibliography includes author initials, publication year, title, journal, volume/issue/pages if available, and DOI links. Reference list uses hanging indentation.
- Citations are **APA-style** but still require final human proofreading for strict APA 7 compliance, capitalization rules, source relevance, and whether article claims are actually supported by the underlying papers.

## How the no-web-search comparison works

The application makes a **separate** `responses.parse()` request with **NO tools enabled** and WITHOUT providing any previously discovered verified sources. That model writes another article on the same topic and lists four references it recalls from memory.

**Only after its article is complete**, the application looks up its claimed DOIs in Crossref, and checks:

1. First author's surname.
2. Article title (high-similarity matching).
3. Journal/publication title (high-similarity matching).
4. Publication year (exact).
5. DOI (exact).

A reference becomes `genuine` only when all checks pass. A record with a resolvable DOI but mismatched fields is `incorrect`. An unresolvable DOI or missing DOI becomes `unverified`, not automatically "hallucinated." Results are saved as MD/JSON/CSV and presented in a count table.

This is an honest test: **the number of fabricated references is not predetermined** and might be zero. The program reports whatever the verification actually finds.

## Automated tests

No API key or network needed:

```powershell
python -m unittest discover -s tests -v
```

Tests cover Pydantic validation failures, APA citation formatting, DOI extraction, web-search enforcement, metadata parsing, reference-mismatch detection, unverified categorization, and successful PDF creation.

## Demonstration PDF

`sample_output/example_article.pdf` is a **layout demonstration** on agricultural robotics, with four manually checked bibliographic records from publisher pages. It was NOT produced by the live OpenAI/Crossref pipeline. The evidence links are listed in `sample_output/SAMPLE_ORIGIN.md`. For your assignment, run `generate_article.py` using your API key and submit the new files in `output/` instead.

## Troubleshooting

**`Missing OPENAI_API_KEY`** — create `.env` beside `generate_article.py` and fill in a valid key.

**`No module named openai`** — run `python -m pip install -r requirements.txt` using the same Python interpreter you use to run the app.

**`Only X sources were verified`** — choose a narrower scholarly topic, check your internet connection, and rerun. The program intentionally refuses to invent references.

**Model access error / quota exceeded / web search unavailable** — check your API account model permissions and usage/billing settings. The script does not bypass provider errors.

**Pydantic validation failed** — a generation did not comply with schema or citation rules. This is a protection, not an invitation to copy unverified references into the PDF. Rerun with a more focused topic.

**HTTPS / network failure** — check internet/proxy settings. Both OpenAI and `api.crossref.org` must be reachable.

