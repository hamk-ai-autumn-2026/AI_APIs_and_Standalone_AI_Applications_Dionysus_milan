# Assignment 16 - Product Page Scraper and Rewriter

A local Streamlit application that extracts product details from a public URL,
validates them with Pydantic, and optionally rewrites the description with OpenAI.
HTTPX + BeautifulSoup is the alternative page extraction method allowed by the assignment.
No search API key or Crawl4AI installation is needed.

## 1. Install and run (Windows / VS Code)

Install Python 3.11 or newer. Open this project folder in VS Code and select
**Terminal > New Terminal**. Run the commands below from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` in VS Code and replace the placeholder:

```dotenv
OPENAI_API_KEY=your_actual_api_key
OPENAI_MODEL=gpt-4.1-mini
```

Run the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit opens the app in your browser, usually at http://localhost:8501.
These commands use the environment's Python directly, so PowerShell activation
and execution-policy changes are unnecessary. Stop the app with Ctrl+C.
An OpenAI key is only needed when you click **Rewrite with OpenAI**; extraction
and the included demo work without a key. API use may incur charges.
Restart after editing `.env`. Never upload `.env` or `.venv` to GitHub.

## 2. Use the app

1. Click **Try included demo** to inspect a fictional product from a bundled HTML
   file. This does not test the network and is clearly labeled as demo data.
2. Paste a public product page URL and click **Extract product**.
3. Inspect the name, price, description, rating, and warnings.
4. Expand **Validated structured data** and download `product.json`.
5. Optionally click **Rewrite with OpenAI**. The original stays separate.
6. Download the original and rewrite together and check that the rewrite added no claims.

A useful practice page is:
https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html
Books to Scrape is a demonstration shop; its data is for scraping practice.
You can also try a real public shop page with Schema.org Product JSON-LD.
Amazon/Temu may block requests or require browser JavaScript. A friendly failure
in that case is expected; this app does not bypass access challenges.

## 3. How the code works

```text
Product URL
  -> public URL check
  -> bounded HTTPX HTML download
  -> BeautifulSoup + Product JSON-LD / HTML extraction
  -> Pydantic Product validation
  -> display and JSON download
  -> optional OpenAI structured rewrite
  -> Pydantic Rewrite validation
```

`app.py` is the Streamlit interface. Forms run extraction only when submitted;
session state retains the result when you click rewrite or download.
`scraper.py` handles downloading and extraction. `models.py` defines validated
types. `rewriter.py` makes the optional LLM call. `test_app.py` covers extraction,
network errors, limits, rewrite request isolation, and Streamlit interactions.

Many pages publish a JSON-LD script with `@type: Product`. This app reads its
`name`, `description`, `offers.price`, `offers.priceCurrency`, and
`aggregateRating`. It also handles graphs and offer lists. It prefers a matching
Product URL, otherwise the first Product, and warns about multiple products or
offers. Supported HTML fallbacks include product metadata, microdata, common
Amazon HTML selectors, and Books to Scrape selectors. Coverage varies by shop.

Every result is a real Pydantic `Product`, not unchecked JSON text. Price is a
nonnegative finite Decimal; rating is finite and checked against an explicit
scale when available. A rating without an explicit scale is displayed with
"scale unknown". Decimal amounts are serialized as strings to preserve precision.
Missing fields are `None` in Python and `null` in JSON. Invalid fields become
null with a warning; we do not invent values. A non-product page or page with no
usable product facts produces a handled error.

Example of validated output:

```json
{
  "source_url": "https://example.com/demo-product",
  "name": "Demo USB Desk Fan",
  "price": {"amount": "19.90", "currency": "EUR"},
  "description": "A compact USB-powered desk fan with three speed settings and an adjustable stand. Includes a 1-metre USB cable. Available in white.",
  "rating": {"value": 4.3, "best": 5.0, "worst": 1.0},
  "extraction_method": "JSON-LD Product",
  "warnings": []
}
```

## 4. Assignment requirements

| Requirement | Implementation |
|---|---|
| Accept a product URL | Streamlit URL input and extraction button |
| Retrieve the page | HTTPX streaming GET + BeautifulSoup parsing |
| Name, price, description, rating | JSON-LD first, supported HTML fallbacks second |
| Validated structured output | Pydantic Product, Price, Rating, and Rewrite models |
| Limit downloaded content | Count streamed bytes; reject over 2,000,000 bytes even without Content-Length |
| Limit LLM input | Only name (500 characters) + description (6000 characters), plus JSON formatting; no full HTML |
| Bound model output | max_output_tokens=1200 and a 4000-character Rewrite field |
| Handle unavailable/blocked pages | HTTP status handling, CAPTCHA checks, no-product errors |
| Handle network failures | Connection/read timeouts and handled HTTPX exceptions |
| Treat page instructions as untrusted | Deterministic extraction; separate model instructions and untrusted user data; no tools |
| Missing fields never crash the app | Optional fields with explicit None defaults, validation, UI fallbacks |
| Optional LLM rewrite | Separate button, OpenAI Responses API with Pydantic output parsing |

The downloader requests uncompressed content and rejects a server that ignores
that request. It checks content type, limits redirects to three, uses 5-second
connect / 10-second inactivity timeouts, and checks a 30-second download budget
between chunks. A pending read or DNS lookup may extend the elapsed time; this
is not a strict wall-clock deadline. Initial URLs and redirect destinations are
checked for public addresses. This is a local coursework app, not a hardened
public URL-fetching service; DNS checking is not a complete SSRF defense.

## 5. Webpage instructions and rewriting

The app never executes HTML, JavaScript, shell commands, or instructions from
the page. It extracts fields with Python. Only the extracted name and description
enter the model request as untrusted JSON in a user message, separately from
developer-controlled instructions. No model tools or browsing are enabled.
The prompt tells the model to ignore embedded commands and use only supplied
facts. These controls reduce prompt-injection risk; they do not prove that model
text is correct. Pydantic validates structure, not factual truth. Review every
rewrite before using it. Source text and rewrites are rendered as plain text.

## 6. Tests and submission evidence

Install the test dependencies and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Automated tests use fixture HTML and mocked HTTP/API responses, so they need
neither network access nor paid API calls. The rewrite injection test checks
message separation and disabled tools; it is not a live model security guarantee.

Before submission, do these manual checks:

| Check | Expected result |
|---|---|
| Included demo | Fictional fan, EUR 19.90, rating 4.3/5 |
| Public product URL | Values match the fetched page; inspect variants/currency |
| Product without a rating | Rating is null, warning shown |
| Invalid URL / unavailable URL | Friendly error, application still responds |
| Blocked shop page | Friendly blocked or unusable-page message |
| Rewrite with valid OpenAI key | Original retained, rewritten description shown separately |
| Missing/invalid API key | Friendly rewrite error, original retained |


## Documentation

- HTTPX streaming and timeouts: https://www.python-httpx.org/quickstart/ and https://www.python-httpx.org/advanced/timeouts/
- Product JSON-LD vocabulary: https://schema.org/Product
- Pydantic field defaults and validation: https://docs.pydantic.dev/latest/concepts/fields/
- Streamlit state: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
- OpenAI Python SDK: https://github.com/openai/openai-python
