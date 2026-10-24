# Verification - 23 October 2026

## Automated checks

Result: **37 tests passed** using Python 3.11.9, Streamlit 1.66.0,
HTTPX 0.28.1, Pydantic 2.13.5, OpenAI SDK 2.54.0, BeautifulSoup 4.15.0,
and python-dotenv 1.2.4.

Coverage includes:

- Complete JSON-LD product and JSON round-trip validation.
- Missing name, description, price, and rating.
- Invalid price/rating, JSON-LD graph and offer list, malformed JSON fallback.
- HTML extraction, description length limit, non-product and CAPTCHA pages.
- HTTP 401/403/429/404/410/500, connection failures, and timeouts.
- Streamed size limit without Content-Length, non-HTML and compressed responses.
- Redirect limits, invalid URLs, and rejection of private destination addresses.
- Rewrite request boundaries, absent description, and missing structured output.
- Streamlit startup, demo extraction, missing API key, and handled scrape failure.

The runner required environment-specific dependency setup and a home-directory
shim for Streamlit in the isolated execution environment. The final tests passed
without modifications to the application for that environment.

## Public page parsing check

A live HTML response (HTTP 200) was fetched from:
https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html

Both a separate HTML parsing check and a live end-to-end `scrape_product()`
download-and-extraction call returned:

- Name: A Light in the Attic
- Price: 51.77 GBP
- Description: present
- Rating: 3 out of 5
- Method: HTML selectors / metadata

This is a scraping practice shop. It does not establish compatibility with
Amazon, Temu, or other real shops.

