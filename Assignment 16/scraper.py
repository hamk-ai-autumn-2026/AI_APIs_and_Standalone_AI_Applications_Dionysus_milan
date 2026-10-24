"""Bounded HTML download and deterministic extraction; no webpage code runs."""

import ipaddress
import json
import re
import socket
import time
from decimal import Decimal, InvalidOperation
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from pydantic import ValidationError

from models import Price, Product, Rating

MAX_DOWNLOAD_BYTES = 10_000_000
MAX_DESCRIPTION_CHARS = 6000
MAX_REDIRECTS = 3
TOTAL_SECONDS = 30


class ScrapeError(Exception):
    """An expected failure that the UI can show without a traceback."""


def validate_url(url: str) -> str:
    url = url.strip()
    if len(url) > 2048:
        raise ScrapeError("URL is too long (maximum 2048 characters).")
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError()
        if parsed.username or parsed.password or parsed.port not in (None, 80, 443):
            raise ValueError()
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise ValueError()
    except socket.gaierror:
        raise ScrapeError("The website address could not be resolved. Check the URL and connection.") from None
    except (ValueError, OSError):
        raise ScrapeError("Enter a public HTTP/HTTPS product URL without login details or a custom port.") from None
    return url


def download_page(url: str, client=None) -> tuple[str, str]:
    """Stream and count bytes, including when Content-Length is absent/wrong.

    Request uncompressed HTML and reject compressed responses so a compressed
    payload cannot expand past the download limit. Redirects are checked afresh.
    client is injectable for offline tests.
    """
    if client is None:
        with httpx.Client(timeout=httpx.Timeout(10, connect=5), follow_redirects=False, trust_env=False) as owned:
            return download_page(url, owned)
    started = time.monotonic()
    try:
        for hop in range(MAX_REDIRECTS + 1):
            url = validate_url(url)
            with client.stream("GET", url, headers={
                "User-Agent": "StudentProductScraper/1.0 (educational single-page fetch)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "identity",
            }) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location or hop == MAX_REDIRECTS:
                        raise ScrapeError("The page has an invalid redirect or too many redirects.")
                    url = urljoin(url, location)
                    continue
                if response.status_code in (401, 403, 429):
                    raise ScrapeError(f"Page blocked or rate limited (HTTP {response.status_code}). Try a different public product page.")
                if response.status_code in (404, 410):
                    raise ScrapeError("This product page is unavailable or has been removed.")
                if response.status_code != 200:
                    raise ScrapeError(f"Website returned HTTP {response.status_code}. Try again later.")
                if not any(t in response.headers.get("content-type", "").lower() for t in ("text/html", "application/xhtml+xml")):
                    raise ScrapeError("This URL did not return an HTML product page.")
                if response.headers.get("content-encoding", "identity").lower() not in ("identity", ""):
                    raise ScrapeError("The server ignored the uncompressed download request. Try another product page.")
                length = response.headers.get("content-length", "")
                if length.isdigit() and int(length) > MAX_DOWNLOAD_BYTES:
                    raise ScrapeError("Page exceeds the 10 MB download limit.")
                chunks = bytearray()
                for chunk in response.iter_raw(chunk_size=16_384):
                    if time.monotonic() - started > TOTAL_SECONDS:
                        raise ScrapeError("Page download exceeded the 30-second time budget.")
                    if len(chunks) + len(chunk) > MAX_DOWNLOAD_BYTES:
                        raise ScrapeError("Page exceeds the 10 MB download limit.")
                    chunks.extend(chunk)
                encoding = response.encoding or "utf-8"
                try:
                    html = chunks.decode(encoding, errors="replace")
                except LookupError:
                    html = chunks.decode("utf-8", errors="replace")
                return html, str(response.url)
    except httpx.TimeoutException:
        raise ScrapeError("Website timed out. Check your connection or try again later.") from None
    except httpx.RequestError:
        raise ScrapeError("Network or connection failure. Check your connection and URL.") from None
    raise ScrapeError("Could not download this page.")


def clean_text(value, limit: int) -> str | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    text = BeautifulSoup(str(value)[:MAX_DOWNLOAD_BYTES], "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()[:limit] or None


def element_text(soup, selectors):
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            text = element.get("content") or element.get_text(" ", strip=True)
            if text and text.strip():
                return text
    return None


def product_nodes(data):
    """Walk JSON-LD graphs iteratively, limiting work and avoiding recursion."""
    pending = [data]
    visited = 0
    while pending and visited < 10_000:
        item = pending.pop()
        visited += 1
        if isinstance(item, list):
            pending.extend(reversed(item))
        elif isinstance(item, dict):
            types = item.get("@type", [])
            types = [types] if isinstance(types, str) else types
            if isinstance(types, list) and any(isinstance(t, str) and t.rsplit("/", 1)[-1] == "Product" for t in types):
                yield item
                # Do not traverse related products within this Product.
                continue
            pending.extend(reversed(list(item.values())))


def make_price(amount, currency, warnings):
    if amount is None:
        return None
    raw = str(amount).strip()
    # Schema.org numbers use a decimal point. Display strings may use a
    # decimal comma; mixed or ambiguous thousands formats are left unknown.
    match = re.fullmatch(r"(?:[A-Z]{3}\s*|[\u20ac\u00a3$]\s*)?(\d+(?:[.,]\d{1,2})?)(?:\s*[A-Z]{3})?", raw)
    if not match:
        warnings.append("Price format was ambiguous or invalid; price was left null.")
        return None
    code = str(currency).upper().strip() if isinstance(currency, str) else None
    if not code:
        code = "GBP" if "\u00a3" in raw else "EUR" if "\u20ac" in raw else None
    if code and not re.fullmatch(r"[A-Z]{3}", code):
        warnings.append("Currency code was invalid; currency was left null.")
        code = None
    try:
        return Price(amount=Decimal(match[1].replace(",", ".")), currency=code)
    except (InvalidOperation, ValidationError):
        warnings.append("Invalid price was left null.")
        return None


def make_rating(value, best, worst, warnings):
    if value is None:
        return None
    try:
        if any(isinstance(x, bool) for x in (value, best, worst)):
            raise ValueError()
        return Rating(value=float(value), best=float(best) if best is not None else None,
                      worst=float(worst) if worst is not None else None)
    except (ValueError, TypeError, ValidationError, OverflowError):
        warnings.append("Invalid or out-of-range rating was left null.")
        return None


def extract_product(html: str, url: str) -> Product:
    if len(html.encode("utf-8")) > MAX_DOWNLOAD_BYTES * 3:
        raise ScrapeError("HTML input exceeds the parser limit.")
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True).lower() if soup.title else ""
    if any(x in title for x in ("captcha", "robot check", "access denied", "just a moment", "verify you are human")) or soup.select_one("#captchacharacters, #challenge-form, form[action*='validateCaptcha']"):
        raise ScrapeError("This website returned a CAPTCHA or access challenge. Try another public product page.")
    warnings = []
    nodes = []
    for script in soup.select('script[type="application/ld+json"]')[:30]:
        raw = script.string or script.get_text()
        if not raw or len(raw) > 500_000:
            continue
        try:
            nodes.extend(product_nodes(json.loads(raw)))
        except (ValueError, RecursionError):
            warnings.append("A malformed JSON-LD block was skipped.")
    # Prefer a Product whose URL matches the fetched page; otherwise the first.
    page_key = url.split("?", 1)[0].rstrip("/")
    node = next((n for n in nodes if isinstance(n.get("url"), str) and urljoin(url, n["url"]).split("?", 1)[0].rstrip("/") == page_key), nodes[0] if nodes else {})
    if len(nodes) > 1:
        warnings.append("Multiple Product records found; verify that the selected product matches the page.")
    method = "JSON-LD Product" if node else "HTML selectors / metadata"

    name = clean_text(node.get("name"), 500) or clean_text(element_text(soup, ["#productTitle", ".product_main h1", "[itemtype*='Product'] [itemprop='name']", "main h1", "h1"]), 500)
    description = clean_text(node.get("description"), MAX_DESCRIPTION_CHARS)
    if not description:
        description = clean_text(element_text(soup, ["#productDescription", "#feature-bullets", "[itemprop='description']", "#product_description + p", "meta[property='og:description']", "meta[name='description']"]), MAX_DESCRIPTION_CHARS)
    offers = node.get("offers", {})
    if isinstance(offers, list):
        if len(offers) > 1:
            warnings.append("Multiple offers found; the first offer is shown. Check variant and seller.")
        offers = offers[0] if offers else {}
    offers = offers if isinstance(offers, dict) else {}
    amount = offers.get("price")
    currency = offers.get("priceCurrency")
    if amount is None and offers.get("lowPrice") is not None:
        amount = offers["lowPrice"]
        warnings.append("Price is the lowest listed offer, not necessarily the selected variant price.")
    if amount is None and not node:
        amount = element_text(soup, ["meta[property='product:price:amount']", "[itemprop='price']", "#corePrice_feature_div .a-price .a-offscreen", ".product_main .price_color", ".product .price"])
        currency = element_text(soup, ["meta[property='product:price:currency']", "[itemprop='priceCurrency']"])
    price = make_price(amount, currency, warnings)

    rating_data = node.get("aggregateRating", {})
    rating_data = rating_data if isinstance(rating_data, dict) else {}
    value, best, worst = (rating_data.get(k) for k in ("ratingValue", "bestRating", "worstRating"))
    if value is None and not node:
        value = element_text(soup, ["[itemprop='aggregateRating'] [itemprop='ratingValue']", "[itemprop='ratingValue']"])
        best = element_text(soup, ["[itemprop='aggregateRating'] [itemprop='bestRating']"])
        worst = element_text(soup, ["[itemprop='aggregateRating'] [itemprop='worstRating']"])
        stars = soup.select_one(".product_main .star-rating")
        if value is None and stars:
            names = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
            value = next((names[c] for c in stars.get("class", []) if c in names), None)
            best, worst = 5, 1
    rating = make_rating(value, best, worst, warnings)
    evidence = bool(node) or soup.select_one("#productTitle, .product_main, [itemtype*='Product'], meta[property='product:price:amount'], [itemprop='price'], .product .price") is not None
    if not evidence or not any((name, description, price, rating)):
        raise ScrapeError("No usable product data found. The page may require JavaScript, login, or a different extraction method.")
    for key, value in (("name", name), ("price", price), ("description", description), ("rating", rating)):
        if value is None:
            warnings.append(f"{key.capitalize()} was not available; stored as null.")
    if description and len(description) == MAX_DESCRIPTION_CHARS:
        warnings.append("Description may have been shortened to the 6000-character limit.")
    try:
        return Product(source_url=url, name=name, price=price, description=description,
                       rating=rating, extraction_method=method, warnings=warnings[:30])
    except ValidationError:
        raise ScrapeError("The extracted product could not be validated.") from None


def scrape_product(url: str) -> Product:
    html, final_url = download_page(url)
    return extract_product(html, final_url)
