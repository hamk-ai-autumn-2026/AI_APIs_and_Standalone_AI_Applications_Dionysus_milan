"""Run with: python -m streamlit run app.py"""

import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rewriter import RewriteError, rewrite_description
from scraper import ScrapeError, extract_product, scrape_product

load_dotenv(Path(__file__).with_name(".env"))
st.set_page_config(page_title="Product Page Scraper", page_icon="🛍️", layout="centered")
st.title("Product Page Scraper & Rewriter")
st.write("Extract product details from a public page, inspect the results, and optionally rewrite the description.")
st.caption("Reads HTML without running JavaScript. Some shops block automated access or require login.")

with st.form("product_form"):
    url = st.text_input("Product page URL", placeholder="https://shop.example/product")
    submitted = st.form_submit_button("Extract product", type="primary")

demo = st.button("Try included demo (no network needed)")
if submitted or demo:
    st.session_state.pop("product", None)
    st.session_state.pop("rewrite", None)
    st.session_state["demo"] = demo
    try:
        with st.spinner("Extracting and validating product details..."):
            if demo:
                html = Path(__file__).with_name("demo_product.html").read_text(encoding="utf-8")
                product = extract_product(html, "https://example.com/demo-product")
            else:
                product = scrape_product(url)
        st.session_state["product"] = product
    except ScrapeError as exc:
        st.error(str(exc))
    except Exception:
        st.error("The page could not be processed. Please try a different product URL.")

product = st.session_state.get("product")
if product is not None:
    if st.session_state.get("demo"):
        st.info("Demo data from the included HTML file. This is a fictional product, not a live scrape.")
    st.subheader("Extracted product")
    # Source text is displayed as plain text, never as executable HTML.
    st.text(product.name or "Name unavailable")
    left, right = st.columns(2)
    with left:
        st.write("Price")
        st.text(f"{product.price.amount} {product.price.currency or '(currency unknown)'}" if product.price else "Not available")
    with right:
        st.write("Review rating")
        st.text(f"{product.rating.value:g}" + (f" / {product.rating.best:g}" if product.rating.best is not None else " (scale unknown)") if product.rating else "Not available")
    st.write("Original description")
    st.text(product.description or "Not available")
    st.caption(f"Extraction: {product.extraction_method}")
    for warning in product.warnings:
        st.warning(warning)
    with st.expander("Validated structured data"):
        st.json(product.model_dump(mode="json"))
    st.download_button("Download product JSON", product.model_dump_json(indent=2),
                       file_name="product.json", mime="application/json")
    st.subheader("Optional description rewrite")
    st.caption("Sends only the extracted name and description to OpenAI. Review the rewrite for factual accuracy.")
    if st.button("Rewrite with OpenAI", disabled=not bool(product.description)):
        st.session_state.pop("rewrite", None)
        try:
            with st.spinner("Rewriting description..."):
                st.session_state["rewrite"] = rewrite_description(product)
        except RewriteError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Rewrite failed. The original extracted product is still available.")
    rewrite = st.session_state.get("rewrite")
    if rewrite:
        st.write("Rewritten description")
        st.text(rewrite.description)
        bundle = {"original_product": product.model_dump(mode="json"), "rewrite": rewrite.model_dump()}
        st.download_button("Download original + rewrite", json.dumps(bundle, indent=2, ensure_ascii=False),
                           file_name="product_with_rewrite.json", mime="application/json")
