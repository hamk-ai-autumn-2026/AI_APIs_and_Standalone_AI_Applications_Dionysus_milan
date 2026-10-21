# Assignment #15 — AI News Explorer

A Streamlit application integrating **Serper News Search**, **OpenAI**, and environment variables.

## Setup

Requires Python 3.10+ and active Serper/OpenAI API keys. OpenAI usage may incur charges.

1. Open this folder in VS Code.
2. Create and activate a virtual environment (optional):
   - Windows: `py -m venv .venv` then `.venv\\Scripts\\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Install: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` in **this project folder**.
5. Replace the placeholders with your actual Serper and OpenAI keys. Never share or commit `.env`.
6. Launch: `streamlit run app.py`
7. Open the URL shown in your terminal (usually http://localhost:8501).

## Features / assignment mapping

1. **Topic and time period:** topic input and selectable 24 hours, 7 days, or 30 days.
2. **Current news retrieval:** Serper News API via a single POST to `https://google.serper.dev/news` with `q`, `num`, and `tbs` date filters.
3. **Summary:** OpenAI Responses API summarizes retrieved article titles and snippets.
4. **Sources:** each result includes title, publisher, reported date, excerpt, and clickable source link.
5. **Zero results/errors:** empty result warning, missing-key check, timeout/network/HTTP handling, LLM failure handling.
6. **Search limits:** one Serper request per submission, maximum 5 results and 15-minute response cache; one OpenAI call for a new result set.
7. **Evidence vs interpretation:** original Serper evidence listed separately; model instructed to split evidence-based summary from clearly labeled interpretation with `[S1]` source IDs.

## Important limitations

- Search-result snippets are **retrieved claims**, not independently verified facts. The model does not read full articles.
- Serper's date filter limits the requested period; results may show fuzzy dates like "2 days ago" or missing dates, which are not independently verified. ISO-formatted dates outside the requested window are filtered out.
- The application can summarize conflicting reports but does not independently fact-check them.
- OpenAI API is billed separately from ChatGPT subscriptions.

## Manual tests

| Test | Input / condition | Expected behavior |
|---|---|---|
| Normal search | "robotics" / Past 7 days | Up to five sources plus AI summary |
| Empty input | blank topic | Warning; no API call |
| No results | obscure phrase | Warning; no OpenAI call |
| Missing key | remove SERPER_API_KEY from `.env` | Missing-key error |
| Bad Serper key | invalid key | Serper HTTP error handled |
| Network failure | disable network | Network/timeout error |
| Low limit | select 1 result | Up to 1 source |
| AI error | invalid OpenAI key / no quota | OpenAI error message; source results remain visible |

## Architecture

User → Streamlit form → Serper News API → bounded article data → OpenAI Responses API → summary with citations and interpretation.

## Academic report summary

This retrieval-augmented application demonstrates integrating external web data with an LLM. It distinguishes externally retrieved article details from model-generated synthesis; limits API use; and gracefully handles incomplete results and connection errors. The interface provides links for users to check source claims.
