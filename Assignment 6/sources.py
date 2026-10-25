"""Live web-search evidence and DOI metadata verification via Crossref."""
from __future__ import annotations

import html
import re
from urllib.parse import unquote, quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from models import Reference, Author

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)


def extract_dois(value: str) -> list[str]:
    seen: set[str] = set()
    found: list[str] = []
    for match in DOI_PATTERN.finditer(unquote(value)):
        doi = match.group().rstrip(".,;:)]}\\\"'").lower()
        if doi not in seen:
            seen.add(doi)
            found.append(doi)
    return found


def session_with_retries() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "HAMK-ArticleGenerator/1.0 (educational reference checking)"})
    retry = Retry(total=2, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _response_data(response) -> dict:
    """OpenAI SDK response -> data, compatible with SDK model versions."""
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, dict):
        return response
    raise TypeError("Unsupported OpenAI response type")


def run_web_search(client, topic: str, model: str) -> dict:
    """OpenAI web search is required, not optional; fail if no search call happened."""
    prompt = (
        f"Search the WEB for peer-reviewed scientific publications about: {topic}. "
        "Find 6-9 genuinely relevant, existing publications. Include their exact DOI "
        "(starting 10.) and descriptive title and an evidence-based one-sentence summary. "
        "Prefer official publisher pages, DOI landing pages and scientific repositories. "
        "Do not invent DOIs or sources. Give DOI strings as plain text in the response."
    )
    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search", "search_context_size": "medium"}],
        tool_choice="required",
        include=["web_search_call.action.sources"],
        input=prompt,
    )
    data = _response_data(response)
    calls = [o for o in data.get("output", []) if o.get("type") == "web_search_call"]
    if not calls:
        raise RuntimeError("The model did not perform a web search; no article will be generated")
    text = response.output_text or ""
    urls: list[str] = []
    for entry in data.get("output", []):
        if entry.get("type") == "web_search_call":
            for src in (entry.get("action") or {}).get("sources", []) or []:
                if isinstance(src, dict) and src.get("url"):
                    urls.append(src["url"])
        if entry.get("type") == "message":
            for part in entry.get("content", []):
                for ann in part.get("annotations", []) or []:
                    if ann.get("type") == "url_citation" and ann.get("url"):
                        urls.append(ann["url"])
    urls = list(dict.fromkeys(urls))
    # Only DOIs surfaced in the actual web-search response/cited source URLs
    # can become candidates. The article-writing LLM cannot add new DOIs.
    doi_candidates = extract_dois(text + "\n" + "\n".join(urls))
    return {"searched": True, "response_text": text, "result_urls": urls,
            "candidate_dois": doi_candidates, "search_call_count": len(calls)}


def _year(meta: dict) -> int:
    for field in ("published", "published-print", "published-online", "issued", "created"):
        parts = (meta.get(field) or {}).get("date-parts", [])
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return 0


def _clean_html(raw: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw)).split())


def reference_from_crossref(meta: dict, source_id: str, discovered_via: str) -> Reference:
    doi = str(meta.get("DOI") or "").lower().strip()
    authors = [Author(given=str(a.get("given", "")).strip(),
                      family=str(a.get("family", "")).strip())
               for a in meta.get("author", []) if a.get("family")]
    titles = meta.get("title") or []
    venues = meta.get("container-title") or []
    year = _year(meta)
    if not (doi and titles and venues and authors and year):
        raise ValueError("Crossref record lacks a DOI, title, journal, author or year")
    return Reference(
        id=source_id, authors=authors, year=year, title=_clean_html(titles[0]),
        publication_title=_clean_html(venues[0]), doi=doi,
        url="https://doi.org/" + doi,
        volume=str(meta.get("volume") or ""), issue=str(meta.get("issue") or ""),
        pages=str(meta.get("page") or meta.get("article-number") or ""),
        abstract=_clean_html(meta.get("abstract") or "")[:2000],
        verified=True,
        verification_url="https://api.crossref.org/works/" + quote(doi, safe=""),
        discovered_via=discovered_via,
    )


def lookup_doi(session: requests.Session, doi: str) -> dict:
    url = "https://api.crossref.org/works/" + quote(doi, safe="")
    response = session.get(url, timeout=18)
    response.raise_for_status()
    meta = response.json()["message"]
    if (meta.get("DOI") or "").lower() != doi.lower():
        raise ValueError("DOI mismatch in metadata response")
    return meta


def crossref_search(session: requests.Session, topic: str, rows: int = 20) -> list[dict]:
    """Second actual ONLINE scholarly search, useful when general web results lack DOI links."""
    response = session.get("https://api.crossref.org/works",
                           params={"query.bibliographic": topic, "rows": rows,
                                   "filter": "type:journal-article"}, timeout=20)
    response.raise_for_status()
    return response.json().get("message", {}).get("items", [])


def find_verified_sources(client, topic: str, model: str, source_count: int,
                          session: requests.Session | None = None) -> tuple[list[Reference], dict]:
    session = session or session_with_retries()
    search = run_web_search(client, topic, model)
    refs: list[Reference] = []
    seen: set[str] = set()
    errors: list[str] = []
    for doi in search["candidate_dois"][:22]:
        if len(refs) >= source_count:
            break
        try:
            meta = lookup_doi(session, doi)
            ref = reference_from_crossref(meta, f"S{len(refs)+1}", "OpenAI web_search result")
            if ref.doi not in seen:
                refs.append(ref); seen.add(ref.doi)
        except (ValueError, KeyError, requests.RequestException) as exc:
            errors.append(f"{doi}: {str(exc)[:110]}")
    # Crossref's /works query itself is an ONLINE scholarly search. Each of its
    # returned metadata records is separately verified by fetching /works/{doi}.
    if len(refs) < source_count:
        try:
            candidates = crossref_search(session, topic)
            search["crossref_search_hits"] = len(candidates)
            for hit in candidates:
                if len(refs) >= source_count:
                    break
                doi = str(hit.get("DOI") or "").lower()
                if not doi or doi in seen:
                    continue
                try:
                    meta = lookup_doi(session, doi)
                    ref = reference_from_crossref(meta, f"S{len(refs)+1}", "Crossref scholarly web search")
                    refs.append(ref); seen.add(ref.doi)
                except (ValueError, KeyError, requests.RequestException) as exc:
                    errors.append(f"{doi}: {str(exc)[:110]}")
        except requests.RequestException as exc:
            errors.append("Crossref search failed: " + str(exc)[:140])
    search["verification_errors"] = errors
    search["verified_count"] = len(refs)
    if len(refs) < 3:
        raise RuntimeError(f"Only {len(refs)} sources were verified (minimum 3). "
                           "Try a more specific topic or check internet access.")
    return refs, search
