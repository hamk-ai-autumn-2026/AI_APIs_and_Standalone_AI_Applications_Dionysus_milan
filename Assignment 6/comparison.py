"""Post-hoc evaluation of NO-WEB article reference accuracy."""
from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import unquote
import json
import csv

import requests

from models import MemoryReference, Reference, Author
from sources import extract_dois, lookup_doi, reference_from_crossref


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, " ".join(a.lower().split()), " ".join(b.lower().split())).ratio()


def assess_memory_reference(item: MemoryReference, session: requests.Session) -> dict:
    """Independent check; nothing found here is fed back into the no-web generation."""
    d = {"id": item.id, "claimed_title": item.title, "claimed_year": item.year,
         "claimed_authors": item.authors, "claimed_publication_title": item.publication_title,
         "claimed_doi_or_url": item.doi_or_url, "status": "unverified", "checks": {},
         "verified_metadata": None, "reason": ""}
    dois = extract_dois(item.doi_or_url)
    if not dois:
        d["reason"] = "No DOI provided; not counted as genuine or hallucinated."
        return d
    doi = dois[0]
    try:
        real = reference_from_crossref(lookup_doi(session, doi), item.id, "post-hoc evaluation")
    except (requests.RequestException, ValueError, KeyError) as exc:
        d["reason"] = "DOI could not be verified: " + str(exc)[:180]
        return d
    title_ok = _similar(item.title, real.title) >= 0.88
    year_ok = item.year == real.year
    venue_ok = _similar(item.publication_title, real.publication_title) >= 0.80
    family_claimed = item.authors[0].split(",")[0].strip().lower() if item.authors else ""
    family_actual = real.authors[0].family.lower() if real.authors else ""
    author_ok = bool(family_claimed and family_actual and
                     (family_claimed == family_actual or family_actual in family_claimed))
    url_ok = doi == real.doi
    d["checks"] = {"first_author": author_ok, "publication_title": venue_ok,
                   "publication_year": year_ok, "article_title": title_ok, "DOI": url_ok}
    d["verified_metadata"] = {"title": real.title, "year": real.year,
        "first_author": real.authors[0].family, "publication_title": real.publication_title,
        "doi": real.doi, "url": real.url, "checked_against": real.verification_url}
    d["status"] = "genuine" if all(d["checks"].values()) else "incorrect"
    if d["status"] != "genuine":
        d["reason"] = "DOI exists but one or more cited bibliographic fields disagree."
    return d


def evaluate_memory_references(refs: list[MemoryReference], session: requests.Session) -> dict:
    results = [assess_memory_reference(r, session) for r in refs]
    counts = {k: sum(r["status"] == k for r in results) for k in ("genuine", "incorrect", "unverified")}
    return {"total": len(results), "counts": counts, "results": results,
            "method": "Crossref DOI lookup, first-author surname, article and journal titles, year, and DOI. "
                      "Unable-to-check is UNVERIFIED, not automatically hallucinated."}


def reference_from_memory(mem: MemoryReference) -> Reference:
    authors = []
    for raw in mem.authors:
        left, sep, right = raw.partition(",")
        if not sep:
            words = raw.strip().split()
            authors.append(Author(family=words[-1] if words else "Unknown", given=" ".join(words[:-1])))
        else:
            authors.append(Author(family=left.strip() or "Unknown", given=right.strip()))
    dois = extract_dois(mem.doi_or_url)
    doi = dois[0] if dois else ""
    return Reference(id=mem.id, authors=authors, year=mem.year, title=mem.title,
                     publication_title=mem.publication_title, doi=doi,
                     url="https://doi.org/" + doi if doi else mem.doi_or_url,
                     verified=False, discovered_via="Model memory only")


def write_comparison(folder: Path, grounded: list[Reference], evaluation: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    total_g = len(grounded)
    counts = evaluation["counts"]
    doc = ["# Grounded vs. no-web-search reference evaluation", "",
        "**Method:** The same topic was used for both articles. The grounded article "
        "obtained candidates through live web search and kept only Crossref-confirmed metadata. "
        "The baseline article was generated with NO browsing tools. Its references were checked only AFTER generation; "
        "the checks did not affect what the model wrote.", "",
        "| Mode | Total references | Genuine/verified | Incorrect | Unverified |",
        "|---|---:|---:|---:|---:|",
        f"| Web-grounded | {total_g} | {total_g} | 0 | 0 |",
        f"| Without web search | {evaluation['total']} | {counts['genuine']} | {counts['incorrect']} | {counts['unverified']} |",
        "", "**Warning:** These counts apply only to bibliographic details, not whether the papers "
        "support every claim. An unverifiable item is not proven fabricated.", "",
        "## Grounded references (checked in Crossref)", ""]
    for r in grounded:
        doc.append(f"- {r.authors[0].family} ({r.year}). {r.title}. *{r.publication_title}*. {r.url} — {r.verification_url}")
    doc += ["", "## No-web-search reference checks", ""]
    for r in evaluation["results"]:
        checks = ", ".join(k + "=" + ("pass" if v else "FAIL") for k, v in r["checks"].items())
        doc.append(f"- **{r['id']} — {r['status'].upper()}**: {r['claimed_title']} ({r['claimed_year']}). "
                   f"{r['claimed_doi_or_url']}. {checks}. {r['reason']}")
    (folder / "reference_comparison.md").write_text("\n".join(doc) + "\n", encoding="utf-8")
    (folder / "reference_comparison.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
    with (folder / "reference_comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["reference_id", "status", "claimed_title", "claimed_year", "claimed_doi_or_url", "reason"])
        for r in evaluation["results"]:
            writer.writerow([r[k] for k in ("id", "status", "claimed_title", "claimed_year", "claimed_doi_or_url", "reason")])
