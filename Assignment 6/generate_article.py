"""Assignment #6 - Grounded technical article generator.

Flow: topic -> OpenAI web_search -> Crossref-verified papers -> structured
      LLM ReportDraft -> Pydantic Report validation -> article.pdf.

Default also writes a no-web-search article and evaluates its references AFTER
it has been generated, to show potential reference hallucinations.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import uuid

from dotenv import load_dotenv
from pydantic import ValidationError
import requests

from comparison import evaluate_memory_references, reference_from_memory, write_comparison
from models import MemoryDraft, Report, ReportDraft, Reference
from pdf_renderer import render_pdf
from sources import find_verified_sources, session_with_retries


def arguments(argv=None):
    parser = argparse.ArgumentParser(description="Generate a grounded APA-style scientific article PDF")
    parser.add_argument("--topic", help="Research topic (asks interactively when missing)")
    parser.add_argument("--output-dir", default="output", help="Base output folder (default: output)")
    parser.add_argument("--model", default="gpt-5.5", help="Text/search model (default: gpt-5.5)")
    parser.add_argument("--source-count", type=int, default=4, help="Desired verified papers (3-8)")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip memory-only baseline and comparison")
    args = parser.parse_args(argv)
    if not 3 <= args.source_count <= 8:
        parser.error("--source-count must be 3-8")
    if not args.topic:
        try:
            args.topic = input("Enter a scientific or technical article topic: ").strip()
        except EOFError:
            parser.error("Provide --topic when input is noninteractive")
    args.topic = args.topic.strip()
    if not 5 <= len(args.topic) <= 350:
        parser.error("--topic must contain 5-350 characters")
    if not args.output_dir.strip():
        parser.error("--output-dir cannot be empty")
    return args


def _save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return (value[:55].rstrip("-") or "technical-topic")


def create_run_folder(base: str, topic: str) -> Path:
    root = Path(base).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f"{_slug(topic)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=False, exist_ok=False)
    return folder


def draft_grounded(client: OpenAI, model: str, topic: str, refs: list[Reference]) -> ReportDraft:
    source_briefs = [{
        "id": r.id, "author": r.authors[0].family + " et al.", "year": r.year,
        "title": r.title, "journal": r.publication_title,
        "doi": r.doi, "abstract_if_available": r.abstract or "[No abstract available: only use the title as evidence]",
    } for r in refs]
    instruction = (
        "Create an accurate, readable SHORT technical/scientific overview article. "
        "Return the schema's title, abstract, introduction, 2-4 headed main sections, and conclusion. "
        "Introduction should have 2 paragraphs, main sections 1-2 paragraphs each, conclusion 1-2 paragraphs. "
        "Each paragraph should be 70-140 words and cite at least one PROVIDED source by exact ID "
        "in source_ids. Cite AT LEAST THREE different source IDs across the article. "
        "Only cite the listed sources. No invented authors, statistics, numerical outcomes, "
        "quotes, study results, references, DOIs, or URLs. "
        "Only assert findings supported by supplied abstracts. If no abstract is supplied, "
        "a title can establish the paper's topic but not empirical findings. "
        "Use cautious, specific technical language, and distinguish established research from future work. "
        "DO NOT put APA author/year citations directly into text: the renderer adds them "
        "from verified metadata after Pydantic validation. Never copy a reference list into fields."
    )
    response = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": instruction},
               {"role": "user", "content": "Topic: " + topic + "\nVerified publication records:\n" +
                json.dumps(source_briefs, ensure_ascii=False)}],
        text_format=ReportDraft,
    )
    if response.output_parsed is None:
        raise RuntimeError("No structured grounded report returned (possible refusal/truncation)")
    return response.output_parsed


def draft_memory_only(client: OpenAI, model: str, topic: str) -> MemoryDraft:
    """No tools, no search output, and no verified sources passed in this request."""
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": (
                "You are generating a comparison baseline for a citation-quality exercise. "
                "Use only your internal knowledge. You have NO access to web search. "
                "Write a short academic technical article about the supplied topic, with "
                "2-4 logically divided main sections, abstract, introduction and conclusion. "
                "Supply FOUR references from memory in the references field, in good faith. "
                "Do not invent sources deliberately; state details you believe to be accurate. "
                "Give each reference ID R1, R2, R3, R4 and ensure ALL four are cited "
                "in paragraph source_ids. Provide the author names, exact titles, journals, years, "
                "and DOI/URLs as best you remember. Do not add text citations in the prose. "
                "Every paragraph must cite one or more R-IDs. "
                "Do not browse or use tools; this is intentionally ungrounded.")},
            {"role": "user", "content": "Write the same-topic article: " + topic},
        ],
        text_format=MemoryDraft,
    )
    if response.output_parsed is None:
        raise RuntimeError("No structured no-web article returned")
    return response.output_parsed


def draft_to_report(draft: ReportDraft, topic: str, refs: list[Reference], mode: str) -> Report:
    used: set[str] = set()
    for p in draft.introduction + [p for s in draft.sections for p in s.paragraphs] + draft.conclusion:
        used.update(p.source_ids)
    chosen = [r for r in refs if r.id in used]
    # Reports MUST be validated fully before render_pdf is called.
    return Report.model_validate({**draft.model_dump(exclude={"references"}),
                                  "topic": topic, "references": [r.model_dump() for r in chosen],
                                  "mode": mode})


def run():
    load_dotenv()
    args = arguments()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing. Copy .env.example to .env and add your API key.")
    try:
        from openai import OpenAI  # installed via requirements.txt
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install -r requirements.txt") from exc
    client = OpenAI(timeout=120.0, max_retries=2)
    folder = create_run_folder(args.output_dir, args.topic)
    print(f"\nTopic: {args.topic}\nRun folder: {folder}\n")
    print("[1/5] Searching the web and checking DOI metadata via Crossref...")
    session = session_with_retries()
    refs, search_details = find_verified_sources(client, args.topic, args.model, args.source_count, session)
    _save_json(folder / "web_search_evidence.json", search_details)
    _save_json(folder / "verified_sources.json", [r.model_dump() for r in refs])
    print(f"      {len(refs)} references confirmed against Crossref metadata")
    for r in refs:
        print(f"      {r.id}: {r.authors[0].family} ({r.year}) - {r.doi}")

    print("[2/5] Generating structured grounded article...")
    grounded_draft = draft_grounded(client, args.model, args.topic, refs)
    print("[3/5] Validating Report using Pydantic and rendering PDF...")
    grounded = draft_to_report(grounded_draft, args.topic, refs, "grounded")
    _save_json(folder / "grounded_report.json", grounded.model_dump())
    pdf = folder / "article.pdf"
    render_pdf(grounded, pdf)
    print(f"      SAVED: {pdf}")

    if args.skip_comparison:
        print("[4/5] No-web-search comparison skipped (--skip-comparison)")
        print(f"[5/5] Complete. Main PDF: {pdf}")
        return

    print("[4/5] Generating same-topic baseline WITHOUT web search...")
    baseline = draft_memory_only(client, args.model, args.topic)
    baseline_refs = [reference_from_memory(r) for r in baseline.references]
    no_web = draft_to_report(baseline, args.topic, baseline_refs, "no_web_search")
    _save_json(folder / "no_web_report.json", no_web.model_dump())
    baseline_pdf = folder / "no_web_article.pdf"
    render_pdf(no_web, baseline_pdf)
    print(f"      SAVED: {baseline_pdf}")

    print("[5/5] Checking baseline references only AFTER generation...")
    evaluation = evaluate_memory_references(baseline.references, session)
    write_comparison(folder, grounded.references, evaluation)
    counts = evaluation["counts"]
    print(f"      Web-grounded: {len(grounded.references)} verified; baseline: "
          f"{counts['genuine']} genuine, {counts['incorrect']} incorrect, "
          f"{counts['unverified']} unverified")
    print(f"\nCompleted. Results are in: {folder.resolve()}")
    print(f"Main PDF: {pdf.resolve()}")
    print(f"Comparison: {(folder / 'reference_comparison.md').resolve()}")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print("\nERROR: " + str(exc), file=sys.stderr)
        print("No fabricated article or bibliography will be substituted. "
              "Check the API key, connection, topic, source metadata, and try again.", file=sys.stderr)
        sys.exit(1)
