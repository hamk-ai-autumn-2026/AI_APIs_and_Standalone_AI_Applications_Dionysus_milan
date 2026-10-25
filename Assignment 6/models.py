"""Pydantic models: article structure is validated before any PDF is written."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Author(StrictModel):
    given: str
    family: str


class Reference(StrictModel):
    id: str
    authors: list[Author]
    year: int
    title: str
    publication_title: str
    doi: str
    url: str
    volume: str = ""
    issue: str = ""
    pages: str = ""
    abstract: str = ""
    verified: bool
    verification_url: str = ""
    discovered_via: str = ""


class ArticleParagraph(StrictModel):
    text: str
    source_ids: list[str]


class ArticleSection(StrictModel):
    heading: str
    paragraphs: list[ArticleParagraph]


class ReportDraft(StrictModel):
    """Only fields the LLM is allowed to write; never bibliographic metadata."""
    title: str
    abstract: str
    introduction: list[ArticleParagraph]
    sections: list[ArticleSection]
    conclusion: list[ArticleParagraph]


class Report(ReportDraft):
    topic: str
    references: list[Reference]
    mode: str

    @model_validator(mode="after")
    def check_integrity(self):
        if not 5 <= len(self.topic) <= 350:
            raise ValueError("Topic must contain 5-350 characters")
        if len(self.title.strip()) < 12 or len(self.abstract.strip()) < 80:
            raise ValueError("Title/abstract is too short")
        if not self.introduction or not self.conclusion:
            raise ValueError("Introduction and conclusion must have paragraphs")
        if not 2 <= len(self.sections) <= 6:
            raise ValueError("The article needs 2-6 main sections")
        if not 3 <= len(self.references) <= 12:
            raise ValueError("The article needs 3-12 references")
        if self.mode not in {"grounded", "no_web_search"}:
            raise ValueError("Unknown article mode")
        ids = [ref.id for ref in self.references]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate reference IDs")
        if len({s.heading.casefold().strip() for s in self.sections}) != len(self.sections):
            raise ValueError("Duplicate section headings")
        paragraphs = self.introduction + [p for s in self.sections for p in s.paragraphs] + self.conclusion
        if any(not s.paragraphs for s in self.sections):
            raise ValueError("A section is empty")
        used = set()
        for p in paragraphs:
            if len(p.text.strip()) < 45:
                raise ValueError("A paragraph is too short")
            if not p.source_ids:
                raise ValueError("Every paragraph needs an in-text citation")
            if len(set(p.source_ids)) != len(p.source_ids):
                raise ValueError("Duplicate citation IDs in a paragraph")
            used.update(p.source_ids)
        if used != set(ids):
            raise ValueError(f"Cited IDs and reference IDs differ (used={used}, references={set(ids)})")
        if self.mode == "grounded":
            if not all(r.verified and r.doi and r.verification_url for r in self.references):
                raise ValueError("Grounded report contains unverified references")
        return self


class MemoryReference(StrictModel):
    """Reference invented/recalled by the model WITHOUT any web tools."""
    id: str
    authors: list[str]
    year: int
    title: str
    publication_title: str
    doi_or_url: str


class MemoryDraft(ReportDraft):
    references: list[MemoryReference]
