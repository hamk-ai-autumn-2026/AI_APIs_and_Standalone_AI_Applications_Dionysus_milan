"""Academic-looking PDF output via ReportLab (no external PDF executables)."""
from __future__ import annotations

from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether,
)

from models import Report, Reference, ArticleParagraph, Author


def _fonts() -> tuple[str, str]:
    # Fonts are loaded from the user's OS; no font files are packaged.
    paths = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ]
    for regular, bold in paths:
        if Path(regular).is_file() and Path(bold).is_file():
            try:
                pdfmetrics.registerFont(TTFont("ArticleRegular", regular))
                pdfmetrics.registerFont(TTFont("ArticleBold", bold))
                pdfmetrics.registerFontFamily("ArticleRegular", normal="ArticleRegular", bold="ArticleBold")
                return "ArticleRegular", "ArticleBold"
            except Exception:
                pass
    return "Helvetica", "Helvetica-Bold"


def _initials(given: str) -> str:
    if not given:
        return ""
    parts = re.findall(r"[A-Za-zÀ-ÿ]+", given)
    return " ".join(p[0].upper() + "." for p in parts)


def apa_author_list(authors: list[Author]) -> str:
    names = [f"{a.family}, {_initials(a.given)}".rstrip() for a in authors if a.family]
    if not names:
        return "Unknown author"
    if len(names) > 20:
        names = names[:19] + [names[-1]]
        return ", ".join(names[:19]) + ", ... " + names[-1]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return names[0] + ", & " + names[1]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def apa_inline(ref: Reference) -> str:
    families = [a.family for a in ref.authors if a.family]
    if not families:
        label = "Unknown author"
    elif len(families) == 1:
        label = families[0]
    elif len(families) == 2:
        label = families[0] + " & " + families[1]
    else:
        label = families[0] + " et al."
    return f"{label}, {ref.year}"


def apa_reference(ref: Reference) -> str:
    authors = escape(apa_author_list(ref.authors))
    title = escape(ref.title.rstrip("."))
    venue = escape(ref.publication_title)
    volume = (", <i>" + escape(ref.volume) + "</i>") if ref.volume else ""
    issue = ("(" + escape(ref.issue) + ")") if ref.issue else ""
    pages = (", " + escape(ref.pages)) if ref.pages else ""
    url = ref.url
    doi_html = (f' <link href="{escape(url, quote=True)}" color="#176a93">'
                f'{escape(url)}</link>') if url.startswith("https://doi.org/") else ""
    return (f"{authors} ({ref.year}). {title}. <i>{venue}</i>"
            f"{volume}{issue}{pages}.{doi_html}")


def in_text_paragraph(p: ArticleParagraph, by_id: dict[str, Reference]) -> str:
    text = p.text.strip().rstrip(". ")
    cited = sorted((by_id[i] for i in p.source_ids), key=lambda r: (r.authors[0].family.lower() if r.authors else "", r.year))
    citation = "; ".join(apa_inline(ref) for ref in cited)
    return f"{text} ({citation})."


def _draw_footer(canvas, doc, mode: str, base_font: str):
    canvas.saveState()
    w, _ = A4
    canvas.setStrokeColor(colors.HexColor("#D6DEE6"))
    canvas.line(20*mm, 17*mm, w-20*mm, 17*mm)
    canvas.setFont(base_font, 8)
    canvas.setFillColor(colors.HexColor("#586A79"))
    canvas.drawString(20*mm, 12*mm, "GROUNDED ARTICLE" if mode == "grounded" else "NO-WEB-SEARCH COMPARISON")
    canvas.drawRightString(w-20*mm, 12*mm, f"Page {doc.page}")
    canvas.restoreState()


def render_pdf(report: Report, path: str | Path) -> None:
    """Call only after Report.model_validate()/Report.model_validate_json()."""
    report = Report.model_validate(report.model_dump())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    font, bold = _fonts()
    dark = colors.HexColor("#18344A")
    accent = colors.HexColor("#1E617E")
    styles = {
        "title": ParagraphStyle("Title_A", fontName=bold, fontSize=20, leading=26,
                                alignment=TA_CENTER, textColor=dark, spaceAfter=15),
        "meta": ParagraphStyle("Meta_A", fontName=font, fontSize=8.5, leading=13,
                               alignment=TA_CENTER, textColor=colors.HexColor("#557083"), spaceAfter=14),
        "heading": ParagraphStyle("Heading_A", fontName=bold, fontSize=12.5, leading=17,
                                  textColor=dark, spaceBefore=16, spaceAfter=6, keepWithNext=True),
        "body": ParagraphStyle("Body_A", fontName=font, fontSize=9.4, leading=15,
                               textColor=dark, alignment=TA_LEFT, spaceAfter=8),
        "abstract": ParagraphStyle("Abstract_A", fontName=font, fontSize=9.2, leading=14.5,
                                   textColor=dark, leftIndent=8*mm, rightIndent=8*mm, spaceAfter=10),
        "reference": ParagraphStyle("Ref_A", fontName=font, fontSize=8.5, leading=13,
                                    textColor=dark, leftIndent=7*mm, firstLineIndent=-7*mm,
                                    spaceAfter=7, splitLongWords=True),
        "warning": ParagraphStyle("Warning_A", fontName=bold, fontSize=8.5, leading=13,
                                  textColor=colors.HexColor("#A34B20"), spaceAfter=10),
    }
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=22*mm,
                            rightMargin=22*mm, topMargin=22*mm,
                            bottomMargin=24*mm, title=report.title, author="HAMK article generator")
    story = [Paragraph(escape(report.title), styles["title"]),
             Paragraph(escape(report.topic), styles["meta"]),
             HRFlowable(width="100%", thickness=0.8, color=accent, spaceAfter=9)]
    if report.mode == "no_web_search":
        story.append(Paragraph("EXPERIMENTAL BASELINE: references below were generated without web search and may be incorrect. See comparison report.", styles["warning"]))
    story += [Paragraph("Abstract", styles["heading"]),
              Paragraph(escape(report.abstract), styles["abstract"]),
              Paragraph("1. Introduction", styles["heading"])]
    by_id = {r.id: r for r in report.references}
    for p in report.introduction:
        story.append(Paragraph(escape(in_text_paragraph(p, by_id)), styles["body"]))
    for n, section in enumerate(report.sections, start=2):
        story.append(Paragraph(f"{n}. {escape(section.heading)}", styles["heading"]))
        for p in section.paragraphs:
            story.append(Paragraph(escape(in_text_paragraph(p, by_id)), styles["body"]))
    story.append(Paragraph(f"{len(report.sections)+2}. Conclusion", styles["heading"]))
    for p in report.conclusion:
        story.append(Paragraph(escape(in_text_paragraph(p, by_id)), styles["body"]))
    story.append(Paragraph("References", styles["heading"]))
    for ref in sorted(report.references, key=lambda r: (r.authors[0].family.lower() if r.authors else "", r.year)):
        story.append(Paragraph(apa_reference(ref), styles["reference"]))
    doc.build(story,
              onFirstPage=lambda c, d: _draw_footer(c, d, report.mode, font),
              onLaterPages=lambda c, d: _draw_footer(c, d, report.mode, font))
