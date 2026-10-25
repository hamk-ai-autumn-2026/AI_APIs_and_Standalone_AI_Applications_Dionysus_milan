"""Run: python -m unittest discover -s tests -v (NO API key needed)."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock

from pydantic import ValidationError

from comparison import assess_memory_reference, evaluate_memory_references
from generate_article import draft_to_report, arguments
from models import Author, ArticleParagraph, ArticleSection, MemoryReference, Report, ReportDraft, Reference
from pdf_renderer import apa_author_list, apa_inline, in_text_paragraph, render_pdf
from sources import extract_dois, reference_from_crossref, run_web_search


def refs():
    out = []
    for i, last in enumerate(("Smith", "Chen", "Johnson"), 1):
        out.append(Reference(id=f"S{i}", authors=[Author(given="Ann", family=last)],
                             year=2020+i, title="Paper title " + str(i),
                             publication_title="Journal of Robotics",
                             doi=f"10.1234/paper{i}", url=f"https://doi.org/10.1234/paper{i}",
                             verification_url=f"https://api.crossref.org/works/10.1234%2Fpaper{i}",
                             verified=True))
    return out


def draft():
    return ReportDraft(title="Agricultural robot perception and control",
        abstract="Agricultural robots require sensing, navigation and reliable control in changing environments. This technical discussion examines research on perception, system integration and limitations of autonomous field systems.",
        introduction=[ArticleParagraph(text="Agricultural robots perform complex tasks in variable environments. Sensor feedback is needed for navigation and safe field motion.", source_ids=["S1"])],
        sections=[
            ArticleSection(heading="Perception", paragraphs=[ArticleParagraph(text="Imaging provides information for locating obstacles and rows. Real-world image quality depends on changing illumination and occlusion.", source_ids=["S2"])]),
            ArticleSection(heading="Control", paragraphs=[ArticleParagraph(text="Controllers translate navigation decisions into mechanical motion, but uncertainty from field terrain and sensor drift must be considered.", source_ids=["S3"])])],
        conclusion=[ArticleParagraph(text="The studies outline a need for robust integrated sensing and control, with evaluations beyond controlled laboratory settings.", source_ids=["S1", "S2", "S3"])])


class TestModels(unittest.TestCase):
    def test_valid_report(self):
        self.assertEqual(len(draft_to_report(draft(), "Agricultural robotics", refs(), "grounded").references), 3)

    def test_rejects_unknown_citation(self):
        d = draft()
        d.introduction[0].source_ids = ["S9"]
        with self.assertRaises(ValidationError):
            draft_to_report(d, "Agricultural robotics", refs(), "grounded")

    def test_rejects_unverified_grounded(self):
        r = refs()
        r[0].verified = False
        with self.assertRaises(ValidationError):
            draft_to_report(draft(), "Agricultural robotics", r, "grounded")

    def test_rejects_missing_section(self):
        d = draft()
        d.sections = d.sections[:1]
        with self.assertRaises(ValidationError):
            draft_to_report(d, "Agricultural robotics", refs(), "grounded")

    def test_rejects_bibliography_not_cited(self):
        d = draft()
        d.conclusion[0].source_ids = ["S1"]
        d.sections[1].paragraphs[0].source_ids = ["S1"]
        with self.assertRaises(ValidationError):
            draft_to_report(d, "Agricultural robotics", refs(), "grounded")

    def test_cli_validation(self):
        with self.assertRaises(SystemExit):
            arguments(["--topic", "AI", "--source-count", "99"])


class TestSearch(unittest.TestCase):
    def test_doi_extraction_and_case(self):
        s = "See https://doi.org/10.1234/ABC. Also DOI:10.5555/xyz). 10.1234/abc"
        self.assertEqual(extract_dois(s), ["10.1234/abc", "10.5555/xyz"])

    def test_web_search_must_have_tool_call(self):
        c = MagicMock()
        output = MagicMock()
        output.output_text = "10.1234/paper"
        output.model_dump.return_value = {"output": [{"type": "message", "content": []}]}
        c.responses.create.return_value = output
        with self.assertRaises(RuntimeError):
            run_web_search(c, "agricultural robotics", "gpt-5.5")

    def test_web_search_collects_actual_results(self):
        c = MagicMock()
        output = MagicMock()
        output.output_text = "Relevant study: DOI 10.5555/FOO."
        output.model_dump.return_value = {"output": [
            {"type": "web_search_call", "action": {"sources": [{"url": "https://doi.org/10.1234/BAR"}]}},
            {"type": "message", "content": [{"annotations": [{"type": "url_citation", "url": "https://doi.org/10.6543/baz"}]}]},
        ]}
        c.responses.create.return_value = output
        result = run_web_search(c, "agricultural robotics", "gpt-5.5")
        self.assertEqual(set(result["candidate_dois"]), {"10.5555/foo", "10.1234/bar", "10.6543/baz"})
        self.assertTrue(result["searched"])

    def test_crossref_metadata_author_year(self):
        meta = {"DOI": "10.1000/test", "title": ["An article"],
                "container-title": ["A Journal"], "published": {"date-parts": [[2024, 3, 1]]},
                "author": [{"given": "Alice B", "family": "Smith"}],
                "abstract": "<p>Some science.</p>", "volume": "22", "issue": "3"}
        r = reference_from_crossref(meta, "S1", "test search")
        self.assertTrue(r.verified)
        self.assertEqual(r.year, 2024)
        self.assertEqual(r.authors[0].family, "Smith")
        self.assertEqual(r.abstract, "Some science.")


class TestFormattingAndEvaluation(unittest.TestCase):
    def test_apa_authors(self):
        a = [Author(given="Alice B", family="Smith"), Author(given="David", family="Brown")]
        self.assertIn("Smith, A. B., & Brown, D.", apa_author_list(a))
        self.assertEqual(apa_inline(Reference(id="S1", authors=a, year=2024, title="T", publication_title="J", doi="10.1234/x", url="https://doi.org/10.1234/x", verified=True)), "Smith & Brown, 2024")

    def test_apa_inline_insertion(self):
        text = in_text_paragraph(ArticleParagraph(text="Some information.", source_ids=["S1"]), {"S1": refs()[0]})
        self.assertEqual(text, "Some information (Smith, 2021).")

    def test_unverified_does_not_mean_hallucination(self):
        r = MemoryReference(id="R1", authors=["Smith, A"], year=2020, title="Title",
                            publication_title="Journal", doi_or_url="https://example.org/x")
        d = assess_memory_reference(r, MagicMock())
        self.assertEqual(d["status"], "unverified")

    def test_false_metadata_detected(self):
        r = MemoryReference(id="R1", authors=["Fake, J"], year=2010, title="Wrong Title",
                            publication_title="Wrong Journal", doi_or_url="10.1234/paper")
        import comparison
        from unittest.mock import patch
        actual = {"DOI": "10.1234/paper", "title": ["Actual real title"],
                  "container-title": ["Real Journal"], "published": {"date-parts": [[2025]]},
                  "author": [{"given": "Alice", "family": "Smith"}]}
        with patch.object(comparison, "lookup_doi", return_value=actual):
            outcome = assess_memory_reference(r, MagicMock())
        self.assertEqual(outcome["status"], "incorrect")
        self.assertFalse(outcome["checks"]["publication_year"])
        self.assertFalse(outcome["checks"]["first_author"])

    def test_pdf_render_writes_file(self):
        report = draft_to_report(draft(), "Agricultural robotics", refs(), "grounded")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "article.pdf"
            render_pdf(report, p)
            self.assertTrue(p.is_file())
            self.assertGreater(p.stat().st_size, 2200)
            self.assertEqual(p.read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
