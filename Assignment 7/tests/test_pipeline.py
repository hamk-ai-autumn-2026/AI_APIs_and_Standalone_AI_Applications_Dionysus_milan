"""Offline tests: no OpenAI API key or paid requests are used."""

import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from main import encode_source_image, run_pipeline


class FakeResponses:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text="A small green robot waters a flower in a sunny garden.")


class FakeImages:
    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        image = Image.new("RGB", (4, 4), "green")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        return SimpleNamespace(data=[SimpleNamespace(b64_json=encoded)])


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.input_path = self.folder / "input.png"
        Image.new("RGB", (12, 12), "purple").save(self.input_path)
        self.responses = FakeResponses()
        self.images = FakeImages()
        self.client = SimpleNamespace(responses=self.responses, images=self.images)

    def test_image_encoding(self):
        value = encode_source_image(self.input_path)
        self.assertTrue(value.startswith("data:image/png;base64,"))
        self.assertGreater(len(value), 40)

    def test_pipeline_is_image_to_text_to_image(self):
        out = self.folder / "generated.png"
        desc = self.folder / "description.txt"
        description = run_pipeline(
            self.client, self.input_path, out, desc,
            "gpt-5.6-luna", "gpt-image-2", "1024x1024", "low"
        )
        vision_content = self.responses.kwargs["input"][0]["content"]
        self.assertEqual(vision_content[1]["type"], "input_image")
        self.assertTrue(vision_content[1]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(self.images.kwargs["prompt"], description)
        self.assertNotIn("image", self.images.kwargs)
        self.assertNotIn("input_image", self.images.kwargs)
        self.assertNotIn("source_data_url", self.images.kwargs)
        self.assertTrue(out.exists())
        self.assertEqual(desc.read_text(encoding="utf-8").strip(), description)
        with Image.open(out) as generated:
            self.assertEqual(generated.format, "PNG")

    def test_missing_input(self):
        with self.assertRaises(FileNotFoundError):
            encode_source_image(self.folder / "missing.png")

    def test_cannot_overwrite_input(self):
        with self.assertRaisesRegex(ValueError, "different paths"):
            run_pipeline(
                self.client, self.input_path, self.input_path, self.folder / "desc.txt",
                "gpt-5.6-luna", "gpt-image-2", "1024x1024", "low"
            )


if __name__ == "__main__":
    unittest.main()
