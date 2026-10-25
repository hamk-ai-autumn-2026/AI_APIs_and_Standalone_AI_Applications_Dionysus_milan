"""Offline tests: run python -m unittest discover -s tests -v."""
import base64
from io import BytesIO
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

import generate


def sample_png():
    data = BytesIO()
    Image.new("RGB", (3, 2), "red").save(data, format="PNG")
    return data.getvalue()


class CliTests(unittest.TestCase):
    def parse(self, *args):
        parser = generate.build_parser()
        settings = parser.parse_args(["--prompt", "a small robot", *args])
        generate.validate_args(settings, parser)
        return settings

    def test_required_arguments(self):
        parser = generate.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        for args in [("--count", "0"), ("--count", "11"), ("--aspect-ratio", "7:2"),
                     ("--seed", "123"), ("--negative-prompt", "fog")]:
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.parse(*args)

    def test_capability_validation(self):
        self.parse("--provider", "openai", "--quality", "medium")
        self.parse("--provider", "fal", "--seed", "1", "--negative-prompt", "blur")
        self.parse("--provider", "replicate", "--seed", "0")
        with self.assertRaises(SystemExit):
            self.parse("--provider", "replicate", "--quality", "high")
        with self.assertRaises(SystemExit):
            self.parse("--provider", "fal", "--seed", str(2**32 - 1), "--count", "2")

    def test_safe_slug(self):
        self.assertEqual(generate.slugify("../../ Finnish lake :\\ boat!!!"), "finnish-lake-boat")
        self.assertEqual(generate.slugify("🏔️🌸"), "image")
        self.assertLessEqual(len(generate.slugify("a" * 1000)), 48)

    def test_base64_and_data_uri(self):
        raw = sample_png()
        encoded = base64.b64encode(raw).decode("ascii")
        self.assertEqual(generate.decode_base64_image(encoded), raw)
        self.assertEqual(generate.image_to_bytes("data:image/png;base64," + encoded), raw)
        with self.assertRaises(generate.GenerationError):
            generate.decode_base64_image("not base64 %%%")

    def test_temporary_https_url_download(self):
        class FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def raise_for_status(self):
                pass
            def iter_bytes(self):
                yield sample_png()[:10]
                yield sample_png()[10:]
        with patch("httpx.stream", return_value=FakeResponse()) as mocked:
            data = generate.image_to_bytes("https://images.example.org/temporary.png")
        self.assertEqual(data, sample_png())
        self.assertTrue(mocked.call_args.kwargs["follow_redirects"])
        with self.assertRaises(generate.GenerationError):
            generate.image_to_bytes("http://example.org/untrusted.png")

    def test_network_error_is_reported(self):
        import httpx
        with patch("httpx.stream", side_effect=httpx.ConnectError("cannot connect")):
            with self.assertRaisesRegex(generate.GenerationError, "Image download failed"):
                generate.download_url("https://images.example.org/temporary.png")

    def test_jpeg_extension_matches_content(self):
        bio = BytesIO()
        Image.new("RGB", (3, 2), "blue").save(bio, format="JPEG")
        with tempfile.TemporaryDirectory() as tmp:
            saved = generate.save_image(bio.getvalue(), Path(tmp), "test", "fal", 1)
            self.assertEqual(saved.suffix, ".jpg")
            self.assertTrue(saved.exists())

    def test_file_object(self):
        class MockFileOutput:
            def read(self):
                return sample_png()
        self.assertEqual(generate.image_to_bytes(MockFileOutput()), sample_png())

    def test_unique_filename_and_real_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            first = generate.save_image(sample_png(), output, "../../ test image", "fal", 1)
            second = generate.save_image(sample_png(), output, "../../ test image", "fal", 1)
            self.assertNotEqual(first, second)
            self.assertEqual(first.suffix, ".png")
            self.assertTrue(first.name.startswith("test-image_fal_01_"))
            self.assertEqual(first.read_bytes(), sample_png())
            with self.assertRaises(generate.GenerationError):
                generate.save_image(b"<html>error</html>", output, "test", "fal", 2)

    def test_openai_base64_adapter(self):
        fake_response = types.SimpleNamespace(data=[types.SimpleNamespace(
            b64_json=base64.b64encode(sample_png()).decode("ascii"))])
        calls = []
        class MockOpenAI:
            def __init__(self, **kwargs):
                self.images = types.SimpleNamespace(generate=self.generate)
            def generate(self, **kwargs):
                calls.append(kwargs)
                return fake_response
        with patch.dict("sys.modules", {"openai": types.SimpleNamespace(OpenAI=MockOpenAI)}):
            args = self.parse("--quality", "low", "--aspect-ratio", "16:9")
            self.assertEqual(generate.generate_openai(args, None), sample_png())
        self.assertEqual(calls[0]["model"], "gpt-image-2")
        self.assertEqual(calls[0]["size"], "1536x864")
        self.assertEqual(calls[0]["quality"], "low")
        self.assertNotIn("response_format", calls[0])

    def test_fal_adapter(self):
        calls = []
        def subscribe(model, arguments):
            calls.append((model, arguments))
            return {"images": [{"url": "data:image/png;base64," +
                    base64.b64encode(sample_png()).decode()}]}
        with patch.dict("sys.modules", {"fal_client": types.SimpleNamespace(subscribe=subscribe)}):
            args = self.parse("--provider", "fal", "--negative-prompt", "text", "--seed", "12")
            self.assertEqual(generate.generate_fal(args, 12), sample_png())
        self.assertEqual(calls[0][0], "fal-ai/flux-general")
        self.assertEqual(calls[0][1]["negative_prompt"], "text")
        self.assertEqual(calls[0][1]["seed"], 12)

    def test_replicate_adapter_file_output(self):
        class FileOutput:
            def read(self):
                return sample_png()
        calls = []
        def run(model, input):
            calls.append((model, input))
            return [FileOutput()]
        with patch.dict("sys.modules", {"replicate": types.SimpleNamespace(run=run)}):
            args = self.parse("--provider", "replicate", "--seed", "17")
            self.assertEqual(generate.generate_replicate(args, 17), sample_png())
        self.assertEqual(calls[0][1]["seed"], 17)
        self.assertEqual(calls[0][1]["num_outputs"], 1)

    def test_three_images_are_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("generate.check_key"), patch("generate.generate_image", return_value=sample_png()) as m:
                result = generate.main(["--prompt", "robots in a greenhouse", "--count", "3", "--output", tmp])
            self.assertEqual(result, 0)
            self.assertEqual(m.call_count, 3)
            self.assertEqual(len(list(Path(tmp).glob("*.png"))), 3)

    def test_api_failure_keeps_earlier_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("generate.check_key"), patch("generate.generate_image",
                side_effect=[sample_png(), TimeoutError("connection dropped")]):
                result = generate.main(["--prompt", "robot", "--count", "2", "--output", tmp])
            self.assertEqual(result, 1)
            self.assertEqual(len(list(Path(tmp).glob("*.png"))), 1)

    def test_invalid_output_folder_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "file.txt"
            file_path.write_text("exists")
            parser = generate.build_parser()
            with self.assertRaises(SystemExit):
                settings = parser.parse_args(["--prompt", "photo", "--output", str(file_path)])
                generate.validate_args(settings, parser)


if __name__ == "__main__":
    unittest.main()
