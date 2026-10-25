"""Assignment 12: free cloud music generation via a public Hugging Face Space.

The remote Space runs ACE-Step Model 1.0. No local model weights or GPU needed.
Public ZeroGPU use is free but quota-limited; availability is not guaranteed.
"""

import os
import shutil
import wave
from pathlib import Path
from urllib.parse import urlparse

from gradio_client import Client

SPACE = "ACE-Step/ACE-Step"  # Official public model demo (ACE-Step 1.0)
DURATION_SECONDS = 180  # 3 minutes
HERE = Path(__file__).resolve().parent


def find_audio_path(result):
    """Locate the audio returned by the Gradio API."""
    audio = result[0] if isinstance(result, (tuple, list)) else result
    if isinstance(audio, dict):
        audio = audio.get("path") or audio.get("url") or audio.get("name")
    if not isinstance(audio, str) or not audio:
        raise ValueError(f"Unexpected audio response: {type(audio).__name__} {audio!r}")
    return audio


def main():
    prompt = (HERE / "prompt.txt").read_text(encoding="utf-8").strip()
    lyrics = (HERE / "lyrics.txt").read_text(encoding="utf-8").strip()

    # Use a new free Hugging Face read token if available. Never put tokens in code.
    token = os.getenv("HF_TOKEN") or None
    print(f"Connecting to free cloud Space: {SPACE}")
    client = Client(SPACE, token=token)
    print("Requesting 180 seconds of music with English vocals...")

    # The keywords match the official demo's text2music Gradio API.
    result = client.predict(
        audio_duration=float(DURATION_SECONDS),
        prompt=prompt,
        lyrics=lyrics,
        infer_step=60,
        guidance_scale=15.0,
        scheduler_type="euler",
        cfg_type="apg",
        omega_scale=10.0,
        manual_seeds=None,
        guidance_interval=0.5,
        guidance_interval_decay=0.0,
        min_guidance_scale=3.0,
        use_erg_tag=True,
        use_erg_lyric=False,
        use_erg_diffusion=True,
        oss_steps=None,
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
        audio2audio_enable=False,
        ref_audio_strength=0.5,
        ref_audio_input=None,
        lora_name_or_path="none",
        api_name="/__call__",
    )

    source = find_audio_path(result)
    if source.startswith(("https://", "http://")):
        # Usually Gradio downloads the audio for you, but handle URL output too.
        import requests
        suffix = Path(urlparse(source).path).suffix or ".wav"
        out = HERE / f"generated_song{suffix}"
        response = requests.get(source, timeout=180)
        response.raise_for_status()
        out.write_bytes(response.content)
    else:
        src = Path(source)
        if not src.is_file():
            raise FileNotFoundError(f"Returned audio file not found: {src}")
        out = HERE / f"generated_song{src.suffix or '.wav'}"
        if src.resolve() != out.resolve():
            shutil.copy2(src, out)

    print(f"Saved song to: {out}")
    if out.suffix.lower() == ".wav":
        try:
            with wave.open(str(out), "rb") as f:
                actual_seconds = f.getnframes() / f.getframerate()
            print(f"Actual WAV duration: {actual_seconds:.1f} seconds")
        except (wave.Error, EOFError):
            print("Audio saved, but duration could not be read with the WAV parser.")
    print("Listen to the audio and check lyrics/duration before submission.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Generation failed: {type(exc).__name__}: {exc}")
        print("Check the Hugging Face Space page, API tab and daily ZeroGPU quota.")
        raise SystemExit(1)
