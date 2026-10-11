"""Build the synthetic advert videos used by the video eval cases.

Each video is three still scenes with a synthetic voice-over. In v01 and v02 the
breach is only in the voice-over (the screen shows nothing wrong), so passing them
proves the audio track is reviewed, not just the frames.

Needs macOS `say` and `ffmpeg` on PATH. Run: uv run python eval/make_videos.py
"""

import subprocess
import tempfile
from pathlib import Path

from make_images import font  # same fonts as the image cases
from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "videos"
W, H = 640, 360
SCENE_SECONDS = 3.5

VIDEOS = {
    "v01_crypto_voiceover": {
        "scenes": [
            ("#0b0b0b", "#f7931a", "CRYPTOMAX", "Trade smarter"),
            ("#0b0b0b", "white", "Bitcoin · Ether · More", "All in one app"),
            ("#0b0b0b", "#f7931a", "Download CryptoMax", "Available on iOS and Android"),
        ],
        "voice": "CryptoMax. Guaranteed twenty percent returns, every single month. "
        "Download the app today.",
    },
    "v02_slimming_voiceover": {
        "scenes": [
            ("#f1faee", "#2a9d8f", "SlimTea", "A delicious herbal tea"),
            ("#f1faee", "#1d3557", "Made with green tea", "and natural herbs"),
            ("#f1faee", "#2a9d8f", "SlimTea", "In stores now"),
        ],
        "voice": "SlimTea. Lose ten kilos in one week. No diet and no exercise needed. "
        "In stores now.",
    },
    "v03_beer_compliant": {
        "scenes": [
            ("#2b2d42", "#edf2f4", "BRAND X BEER", "Brewed for grown-ups"),
            ("#2b2d42", "#ffd166", "Crafted in small batches", "since 1998"),
            ("#2b2d42", "#edf2f4", "Please enjoy responsibly", "Not for sale to persons under 18"),
        ],
        "voice": "Brand X beer. Crafted in small batches since nineteen ninety eight. "
        "Please enjoy responsibly.",
    },
}


def scene(path: Path, bg: str, ink: str, title: str, sub: str) -> None:
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    for text, size, y in [(title, 46, 120), (sub, 26, 200)]:
        f = font(size)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=ink)
    img.save(path)


def build(name: str, spec: dict) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inputs = []
        for i, s in enumerate(spec["scenes"]):
            png = tmp / f"s{i}.png"
            scene(png, *s)
            inputs += ["-loop", "1", "-t", str(SCENE_SECONDS), "-i", str(png)]
        voice = tmp / "voice.aiff"
        subprocess.run(["say", "-v", "Samantha", "-o", str(voice), spec["voice"]], check=True)
        n = len(spec["scenes"])
        concat = "".join(f"[{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0,format=yuv420p[v]"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                *inputs,
                "-i",
                str(voice),
                "-filter_complex",
                concat,
                "-map",
                "[v]",
                "-map",
                f"{n}:a",
                "-c:v",
                "libx264",
                "-r",
                "12",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                str(OUT / f"{name}.mp4"),
            ],
            check=True,
        )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, spec in VIDEOS.items():
        build(name, spec)
        print(OUT / f"{name}.mp4")


if __name__ == "__main__":
    main()
