"""Video input: sample frames for the vision pass and transcribe the audio track.

Adverts often break rules in the voice-over rather than on screen ("guaranteed
returns"), so both halves go into the review: frames as images, the transcript as text.
"""

from __future__ import annotations

import base64
import io
import ipaddress
import socket
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import av
import httpx
from PIL import Image

MAX_FRAMES = 6
FRAME_EDGE = 768  # px, longest side
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 55  # Azure short-audio REST accepts up to 60 s per request
VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".m4v", ".mkv")


@dataclass
class VideoEvidence:
    frames: list[str]  # data: URLs, in time order
    transcript: str | None  # None when the video has no audio track
    duration: float


class Transcriber(Protocol):
    async def transcribe(self, wav_16k_mono: bytes) -> str: ...


def _to_data_url(img: Image.Image) -> str:
    img.thumbnail((FRAME_EDGE, FRAME_EDGE))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def extract_frames(path: Path | str, max_frames: int = MAX_FRAMES) -> tuple[list[str], float]:
    """Evenly spaced frames across the whole video, so a short spot and a long one are
    covered the same way. Returns the frames and the duration in seconds."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        if duration <= 0 and stream.duration:
            duration = float(stream.duration * stream.time_base)
        targets = [duration * (i + 0.5) / max_frames for i in range(max_frames)]
        frames, t_idx = [], 0
        for frame in container.decode(stream):
            if t_idx >= len(targets):
                break
            t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if t >= targets[t_idx]:
                frames.append(_to_data_url(frame.to_image()))
                t_idx += 1
                while t_idx < len(targets) and t >= targets[t_idx]:
                    t_idx += 1
    return frames, duration


def extract_audio(path: Path | str) -> bytes | None:
    """The audio track as 16 kHz mono 16-bit WAV, or None if there is no audio."""
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        pcm = bytearray()
        for frame in container.decode(container.streams.audio[0]):
            for out in resampler.resample(frame):
                pcm += out.to_ndarray().tobytes()
        for out in resampler.resample(None):  # flush
            pcm += out.to_ndarray().tobytes()
    return _wav(bytes(pcm))


def _wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def split_wav(wav_bytes: bytes, seconds: int = CHUNK_SECONDS) -> list[bytes]:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        pcm = w.readframes(w.getnframes())
    step = seconds * SAMPLE_RATE * 2
    return [_wav(pcm[i : i + step]) for i in range(0, len(pcm), step)] or [_wav(b"")]


class AzureSpeechTranscriber:
    """Azure AI Speech short-audio REST API, one request per 55-second chunk."""

    def __init__(self, key: str, region: str, language: str = "en-US") -> None:
        self.key, self.region, self.language = key, region, language

    async def transcribe(self, wav_16k_mono: bytes) -> str:
        url = (
            f"https://{self.region}.stt.speech.microsoft.com/speech/recognition/"
            f"conversation/cognitiveservices/v1?language={self.language}&format=simple"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
            "Accept": "application/json",
        }
        parts = []
        async with httpx.AsyncClient(timeout=60) as client:
            for chunk in split_wav(wav_16k_mono):
                resp = await client.post(url, headers=headers, content=chunk)
                resp.raise_for_status()
                body = resp.json()
                if body.get("RecognitionStatus") == "Success" and body.get("DisplayText"):
                    parts.append(body["DisplayText"])
        return " ".join(parts)


async def gather_evidence(path: Path | str, transcriber: Transcriber | None) -> VideoEvidence:
    frames, duration = extract_frames(path)
    if not frames:
        raise ValueError("no frames could be decoded from the video")
    audio = extract_audio(path)
    transcript = None
    if audio is not None and transcriber is not None:
        transcript = (await transcriber.transcribe(audio)).strip()
    return VideoEvidence(frames=frames, transcript=transcript, duration=duration)


def _public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return True


async def download_video(url: str, allow_private: bool = False) -> Path:
    """Fetch a video URL to a temp file. Only http(s), only public hosts unless allowed,
    only video content, at most 50 MB."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("only http(s) video URLs are accepted")
    if not allow_private and not _public_host(parsed.hostname):
        raise ValueError("video URL must point to a public host")
    suffix = Path(parsed.path).suffix.lower() or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    size = 0
    async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if not (ctype.startswith("video/") or ctype == "application/octet-stream"):
                raise ValueError(f"URL is not a video (content-type {ctype or 'missing'})")
            async for chunk in resp.aiter_bytes():
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    tmp.close()
                    Path(tmp.name).unlink(missing_ok=True)
                    raise ValueError("video is larger than 50 MB")
                tmp.write(chunk)
    tmp.close()
    return Path(tmp.name)
