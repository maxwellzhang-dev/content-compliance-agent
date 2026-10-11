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
    audio_seconds: float = 0.0  # length of audio sent for transcription


class Transcriber(Protocol):
    name: str
    usd_per_hour: float  # for the cost estimate; 0 for local models

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

    name = "azure-speech"
    usd_per_hour = 1.00  # standard list price; the F0 tier is free up to 5 h a month

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


class LocalWhisperTranscriber:
    """faster-whisper on the local CPU: no API key, nothing leaves the machine.

    Install with `uv sync --extra local-asr`. The model (default `base`, ~150 MB) is
    downloaded from Hugging Face on first use and cached. Transcription runs in a worker
    thread so it does not block the event loop."""

    usd_per_hour = 0.0

    def __init__(
        self,
        model_size: str = "base",
        language: str | None = "en",
        device: str = "cpu",
        compute_type: str = "int8",
        model=None,  # injectable for tests
    ) -> None:
        self.model_size, self.language = model_size, language
        self.device, self.compute_type = device, compute_type
        self.name = f"local-whisper-{model_size}"
        self._model = model

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def _run(self, wav_16k_mono: bytes) -> str:
        import numpy as np

        with wave.open(io.BytesIO(wav_16k_mono)) as w:
            pcm = w.readframes(w.getnframes())
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self._load().transcribe(audio, language=self.language, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()

    async def transcribe(self, wav_16k_mono: bytes) -> str:
        import asyncio

        return await asyncio.to_thread(self._run, wav_16k_mono)


def local_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def make_transcriber(settings) -> Transcriber | None:
    """Pick the transcriber from TRANSCRIBER: azure | local | none | auto (default).

    auto uses Azure AI Speech when a key is configured, otherwise local faster-whisper
    when it is installed, otherwise none (videos are then reviewed from frames only)."""
    choice = settings.transcriber
    has_azure = bool(settings.speech_key and settings.speech_region)
    if choice == "auto":
        choice = "azure" if has_azure else ("local" if local_whisper_available() else "none")
    if choice == "azure":
        if not has_azure:
            raise ValueError("TRANSCRIBER=azure needs AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")
        return AzureSpeechTranscriber(
            settings.speech_key, settings.speech_region, settings.speech_language
        )
    if choice == "local":
        if not local_whisper_available():
            raise ValueError("TRANSCRIBER=local needs faster-whisper: uv sync --extra local-asr")
        lang = settings.speech_language
        return LocalWhisperTranscriber(
            model_size=settings.local_whisper_model,
            language=None if lang == "auto" else lang.split("-")[0].lower(),
        )
    if choice == "none":
        return None
    raise ValueError(
        f"unknown TRANSCRIBER {settings.transcriber!r}: use azure, local, none or auto"
    )


async def gather_evidence(path: Path | str, transcriber: Transcriber | None) -> VideoEvidence:
    frames, duration = extract_frames(path)
    if not frames:
        raise ValueError("no frames could be decoded from the video")
    audio = extract_audio(path)
    transcript, audio_seconds = None, 0.0
    if audio is not None and transcriber is not None:
        transcript = (await transcriber.transcribe(audio)).strip()
        with wave.open(io.BytesIO(audio)) as w:
            audio_seconds = w.getnframes() / w.getframerate()
    return VideoEvidence(
        frames=frames, transcript=transcript, duration=duration, audio_seconds=audio_seconds
    )


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
