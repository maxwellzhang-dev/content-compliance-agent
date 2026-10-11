import asyncio
import io
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from compliance_agent import video as v
from compliance_agent.agent import ComplianceAgent
from compliance_agent.api import VIDEO_URL, create_app
from compliance_agent.render import render_markdown
from tests.conftest import FakeJudge

SAMPLE = Path(__file__).resolve().parents[1] / "eval" / "videos" / "v01_crypto_voiceover.mp4"


class FakeTranscriber:
    def __init__(self):
        self.calls = 0

    async def transcribe(self, wav):
        self.calls += 1
        with wave.open(io.BytesIO(wav)) as w:
            assert w.getframerate() == 16_000 and w.getnchannels() == 1
        return "Guaranteed twenty percent returns every month."


def test_frames_are_evenly_sampled_jpegs():
    frames, duration = v.extract_frames(SAMPLE)
    assert len(frames) == v.MAX_FRAMES and 9 < duration < 12
    assert all(f.startswith("data:image/jpeg;base64,") for f in frames)


def test_audio_is_16k_mono_wav():
    wav = v.extract_audio(SAMPLE)
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getframerate() == 16_000 and w.getnchannels() == 1
        assert 4 < w.getnframes() / 16_000 < 12  # voice-over is shorter than the picture


def test_long_audio_is_split_into_chunks_under_a_minute():
    pcm = b"\0\0" * 16_000 * 130  # 130 s of silence
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(pcm)
    chunks = v.split_wav(buf.getvalue())
    assert len(chunks) == 3


def test_video_review_uses_frames_and_transcript(retriever, settings):
    judge, tx = FakeJudge(), FakeTranscriber()
    agent = ComplianceAgent(retriever, judge, settings, transcriber=tx)
    review = asyncio.run(agent.review("Business news break?", video=SAMPLE))
    assert judge.observed.is_video and len(judge.observed.images) == v.MAX_FRAMES
    assert "Guaranteed twenty percent" in judge.decided.query_text()
    assert review.media == "video" and tx.calls == 1
    text = render_markdown(review)
    assert "What the agent saw and heard" in text and "Voice-over:* “Guaranteed" in text


def test_video_without_transcriber_says_so(retriever, settings):
    agent = ComplianceAgent(retriever, FakeJudge(), settings)
    review = asyncio.run(agent.review("", video=SAMPLE))
    assert review.transcript is None
    assert "no speech recognised" in render_markdown(review)


def test_video_url_detection():
    m = VIDEO_URL.search("Check this https://cdn.example.com/ads/spot.MP4?x=1 please")
    assert m and m.group(0) == "https://cdn.example.com/ads/spot.MP4?x=1"
    assert VIDEO_URL.search("see https://example.com/page.html") is None


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/a.mp4",
        "http://127.0.0.1/a.mp4",
        "file:///etc/a.mp4",
        "ftp://example.com/a.mp4",
    ],
)
def test_download_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        asyncio.run(v.download_video(url))


def test_upload_endpoint_and_detail_page(retriever, settings):
    from compliance_agent.store import ReviewStore

    agent = ComplianceAgent(
        retriever, FakeJudge(), settings, ReviewStore(settings.review_db_path), FakeTranscriber()
    )
    with TestClient(create_app(agent=agent, settings=settings)) as c:
        with SAMPLE.open("rb") as f:
            r = c.post(
                "/api/review-video",
                files={"file": ("spot.mp4", f, "video/mp4")},
                data={"scenario": "Business news break?"},
            )
        assert r.status_code == 200, r.text
        assert r.json()["media"] == "video"
        page = c.get(f"/review/{r.json()['id']}").text
        assert "Frames sampled" in page and "Voice-over" in page
        bad = c.post("/api/review-video", files={"file": ("x.txt", b"hi", "text/plain")})
        assert bad.status_code == 415


def test_submit_page_redirects_to_the_review(retriever, settings):
    from compliance_agent.store import ReviewStore

    agent = ComplianceAgent(retriever, FakeJudge(), settings, ReviewStore(settings.review_db_path))
    png = Path(__file__).resolve().parents[1] / "eval" / "images" / "beer_cartoon.png"
    with TestClient(create_app(agent=agent, settings=settings)) as c:
        assert "Submit content for review" in c.get("/submit").text
        with png.open("rb") as f:
            r = c.post(
                "/submit",
                files={"file": ("a.png", f, "image/png")},
                data={"note": "slot?"},
                follow_redirects=False,
            )
    assert r.status_code == 303 and r.headers["location"] == "/review/1"
