"""HTTP API.

* `/v1/models` and `/v1/chat/completions` speak the OpenAI chat protocol, so Open WebUI
  (or any OpenAI-compatible client) can use the agent as if it were a model.
* `/api/review` returns the full structured review as JSON.
* `/api/reviews` lists stored reviews; `/api/reviews/{id}/flag` pulls one into the human
  queue and `/api/reviews/{id}/resolve` records the reviewer's decision.
* `/api/review-video` reviews an uploaded video (frames + transcribed audio); a video
  URL pasted into chat is fetched and reviewed the same way.
* `/submit` is an upload page for an image or video, for clients that cannot attach one.
* `/stats` (and `/api/stats`) shows latency percentiles and estimated cost.
* `/review` is the queue as a page with approve / override forms. It has no login and is
  meant for local use (docker compose binds it to 127.0.0.1).
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import parse_qs

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from .agent import ComplianceAgent
from .config import Settings, make_async_client
from .judge import OpenAIJudge
from .policies import load_corpus
from .render import render_markdown
from .review_page import render_detail, render_queue, render_stats, render_submit
from .schemas import Verdict
from .store import ReviewStore, stats
from .video import MAX_DOWNLOAD_BYTES, VIDEO_SUFFIXES, AzureSpeechTranscriber, download_video

VIDEO_URL = re.compile(
    r"https?://\S+?(?:" + "|".join(re.escape(x) for x in VIDEO_SUFFIXES) + r")(?:\?\S*)?",
    re.IGNORECASE,
)

MODEL_ID = "compliance-agent"


class ReviewRequest(BaseModel):
    scenario: str = ""
    images: list[str] = []  # http(s) or data: URLs


class FlagRequest(BaseModel):
    reason: str = ""


class ResolveRequest(BaseModel):
    approve: bool
    verdict: Verdict | None = None  # required when overriding
    clauses: list[str] = []
    note: str = ""


def build_agent(settings: Settings) -> ComplianceAgent:
    corpus = load_corpus()
    if os.getenv("RETRIEVER", "lightrag") == "offline":
        from .embeddings import HashingEmbedder
        from .retrieval import OfflineHybridRetriever

        retriever = OfflineHybridRetriever(corpus, HashingEmbedder())
    else:
        from .lightrag_store import LightRAGRetriever

        retriever = LightRAGRetriever(corpus, settings)
    judge = OpenAIJudge(make_async_client(settings), settings)
    transcriber = (
        AzureSpeechTranscriber(
            settings.speech_key, settings.speech_region, settings.speech_language
        )
        if settings.speech_key and settings.speech_region
        else None
    )
    return ComplianceAgent(
        retriever, judge, settings, ReviewStore(settings.review_db_path), transcriber
    )


def create_app(agent: ComplianceAgent | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.agent = agent or build_agent(settings)
        await app.state.agent.retriever.start()
        yield
        await app.state.agent.retriever.stop()

    app = FastAPI(title="Content Compliance Agent", lifespan=lifespan)

    def auth(request: Request) -> None:
        if (
            settings.api_key
            and request.headers.get("authorization") != f"Bearer {settings.api_key}"
        ):
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(auth)])
    async def models():
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(auth)])
    async def chat(request: Request):
        body = await request.json()
        user_turns = [m for m in body.get("messages", []) if m.get("role") == "user"]
        if not user_turns:
            raise HTTPException(status_code=400, detail="no user message")
        content_in = user_turns[-1].get("content")
        scenario, images = _text(content_in), _images(content_in)
        if scenario.lstrip().startswith("### Task:"):
            # Open WebUI background tasks (titles, tags, follow-ups) are not scenarios.
            content = ""
        else:
            if not scenario.strip() and not images:
                raise HTTPException(
                    status_code=400, detail="describe the content or attach an image"
                )
            match = VIDEO_URL.search(scenario)
            if match:
                note = (scenario[: match.start()] + scenario[match.end() :]).strip()
                try:
                    path = await download_video(match.group(0), settings.allow_private_video_urls)
                except (ValueError, OSError) as exc:
                    content = f"Could not fetch the video: {exc}"
                else:
                    try:
                        review = await request.app.state.agent.review(note, video=path)
                    finally:
                        path.unlink(missing_ok=True)
                    content = render_markdown(review, settings.review_ui_url)
            else:
                review = await request.app.state.agent.review(scenario, images)
                content = render_markdown(review, settings.review_ui_url)
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        if body.get("stream"):
            return StreamingResponse(_sse(cid, content), media_type="text/event-stream")
        return {
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    @app.post("/api/review", dependencies=[Depends(auth)])
    async def review(req: ReviewRequest, request: Request):
        if not req.scenario.strip() and not req.images:
            raise HTTPException(status_code=400, detail="scenario or images required")
        review = await request.app.state.agent.review(req.scenario, req.images)
        return review.model_dump(mode="json")

    async def review_upload(request: Request, upload: UploadFile, note: str):
        data = await upload.read(MAX_DOWNLOAD_BYTES + 1)
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file is larger than 50 MB")
        ctype = upload.content_type or ""
        suffix = Path(upload.filename or "").suffix.lower()
        if ctype.startswith("image/"):
            url = f"data:{ctype};base64,{base64.b64encode(data).decode()}"
            return await request.app.state.agent.review(note, [url])
        if ctype.startswith("video/") or suffix in VIDEO_SUFFIXES:
            with tempfile.NamedTemporaryFile(suffix=suffix or ".mp4", delete=False) as tmp:
                tmp.write(data)
            try:
                return await request.app.state.agent.review(note, video=tmp.name)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        raise HTTPException(status_code=415, detail="upload an image or a video")

    @app.post("/api/review-video", dependencies=[Depends(auth)])
    async def review_video(
        request: Request, file: Annotated[UploadFile, File()], scenario: Annotated[str, Form()] = ""
    ):
        return (await review_upload(request, file, scenario)).model_dump(mode="json")

    @app.get("/submit", response_class=HTMLResponse)
    async def submit_page():
        return render_submit()

    @app.post("/submit")
    async def submit(
        request: Request, file: Annotated[UploadFile, File()], note: Annotated[str, Form()] = ""
    ):
        review = await review_upload(request, file, note)
        return RedirectResponse(f"/review/{review.id}", status_code=303)

    @app.get("/review/{review_id}", response_class=HTMLResponse)
    async def review_detail(review_id: int, request: Request):
        rec = store(request).get(review_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="no such review")
        return render_detail(rec, request.app.state.agent.policy_versions)

    def store(request: Request) -> ReviewStore:
        st = request.app.state.agent.store
        if st is None:
            raise HTTPException(status_code=404, detail="review storage is not configured")
        return st

    @app.get("/api/reviews", dependencies=[Depends(auth)])
    async def list_reviews(request: Request, status: str | None = None, limit: int = 50):
        return store(request).list(status, limit)

    @app.get("/api/reviews/{review_id}", dependencies=[Depends(auth)])
    async def get_review(review_id: int, request: Request):
        rec = store(request).get(review_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="no such review")
        return rec

    @app.post("/api/reviews/{review_id}/flag", dependencies=[Depends(auth)])
    async def flag_review(review_id: int, req: FlagRequest, request: Request):
        if not store(request).flag(review_id, req.reason):
            raise HTTPException(status_code=409, detail="review is missing or already resolved")
        return {"status": "pending"}

    @app.post("/api/reviews/{review_id}/resolve", dependencies=[Depends(auth)])
    async def resolve_review(review_id: int, req: ResolveRequest, request: Request):
        return _resolve(store(request), review_id, req)

    @app.get("/api/stats", dependencies=[Depends(auth)])
    async def api_stats(request: Request, limit: int = 1000):
        return stats(store(request), limit)

    @app.get("/stats", response_class=HTMLResponse)
    async def stats_page(request: Request, limit: int = 1000):
        return render_stats(stats(store(request), limit))

    @app.get("/review", response_class=HTMLResponse)
    async def review_page(request: Request, tab: str = "pending", limit: int = 50):
        st = store(request)
        if tab == "resolved":
            records = sorted(
                st.list("approved", limit) + st.list("overridden", limit),
                key=lambda r: r["id"],
                reverse=True,
            )[:limit]
        else:
            tab = tab if tab in ("pending", "auto") else "pending"
            records = st.list(tab, limit)
        return render_queue(records, tab, st.counts(), request.app.state.agent.policy_versions)

    @app.post("/review/{review_id}")
    async def review_form(review_id: int, request: Request):
        form = {k: v[0] for k, v in parse_qs((await request.body()).decode()).items()}
        approve = form.get("action") == "approve"
        req = ResolveRequest(
            approve=approve,
            verdict=None if approve else form.get("verdict"),
            clauses=[c.strip().upper() for c in form.get("clauses", "").split(",") if c.strip()],
            note=form.get("note", ""),
        )
        _resolve(store(request), review_id, req)
        return RedirectResponse("/review?tab=pending", status_code=303)

    return app


def _resolve(st: ReviewStore, review_id: int, req: ResolveRequest) -> dict:
    try:
        ok = st.resolve(
            review_id, approve=req.approve, verdict=req.verdict, clauses=req.clauses, note=req.note
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(status_code=409, detail="review is missing or not waiting for review")
    return {"status": "approved" if req.approve else "overridden"}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def _images(content) -> list[str]:
    if not isinstance(content, list):
        return []
    urls = []
    for part in content:
        if part.get("type") == "image_url":
            ref = part.get("image_url")
            url = ref.get("url") if isinstance(ref, dict) else ref
            if isinstance(url, str) and url.startswith(("data:image/", "https://", "http://")):
                urls.append(url)
    return urls


async def _sse(cid: str, content: str):
    def chunk(delta: dict, finish: str | None = None) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield chunk({"role": "assistant"})
    for i in range(0, len(content), 400):
        yield chunk({"content": content[i : i + 400]})
    yield chunk({}, "stop")
    yield "data: [DONE]\n\n"
