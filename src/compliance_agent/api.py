"""HTTP API.

* `/v1/models` and `/v1/chat/completions` speak the OpenAI chat protocol, so Open WebUI
  (or any OpenAI-compatible client) can use the agent as if it were a model.
* `/api/review` returns the full structured review as JSON.
* `/api/reviews` lists stored reviews; `/api/reviews/{id}/flag` pulls one into the human
  queue and `/api/reviews/{id}/resolve` records the reviewer's decision.
* `/review` is the queue as a page with approve / override forms. It has no login and is
  meant for local use (docker compose binds it to 127.0.0.1).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from urllib.parse import parse_qs

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from .agent import ComplianceAgent
from .config import Settings, make_async_client
from .judge import OpenAIJudge
from .policies import load_corpus
from .render import render_markdown
from .review_page import render_queue
from .schemas import Verdict
from .store import ReviewStore

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
    return ComplianceAgent(retriever, judge, settings, ReviewStore(settings.review_db_path))


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
        return render_queue(records, tab, st.counts())

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
