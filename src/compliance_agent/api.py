"""HTTP API.

* `/v1/models` and `/v1/chat/completions` speak the OpenAI chat protocol, so Open WebUI
  (or any OpenAI-compatible client) can use the agent as if it were a model.
* `/api/review` returns the full structured review as JSON.
* `/api/review-queue` lists reviews routed to a human, newest first; `/review` shows
  the same queue as a page.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .agent import ComplianceAgent, read_queue
from .config import Settings, make_async_client
from .judge import OpenAIJudge
from .policies import load_corpus
from .render import render_markdown
from .review_page import render_queue

MODEL_ID = "compliance-agent"


class ReviewRequest(BaseModel):
    scenario: str = ""
    images: list[str] = []  # http(s) or data: URLs


def build_agent(settings: Settings) -> ComplianceAgent:
    corpus = load_corpus()
    if os.getenv("RETRIEVER", "lightrag") == "offline":
        from .embeddings import HashingEmbedder
        from .retrieval import OfflineHybridRetriever

        retriever = OfflineHybridRetriever(corpus, HashingEmbedder())
    else:
        from .lightrag_store import LightRAGRetriever

        retriever = LightRAGRetriever(corpus, settings)
    return ComplianceAgent(retriever, OpenAIJudge(make_async_client(settings), settings), settings)


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
            content = render_markdown(review)
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

    @app.get("/api/review-queue", dependencies=[Depends(auth)])
    async def queue(limit: int = 50):
        return read_queue(settings, limit)

    @app.get("/review", response_class=HTMLResponse)
    async def review_page(limit: int = 50):
        return render_queue(read_queue(settings, limit))

    return app


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
