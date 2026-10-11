"""Per-review cost and latency.

Each review gets a `UsageMeter` held in a context variable. Every model call made while
that review runs adds its token counts to the meter: the judge's own calls, and the
calls LightRAG makes internally (keyword extraction, query embedding) through its
`token_tracker` hook. Stages are timed as they run.

Cost is an **estimate** at list prices (USD per million tokens, per audio hour), not an
Azure bill. The defaults below can be overridden with PRICE_<MODEL>_IN / _OUT env vars.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import BaseModel, Field

# USD list prices, 2026-10 (Azure OpenAI Global Standard / Azure AI Speech standard).
# Model keys match deployment names; unknown models count tokens but cost nothing.
DEFAULT_PRICES = {
    "gpt-4.1-mini": {"in": 0.40, "cached_in": 0.10, "out": 1.60},
    "gpt-5-mini": {"in": 0.25, "cached_in": 0.025, "out": 2.00},
    "text-embedding-3-small": {"in": 0.02, "cached_in": 0.02, "out": 0.0},
}
SPEECH_USD_PER_HOUR = 1.00


def price_for(model: str) -> dict[str, float]:
    base = dict(DEFAULT_PRICES.get(model, {"in": 0.0, "cached_in": 0.0, "out": 0.0}))
    key = model.upper().replace("-", "_").replace(".", "_")
    for part, env in (("in", "IN"), ("out", "OUT")):
        if os.getenv(f"PRICE_{key}_{env}"):
            base[part] = float(os.environ[f"PRICE_{key}_{env}"])
    return base


class ModelTokens(BaseModel):
    calls: int = 0
    prompt: int = 0
    cached: int = 0
    completion: int = 0


class Usage(BaseModel):
    """What one review cost and how long each stage took."""

    total_ms: float = 0.0
    stages_ms: dict[str, float] = Field(default_factory=dict)
    tokens: dict[str, ModelTokens] = Field(default_factory=dict)
    audio_seconds: float = 0.0
    audio_provider: str = ""  # which transcriber handled the audio
    audio_usd_per_hour: float = SPEECH_USD_PER_HOUR
    estimated_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return sum(t.prompt + t.completion for t in self.tokens.values())


class UsageMeter:
    def __init__(self) -> None:
        self.usage = Usage()
        self._start = time.perf_counter()

    def add(self, model: str, prompt: int, completion: int = 0, cached: int = 0) -> None:
        t = self.usage.tokens.setdefault(model, ModelTokens())
        t.calls += 1
        t.prompt += prompt
        t.completion += completion
        t.cached += cached

    def add_response(self, model: str, response) -> None:
        u = getattr(response, "usage", None)
        if u is None:
            return
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        self.add(model, u.prompt_tokens or 0, getattr(u, "completion_tokens", 0) or 0, cached)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - start) * 1000
            self.usage.stages_ms[name] = round(self.usage.stages_ms.get(name, 0.0) + ms, 1)

    def finish(self) -> Usage:
        u = self.usage
        u.total_ms = round((time.perf_counter() - self._start) * 1000, 1)
        usd = 0.0
        for model, t in u.tokens.items():
            p = price_for(model)
            fresh = max(t.prompt - t.cached, 0)
            usd += (fresh * p["in"] + t.cached * p["cached_in"] + t.completion * p["out"]) / 1e6
        usd += u.audio_seconds / 3600 * u.audio_usd_per_hour
        u.estimated_usd = round(usd, 6)
        return u


_current: ContextVar[UsageMeter | None] = ContextVar("usage_meter", default=None)


def current() -> UsageMeter | None:
    return _current.get()


@contextmanager
def metered():
    meter = UsageMeter()
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)


@contextmanager
def stage(name: str):
    """Time a stage of the current review; a no-op outside a review."""
    meter = current()
    if meter is None:
        yield
        return
    with meter.stage(name):
        yield


class LightRAGTracker:
    """Adapter for LightRAG's `token_tracker` hook, labelled with the model it serves.

    LightRAG runs model calls in long-lived worker tasks, which keep the context of
    whichever request started them, so the context variable cannot be trusted there.
    The retriever instead names the meter to charge (`sink`) while it holds its lock."""

    def __init__(self, model: str, sink) -> None:
        self.model = model
        self.sink = sink  # callable returning the UsageMeter to charge, or None

    def add_usage(self, counts: dict) -> None:
        meter = self.sink()
        if meter is not None:
            meter.add(
                self.model, counts.get("prompt_tokens", 0), counts.get("completion_tokens", 0)
            )
