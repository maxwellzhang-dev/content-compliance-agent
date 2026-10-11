import asyncio

import pytest
from fastapi.testclient import TestClient

from compliance_agent import usage
from compliance_agent.api import create_app
from compliance_agent.render import render_markdown


def test_cost_counts_cached_tokens_at_the_cached_price():
    meter = usage.UsageMeter()
    meter.add("gpt-4.1-mini", prompt=1_000_000, completion=1_000_000, cached=500_000)
    u = meter.finish()
    # 0.5M fresh x 0.40 + 0.5M cached x 0.10 + 1M out x 1.60
    assert u.estimated_usd == pytest.approx(0.20 + 0.05 + 1.60)


def test_audio_is_priced_per_hour():
    meter = usage.UsageMeter()
    meter.usage.audio_seconds = 1800
    assert meter.finish().estimated_usd == pytest.approx(0.5)


def test_price_override_from_env(monkeypatch):
    monkeypatch.setenv("PRICE_GPT_4_1_MINI_IN", "1.0")
    assert usage.price_for("gpt-4.1-mini")["in"] == 1.0


def test_unknown_model_counts_tokens_but_costs_nothing():
    meter = usage.UsageMeter()
    meter.add("some-other-model", prompt=10_000, completion=10_000)
    u = meter.finish()
    assert u.total_tokens == 20_000 and u.estimated_usd == 0


def test_lightrag_tracker_charges_the_named_sink_not_the_context():
    target = usage.UsageMeter()
    tracker = usage.LightRAGTracker("text-embedding-3-small", lambda: target)
    with usage.metered() as other:
        tracker.add_usage({"prompt_tokens": 42, "total_tokens": 42})
    assert target.usage.tokens["text-embedding-3-small"].prompt == 42
    assert other.usage.tokens == {}


def test_each_review_gets_its_own_meter_with_stage_timings(make_agent):
    agent = make_agent()

    async def two():
        return await asyncio.gather(agent.review("Beer in cartoons"), agent.review("Beer ad"))

    a, b = asyncio.run(two())
    for r in (a, b):
        assert r.usage.tokens["gpt-4.1-mini"].calls == 1  # not merged across reviews
        assert {"retrieve", "decide", "ground"} <= r.usage.stages_ms.keys()
        assert r.usage.estimated_usd > 0


def test_answer_shows_latency_tokens_and_cost(make_agent):
    review = asyncio.run(make_agent().review("Beer in cartoons"))
    review.id = 9
    text = render_markdown(review)
    assert "*Review #9 · " in text and "2.1k tokens · ≈ $0.0" in text
    assert "Cost and latency" in text and "LightRAG retrieval" in text


def test_stats_page_and_api(make_agent, settings):
    agent = make_agent(prob=0.5)
    with TestClient(create_app(agent=agent, settings=settings)) as c:
        c.post("/api/review", json={"scenario": "Beer in cartoons"})
        c.post("/api/review", json={"scenario": "Beer ad"})
        data = c.get("/api/stats").json()
        page = c.get("/stats").text
    assert data["overall"]["reviews"] == 2 and data["overall"]["escalated"] == 2
    assert data["overall"]["metered"] == 2
    assert data["by_media"]["text"]["mean_cost_usd"] > 0
    assert "Latency p95" in page
