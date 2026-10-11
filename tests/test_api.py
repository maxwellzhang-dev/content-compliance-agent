import json

from fastapi.testclient import TestClient

from compliance_agent.api import create_app


def client(make_agent, settings):
    return TestClient(create_app(agent=make_agent(), settings=settings))


def test_models_lists_the_agent(make_agent, settings):
    with client(make_agent, settings) as c:
        assert c.get("/v1/models").json()["data"][0]["id"] == "compliance-agent"


def test_chat_completion_returns_markdown_review(make_agent, settings):
    with client(make_agent, settings) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "compliance-agent",
                "messages": [{"role": "user", "content": "A beer advert during cartoons"}],
            },
        )
    text = r.json()["choices"][0]["message"]["content"]
    assert "Not allowed" in text and "Confidence signals" in text


def test_streaming_matches_openai_chunk_format(make_agent, settings):
    with client(make_agent, settings) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Beer in cartoons"}]}
                ],
            },
        )
    events = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    body = "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1])
    assert "Route" in body


def test_open_webui_background_tasks_are_not_reviewed(make_agent, settings):
    with client(make_agent, settings) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "### Task:\nGenerate a concise title"}],
            },
        )
    assert r.json()["choices"][0]["message"]["content"] == ""


def test_api_key_is_enforced(make_agent, settings):
    import dataclasses

    locked = dataclasses.replace(settings, api_key="s3cret")
    with TestClient(create_app(agent=make_agent(), settings=locked)) as c:
        assert c.get("/v1/models").status_code == 401
        ok = c.get("/v1/models", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200


def test_escalated_review_appears_on_queue_page(retriever, settings):
    from compliance_agent.agent import ComplianceAgent
    from tests.conftest import FakeJudge

    agent = ComplianceAgent(retriever, FakeJudge(prob=0.5), settings)
    with TestClient(create_app(agent=agent, settings=settings)) as c:
        r = c.post("/api/review", json={"scenario": "Beer <b>in</b> cartoons"})
        assert r.status_code == 200, r.text
        page = c.get("/review").text
    assert "Beer &lt;b&gt;in&lt;/b&gt; cartoons" in page
    assert "verdict probability 0.50" in page or "verdict probability 50%" in page


PIXEL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BAAAMB"
    "AQAY3Y2wAAAAAElFTkSuQmCC"
)


def test_attached_image_is_observed_and_drives_retrieval(retriever, settings):
    from compliance_agent.agent import ComplianceAgent
    from tests.conftest import FakeJudge

    judge = FakeJudge()
    agent = ComplianceAgent(retriever, judge, settings)
    with TestClient(create_app(agent=agent, settings=settings)) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Can this run on Saturday morning?"},
                            {"type": "image_url", "image_url": {"url": PIXEL}},
                        ],
                    }
                ]
            },
        )
    text = r.json()["choices"][0]["message"]["content"]
    assert judge.observed.images == [PIXEL]
    assert judge.decided.images == [PIXEL]  # the judge sees the image, not just the notes
    assert "Brand X Beer" in judge.decided.query_text()
    assert "What the agent saw" in text and "Disclosures:* none visible" in text


def test_non_image_urls_are_ignored(make_agent, settings):
    from compliance_agent.api import _images

    parts = [
        {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}},
        {"type": "image_url", "image_url": {"url": PIXEL}},
    ]
    assert _images(parts) == [PIXEL]


def test_empty_request_is_rejected(make_agent, settings):
    with client(make_agent, settings) as c:
        r = c.post("/api/review", json={"scenario": "  "})
    assert r.status_code == 400
