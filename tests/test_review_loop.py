import pytest
import yaml
from fastapi.testclient import TestClient

from compliance_agent.api import create_app
from compliance_agent.evaluate import export_feedback, load_scenarios
from compliance_agent.schemas import Verdict

PIXEL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BAAAMB"
    "AQAY3Y2wAAAAAElFTkSuQmCC"
)


@pytest.fixture
def app_client(make_agent, settings):
    def make(**judge_kwargs):
        agent = make_agent(**judge_kwargs)
        return TestClient(create_app(agent=agent, settings=settings)), agent.store

    return make


def test_every_review_is_stored_with_its_route(app_client):
    c, store = app_client()
    with c:
        r = c.post("/api/review", json={"scenario": "Beer in cartoons"}).json()
    assert r["id"] == 1 and store.get(1)["status"] == "auto"


def test_answer_footer_names_the_review(app_client):
    c, _ = app_client(prob=0.5)
    with c:
        text = c.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Beer in cartoons"}]},
        ).json()["choices"][0]["message"]["content"]
    assert "Review #1" in text and "review queue" in text


def test_override_via_form_is_recorded(app_client):
    c, store = app_client(prob=0.5)  # not_allowed, escalated
    with c:
        c.post("/api/review", json={"scenario": "Beer in cartoons"})
        r = c.post(
            "/review/1",
            content="action=override&verdict=conditional&clauses=alc-1,+chd-1&note=move+slot",
            headers={"content-type": "application/x-www-form-urlencoded"},
            follow_redirects=False,
        )
        assert r.status_code == 303
        page = c.get("/review?tab=resolved").text
    rec = store.get(1)
    assert rec["status"] == "overridden" and rec["reviewer_verdict"] == "conditional"
    assert rec["reviewer_clauses"] == ["ALC-1", "CHD-1"] and rec["reviewer_note"] == "move slot"
    assert "Reviewer overridden" in page


def test_approve_keeps_the_agents_verdict(app_client):
    c, store = app_client(prob=0.5)
    with c:
        c.post("/api/review", json={"scenario": "Beer in cartoons"})
        assert c.post("/api/reviews/1/resolve", json={"approve": True}).status_code == 200
        # a resolved review cannot be resolved again
        assert c.post("/api/reviews/1/resolve", json={"approve": True}).status_code == 409
    assert store.get(1)["reviewer_verdict"] == "not_allowed"


def test_override_must_change_the_verdict(app_client):
    c, _ = app_client(prob=0.5)
    with c:
        c.post("/api/review", json={"scenario": "Beer in cartoons"})
        r = c.post("/api/reviews/1/resolve", json={"approve": False, "verdict": "not_allowed"})
    assert r.status_code == 422


def test_flag_pulls_an_auto_review_into_the_queue(app_client):
    c, store = app_client()
    with c:
        c.post("/api/review", json={"scenario": "Beer in cartoons"})
        assert c.post("/api/reviews/1/flag", json={"reason": "too strict"}).status_code == 200
        page = c.get("/review").text
    assert store.get(1)["status"] == "pending"
    assert "reported: too strict" in page


def test_feedback_export_round_trips_into_eval(app_client, tmp_path):
    c, store = app_client(prob=0.5)
    with c:
        c.post("/api/review", json={"scenario": "Is this poster OK?", "images": [PIXEL]})
        c.post(
            "/api/reviews/1/resolve",
            json={"approve": False, "verdict": "conditional", "clauses": ["ALC-1"]},
        )
        c.post("/api/review", json={"scenario": "Text only"})  # still pending: not exported
    out = tmp_path / "feedback.yaml"
    assert export_feedback(store, out) == 1
    case = yaml.safe_load(out.read_text())[0]
    assert case["expected"] == Verdict.CONDITIONAL.value and case["image"] == "feedback/fb1.png"
    loaded = load_scenarios(out)[0]
    assert loaded["images"][0] == PIXEL
