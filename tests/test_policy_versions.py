import asyncio
import dataclasses

from fastapi.testclient import TestClient

from compliance_agent.api import create_app
from compliance_agent.policies import diff_versions, load_corpus
from compliance_agent.render import render_markdown


def test_fingerprint_changes_with_any_edit():
    corpus = load_corpus()
    clause = corpus.clause("AI-1")
    edited = dataclasses.replace(clause, text=clause.text + " Extra sentence.")
    assert edited.fingerprint != clause.fingerprint
    assert dataclasses.replace(clause).fingerprint == clause.fingerprint


def test_corpus_version_tracks_every_clause():
    corpus = load_corpus()
    clauses = list(corpus.clauses)
    clauses[0] = dataclasses.replace(clauses[0], title="Renamed")
    changed = dataclasses.replace(corpus, clauses=tuple(clauses))
    assert changed.version != corpus.version


def test_diff_versions():
    d = diff_versions({"A": "1", "B": "2", "C": "3"}, {"A": "1", "B": "9", "D": "4"})
    assert (d.added, d.changed, d.removed) == (["D"], ["B"], ["C"])
    assert diff_versions(None, {"A": "1"}).added == ["A"]
    assert diff_versions({"A": "1"}, {"A": "1"}).empty


def test_review_records_policy_and_clause_versions(make_agent):
    review = asyncio.run(make_agent().review("Beer in cartoons"))
    corpus = load_corpus()
    assert review.policy_version == corpus.version
    assert all(c.version == corpus.clause(c.id).fingerprint for c in review.retrieved)
    assert f"`{corpus.version}`" in render_markdown(review)


def test_queue_flags_reviews_whose_cited_clause_changed(make_agent, settings):
    agent = make_agent(prob=0.5)
    with TestClient(create_app(agent=agent, settings=settings)) as c:
        c.post("/api/review", json={"scenario": "Beer in cartoons"})
        cited = agent.store.get(1)["review"]["decision"]["cited_clauses"][0]
        assert "policy changed since this review" not in c.get("/review").text
        agent.policy_versions = {**agent.policy_versions, cited: "edited"}  # policy updated
        page = c.get("/review").text
        detail = c.get("/review/1").text
    assert f"policy changed since this review: {cited}" in page
    assert "has changed since the review" in detail
