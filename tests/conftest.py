from __future__ import annotations

import dataclasses

import pytest

from compliance_agent.agent import ComplianceAgent
from compliance_agent.config import Settings
from compliance_agent.embeddings import HashingEmbedder
from compliance_agent.policies import load_corpus
from compliance_agent.retrieval import OfflineHybridRetriever
from compliance_agent.schemas import ClaimCheck, Decision, MediaObservation, Verdict
from compliance_agent.store import ReviewStore


class FakeJudge:
    """Scripted judge: verdict, logprob and sample verdicts are set per test."""

    def __init__(
        self, verdict=Verdict.NOT_ALLOWED, prob=0.97, samples=None, cite=None, supported=True
    ):
        self.verdict, self.prob, self.cite, self.supported = verdict, prob, cite, supported
        self.samples = samples

    def _decision(self, retrieval, verdict):
        cited = self.cite if self.cite is not None else [retrieval.clauses[0].id]
        return Decision(
            verdict=verdict,
            rationale="Breaches the clause.",
            cited_clauses=cited,
            required_changes=[],
        )

    async def to_english(self, text):
        self.translated = text
        return "A beer advert during the Saturday cartoon block."

    async def observe(self, case):
        self.observed = case
        return MediaObservation(
            description="A beer advert poster with cartoon characters.",
            visible_text=["Saturday Cartoon Fun", "Brand X Beer"],
            people=[],
            products=["Brand X Beer"],
            disclosures=[],
        )

    async def decide(self, case, retrieval):
        self.decided = case
        return self._decision(retrieval, self.verdict), self.prob

    async def sample(self, case, retrieval, n):
        verdicts = self.samples or [self.verdict] * n
        return [self._decision(retrieval, v) for v in verdicts[:n]]

    async def ground(self, decision, retrieval):
        cid = decision.cited_clauses[0] if decision.cited_clauses else None
        return [
            ClaimCheck(
                claim="Breaches the clause.", kind="policy", supported=self.supported, clause_id=cid
            )
        ]


@pytest.fixture
def corpus():
    return load_corpus()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for var in ("AZURE_OPENAI_ENDPOINT", "AGENT_API_KEY", "JUDGE_MODEL", "JUDGE_IS_REASONING"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("compliance_agent.config.load_dotenv", lambda: None)
    s = Settings.from_env()
    return dataclasses.replace(s, review_db_path=str(tmp_path / "reviews.db"))


@pytest.fixture
def retriever(corpus):
    return OfflineHybridRetriever(corpus, HashingEmbedder())


@pytest.fixture
def make_agent(retriever, settings):
    def make(**judge_kwargs):
        store = ReviewStore(settings.review_db_path)
        return ComplianceAgent(retriever, FakeJudge(**judge_kwargs), settings, store)

    return make
