"""The review pipeline: (observe images) -> retrieve -> decide (+ samples) -> ground ->
score -> route."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from . import confidence as conf
from .config import Settings
from .judge import Case, Judge
from .schemas import Confidence, Review


class ComplianceAgent:
    def __init__(self, retriever, judge: Judge, settings: Settings) -> None:
        self.retriever = retriever
        self.judge = judge
        self.settings = settings

    async def review(self, scenario: str, images: list[str] | None = None) -> Review:
        case = Case(text=scenario or "Review the attached content.", images=list(images or []))
        if case.images:
            case.observation = await self.judge.observe(case)
        retrieval = await self.retriever.retrieve(case.query_text())
        (decision, prob), samples = await asyncio.gather(
            self.judge.decide(case, retrieval),
            self.judge.sample(case, retrieval, max(self.settings.samples - 1, 0)),
        )
        claims = await self.judge.ground(decision, retrieval)

        retrieved_ids = {c.id for c in retrieval.clauses}
        verdicts = [decision.verdict] + [s.verdict for s in samples]
        agree = conf.agreement(verdicts)
        grounded, stray = conf.groundedness(claims, decision, retrieved_ids)
        route, reasons = conf.route(
            self.settings,
            verdict_probability=prob,
            agreement_score=agree,
            groundedness_score=grounded,
            stray_citations=stray,
            n_retrieved=len(retrieval.clauses),
        )
        review = Review(
            scenario=case.text,
            observation=case.observation,
            decision=decision,
            retrieved=retrieval.clauses,
            confidence=Confidence(
                verdict_probability=None if prob is None else round(prob, 4),
                agreement=round(agree, 4),
                groundedness=round(grounded, 4),
                sample_verdicts=verdicts,
            ),
            route=route,
            route_reasons=reasons,
            concepts=retrieval.keywords,
        )
        if route == "human_review":
            self._enqueue(review)
        return review

    def _enqueue(self, review: Review) -> None:
        path = Path(self.settings.review_queue_path)
        record = {"queued_at": datetime.now(UTC).isoformat(), **review.model_dump(mode="json")}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_queue(settings: Settings, limit: int = 50) -> list[dict]:
    path = Path(settings.review_queue_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:]][::-1]
