"""The review pipeline: (observe images) -> retrieve -> decide (+ samples) -> ground ->
score -> route."""

from __future__ import annotations

import asyncio
from pathlib import Path

from . import confidence as conf
from .config import Settings
from .judge import Case, Judge, detect_language
from .schemas import Confidence, Review
from .store import ReviewStore
from .video import gather_evidence


class ComplianceAgent:
    def __init__(
        self,
        retriever,
        judge: Judge,
        settings: Settings,
        store: ReviewStore | None = None,
        transcriber=None,
    ) -> None:
        self.retriever = retriever
        self.judge = judge
        self.settings = settings
        self.store = store
        self.transcriber = transcriber  # needed for video audio; None = frames only

    async def review(
        self, scenario: str, images: list[str] | None = None, video: str | Path | None = None
    ) -> Review:
        case = Case(text=scenario or "Review the attached content.", images=list(images or []))
        case.language = detect_language(case.text)
        if case.language != "en":
            case.english_text = await self.judge.to_english(case.text)
        if video is not None:
            evidence = await gather_evidence(video, self.transcriber)
            case.images, case.transcript, case.is_video = evidence.frames, evidence.transcript, True
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
            transcript=case.transcript,
            language=case.language,
            media="video" if case.is_video else ("image" if case.images else "text"),
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
        if self.store is not None:
            review.id = self.store.add(review, case.images)
        return review
