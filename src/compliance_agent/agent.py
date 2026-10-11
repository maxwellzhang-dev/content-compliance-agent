"""The review pipeline: (observe images) -> retrieve -> decide (+ samples) -> ground ->
score -> route."""

from __future__ import annotations

import asyncio
from pathlib import Path

from . import confidence as conf
from . import usage
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
        corpus = getattr(retriever, "corpus", None)
        self.policy_version = corpus.version if corpus is not None else ""
        self.policy_versions = corpus.versions if corpus is not None else {}

    async def review(
        self, scenario: str, images: list[str] | None = None, video: str | Path | None = None
    ) -> Review:
        with usage.metered() as meter:
            review, shown = await self._review(scenario, images, video)
            review.usage = meter.finish()
        if self.store is not None:
            review.id = self.store.add(review, shown)
        return review

    async def _review(self, scenario, images, video) -> tuple[Review, list[str]]:
        """Returns the review and the images it was judged on (video frames included)."""
        case = Case(text=scenario or "Review the attached content.", images=list(images or []))
        case.language = detect_language(case.text)
        if case.language != "en":
            with usage.stage("translate"):
                case.english_text = await self.judge.to_english(case.text)
        if video is not None:
            with usage.stage("video"):
                evidence = await gather_evidence(video, self.transcriber)
            case.images, case.transcript, case.is_video = evidence.frames, evidence.transcript, True
            meter = usage.current().usage
            meter.audio_seconds = round(evidence.audio_seconds, 2)
            if self.transcriber is not None:
                tx = self.transcriber
                meter.audio_provider = getattr(tx, "name", type(tx).__name__)
                meter.audio_usd_per_hour = getattr(tx, "usd_per_hour", 0.0)
        if case.images:
            with usage.stage("observe"):
                case.observation = await self.judge.observe(case)
        with usage.stage("retrieve"):
            retrieval = await self.retriever.retrieve(case.query_text())
        with usage.stage("decide"):
            (decision, prob), samples = await asyncio.gather(
                self.judge.decide(case, retrieval),
                self.judge.sample(case, retrieval, max(self.settings.samples - 1, 0)),
            )
        with usage.stage("ground"):
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
            policy_version=self.policy_version,
        )
        return review, case.images
