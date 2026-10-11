"""Data shapes shared across the pipeline. The LLM output schemas are enforced with
structured outputs, so a malformed response fails loudly instead of being guessed at."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Verdict(StrEnum):
    ALLOWED = "allowed"
    CONDITIONAL = "conditional"  # allowed only if the listed changes are made
    NOT_ALLOWED = "not_allowed"


class Decision(BaseModel):
    """What the model returns for one scenario. `verdict` is the first field so its
    tokens come first in the output, which keeps the logprob lookup simple."""

    verdict: Verdict
    rationale: str = Field(description="Two to four sentences tying the scenario to the clauses.")
    cited_clauses: list[str] = Field(description="IDs of the clauses the verdict rests on.")
    required_changes: list[str] = Field(
        description="Concrete edits that would make the content compliant. Empty if allowed.",
    )


class ClaimCheck(BaseModel):
    claim: str
    kind: Literal["scenario", "policy"] = Field(
        description="scenario = restates the facts given; policy = says what the rules require"
    )
    supported: bool
    clause_id: str | None


class GroundingReport(BaseModel):
    claims: list[ClaimCheck]


class ConceptSelection(BaseModel):
    concepts: list[str]


class MediaObservation(BaseModel):
    """What the vision call saw in attached images. Feeds retrieval and is shown to the
    reviewer so a misreading can be spotted."""

    description: str = Field(description="One to three sentences on what the content shows.")
    visible_text: list[str] = Field(description="Text overlays, captions, slogans, verbatim.")
    people: list[str] = Field(
        description="Each person shown: apparent age group, role, whether a real public figure "
        "appears to be depicted. Do not identify private individuals."
    )
    products: list[str] = Field(description="Products, brands or services shown or promoted.")
    disclosures: list[str] = Field(
        description='Labels such as "AI-generated", "Sponsored", "Ad"; empty if none visible.'
    )


class RetrievedClause(BaseModel):
    id: str
    title: str
    text: str
    score: float
    sources: list[str]  # "graph", "vector", or both


class Confidence(BaseModel):
    verdict_probability: float | None = Field(
        description="exp(sum of logprobs) of the verdict tokens; None if the model gives none"
    )
    agreement: float = Field(description="1 - normalised semantic entropy across samples")
    groundedness: float = Field(description="share of rationale claims supported by cited clauses")
    sample_verdicts: list[Verdict]


class Review(BaseModel):
    scenario: str
    observation: MediaObservation | None = None
    decision: Decision
    retrieved: list[RetrievedClause]
    confidence: Confidence
    route: str  # "auto" or "human_review"
    route_reasons: list[str]
    concepts: list[str] = Field(default_factory=list, description="retriever keywords")
