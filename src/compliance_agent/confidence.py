"""Turn the three signals into a routing decision.

* **verdict probability** - how sure the model was of the label it emitted, from token
  logprobs. Unavailable on reasoning models, in which case it is skipped, not faked.
* **agreement** - 1 minus the normalised entropy of the verdicts across independent
  samples. This is the discrete form of semantic entropy (Farquhar et al., Nature 2024):
  answers are clustered by meaning, and here the meaning that matters is the verdict.
* **groundedness** - share of the rationale's policy claims that a retrieved clause
  supports (restatements of the scenario are not scored), with
  citations of clauses that were never retrieved counted against it.

Any signal under its threshold sends the review to a human. The reasons are kept so the
reviewer knows what to look at first.
"""

from __future__ import annotations

import math
from collections import Counter

from .config import Settings
from .schemas import ClaimCheck, Decision, Verdict

N_VERDICTS = len(Verdict)


def agreement(verdicts: list[Verdict]) -> float:
    if not verdicts:
        return 0.0
    counts = Counter(verdicts)
    n = len(verdicts)
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    return 1.0 - entropy / math.log(N_VERDICTS)


def groundedness(
    claims: list[ClaimCheck], decision: Decision, retrieved_ids: set[str]
) -> tuple[float, list[str]]:
    """Only policy claims are scored; restating the scenario needs no clause behind it."""
    stray = sorted(set(decision.cited_clauses) - retrieved_ids)
    policy = [c for c in claims if c.kind == "policy"]
    if not policy:
        return 0.0, stray
    supported = sum(1 for c in policy if c.supported and c.clause_id in retrieved_ids)
    return supported / (len(policy) + len(stray)), stray


def route(
    settings: Settings,
    *,
    verdict_probability: float | None,
    agreement_score: float,
    groundedness_score: float,
    stray_citations: list[str],
    n_retrieved: int,
) -> tuple[str, list[str]]:
    reasons = []
    if n_retrieved == 0:
        reasons.append("no policy clause was retrieved for this scenario")
    if verdict_probability is not None and verdict_probability < settings.min_verdict_probability:
        reasons.append(
            f"verdict probability {verdict_probability:.2f} < {settings.min_verdict_probability}"
        )
    if agreement_score < settings.min_agreement:
        reasons.append(f"samples disagree (agreement {agreement_score:.2f})")
    if groundedness_score < settings.min_groundedness:
        reasons.append(f"rationale weakly grounded ({groundedness_score:.2f})")
    if stray_citations:
        reasons.append(f"cites clauses that were not retrieved: {', '.join(stray_citations)}")
    return ("human_review" if reasons else "auto"), reasons
