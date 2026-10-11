"""Evaluation harness over eval/scenarios.yaml.

Reports what matters for a routing system, not just raw accuracy:
* retrieval recall@k - did the deciding clause reach the judge at all;
* verdict accuracy overall, and on the reviews the router let through automatically;
* escalation rate, and how many wrong verdicts the router caught.
A good router keeps auto-routed accuracy high while escalating as few cases as possible.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path

import yaml

from .schemas import Review

DEFAULT_SCENARIOS = Path(__file__).resolve().parents[2] / "eval" / "scenarios.yaml"


@dataclass
class CaseResult:
    id: str
    expected: str
    got: str
    route: str
    recall: float
    hard: bool
    multimodal: bool = False
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.expected == self.got


def load_scenarios(path: Path | str = DEFAULT_SCENARIOS) -> list[dict]:
    path = Path(path)
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    for case in cases:
        if "image" in case:
            data = (path.parent / "images" / case["image"]).read_bytes()
            case["images"] = ["data:image/png;base64," + base64.b64encode(data).decode()]
    return cases


def score_case(case: dict, review: Review) -> CaseResult:
    retrieved = {r.id for r in review.retrieved}
    wanted = set(case["clauses"])
    return CaseResult(
        id=case["id"],
        expected=case["expected"],
        got=review.decision.verdict.value,
        route=review.route,
        recall=len(wanted & retrieved) / len(wanted),
        hard=bool(case.get("hard")),
        multimodal=bool(case.get("images")),
    )


def summarise(results: list[CaseResult]) -> dict:
    n = len(results)
    auto = [r for r in results if r.route == "auto"]
    escalated = [r for r in results if r.route != "auto"]
    wrong = [r for r in results if not r.correct]
    return {
        "cases": n,
        "retrieval_recall_at_k": round(sum(r.recall for r in results) / n, 3),
        "verdict_accuracy": round(sum(r.correct for r in results) / n, 3),
        "auto_routed": len(auto),
        "auto_accuracy": round(sum(r.correct for r in auto) / len(auto), 3) if auto else None,
        "escalation_rate": round(len(escalated) / n, 3),
        "wrong_verdicts": len(wrong),
        "wrong_caught_by_router": sum(1 for r in wrong if r.route != "auto"),
        "hard_cases_escalated": sum(1 for r in escalated if r.hard),
        "hard_cases": sum(r.hard for r in results),
        "errors": sum(1 for r in results if r.error),
        "multimodal_accuracy": _accuracy([r for r in results if r.multimodal]),
    }


def _accuracy(results: list[CaseResult]) -> float | None:
    return round(sum(r.correct for r in results) / len(results), 3) if results else None


async def run(agent, scenarios: list[dict], concurrency: int = 2) -> list[CaseResult]:
    """A failed case is recorded with its error and scored as wrong, never dropped, so
    one exhausted retry cannot abort or silently flatter the whole run."""
    sem = asyncio.Semaphore(concurrency)

    async def one(case):
        async with sem:
            try:
                review = await agent.review(case["scenario"], case.get("images"))
            except Exception as exc:  # noqa: BLE001 - reported per case
                return CaseResult(
                    id=case["id"],
                    expected=case["expected"],
                    got="error",
                    route="error",
                    recall=0.0,
                    hard=bool(case.get("hard")),
                    multimodal=bool(case.get("images")),
                    error=f"{type(exc).__name__}: {exc}",
                )
            result = score_case(case, review)
            print(f"  done {result.id}", flush=True)
            return result

    return list(await asyncio.gather(*(one(c) for c in scenarios)))
