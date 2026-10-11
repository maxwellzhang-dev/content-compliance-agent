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
    recall: float | None  # None when the case names no deciding clause
    hard: bool
    multimodal: bool = False
    error: str | None = None
    # raw signals, so thresholds can be re-chosen later without re-running the model
    verdict_probability: float | None = None
    agreement: float | None = None
    groundedness: float | None = None
    hard_fail: bool = False  # escalated whatever the thresholds (stray citation, nothing found)

    @property
    def correct(self) -> bool:
        return self.expected == self.got


def load_scenarios(path: Path | str = DEFAULT_SCENARIOS) -> list[dict]:
    path = Path(path)
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    for case in cases:
        if "video" in case:
            case["video_path"] = str(path.parent / "videos" / case["video"])
        if "image" in case:
            img = path.parent / "images" / case["image"]
            mime = "jpeg" if img.suffix.lower() in (".jpg", ".jpeg") else img.suffix[1:].lower()
            data = base64.b64encode(img.read_bytes()).decode()
            case["images"] = [f"data:image/{mime};base64,{data}"]
    return cases


def score_case(case: dict, review: Review) -> CaseResult:
    retrieved = {r.id for r in review.retrieved}
    wanted = set(case.get("clauses") or [])
    return CaseResult(
        id=case["id"],
        expected=case["expected"],
        got=review.decision.verdict.value,
        route=review.route,
        recall=len(wanted & retrieved) / len(wanted) if wanted else None,
        hard=bool(case.get("hard")),
        multimodal=bool(case.get("images") or case.get("video_path")),
        verdict_probability=review.confidence.verdict_probability,
        agreement=review.confidence.agreement,
        groundedness=review.confidence.groundedness,
        hard_fail=any(
            r.startswith(("cites clauses", "no policy clause")) for r in review.route_reasons
        ),
    )


def summarise(results: list[CaseResult]) -> dict:
    n = len(results)
    auto = [r for r in results if r.route == "auto"]
    escalated = [r for r in results if r.route != "auto"]
    wrong = [r for r in results if not r.correct]
    return {
        "cases": n,
        "retrieval_recall_at_k": _mean([r.recall for r in results if r.recall is not None]),
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


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def _accuracy(results: list[CaseResult]) -> float | None:
    return round(sum(r.correct for r in results) / len(results), 3) if results else None


async def run(agent, scenarios: list[dict], concurrency: int = 2) -> list[CaseResult]:
    """A failed case is recorded with its error and scored as wrong, never dropped, so
    one exhausted retry cannot abort or silently flatter the whole run."""
    sem = asyncio.Semaphore(concurrency)

    async def one(case):
        async with sem:
            try:
                review = await agent.review(
                    case["scenario"], case.get("images"), video=case.get("video_path")
                )
            except Exception as exc:  # noqa: BLE001 - reported per case
                return CaseResult(
                    id=case["id"],
                    expected=case["expected"],
                    got="error",
                    route="error",
                    recall=0.0,
                    hard=bool(case.get("hard")),
                    multimodal=bool(case.get("images") or case.get("video_path")),
                    error=f"{type(exc).__name__}: {exc}",
                )
            result = score_case(case, review)
            print(f"  done {result.id}", flush=True)
            return result

    return list(await asyncio.gather(*(one(c) for c in scenarios)))


def export_feedback(store, out: Path | str) -> int:
    """Write resolved reviews as eval cases. The reviewer's verdict is the expected
    answer; images are saved next to the other eval images."""
    out = Path(out)
    img_dir = out.parent / "images" / "feedback"
    cases = []
    for rec in sorted(
        store.list("approved", 10_000) + store.list("overridden", 10_000), key=lambda r: r["id"]
    ):
        case = {
            "id": f"fb{rec['id']}",
            "scenario": rec["scenario"],
            "expected": rec["reviewer_verdict"],
            "clauses": rec["reviewer_clauses"],
            "source": f"review #{rec['id']} ({rec['status']})",
        }
        images = [u for u in rec["images"] if u.startswith("data:image/")]
        if images:
            header, data = images[0].split(",", 1)
            ext = header.split("/")[1].split(";")[0]
            img_dir.mkdir(parents=True, exist_ok=True)
            name = f"fb{rec['id']}.{ext}"
            (img_dir / name).write_bytes(base64.b64decode(data))
            case["image"] = f"feedback/{name}"
        cases.append(case)
    out.write_text(yaml.safe_dump(cases, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return len(cases)


def compare(a: Path | str, b: Path | str, prefix: str = "zh-") -> dict:
    """Cross-run consistency: match cases by ID (ignoring `prefix`) and report how often
    the two runs reach the same verdict, and where they differ."""
    import json

    def load(path):
        return {
            c["id"].removeprefix(prefix): c
            for c in json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
            if not c.get("error")
        }

    ra, rb = load(a), load(b)
    shared = sorted(ra.keys() & rb.keys())
    same = [k for k in shared if ra[k]["got"] == rb[k]["got"]]
    return {
        "matched_cases": len(shared),
        "same_verdict": len(same),
        "consistency": round(len(same) / len(shared), 3) if shared else None,
        "accuracy_a": _accuracy_dicts([ra[k] for k in shared]),
        "accuracy_b": _accuracy_dicts([rb[k] for k in shared]),
        "differences": [
            {"id": k, "a": ra[k]["got"], "b": rb[k]["got"], "expected": ra[k]["expected"]}
            for k in shared
            if k not in same
        ],
    }


def _accuracy_dicts(cases: list[dict]) -> float | None:
    return round(sum(c["got"] == c["expected"] for c in cases) / len(cases), 3) if cases else None
