"""Choose routing thresholds from recorded eval results.

Every eval run stores each case's three signals. Re-routing those cases under a
different set of thresholds costs nothing, so we can sweep every combination and see the
trade-off: the more we escalate, the more accurate the auto-decided remainder becomes.

The recommendation is the combination that escalates least while keeping accuracy on
auto-decided cases at or above a target. Calibrate on one set, then measure on another
(`eval/heldout.yaml`), or the reported accuracy is fitted to the cases that chose it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    verdict_probability: float
    agreement: float
    groundedness: float


@dataclass(frozen=True)
class Point:
    thresholds: Thresholds
    escalation_rate: float
    auto_accuracy: float | None
    auto_cases: int
    wrong_caught: int
    wrong_total: int


def load_cases(paths: list[Path | str]) -> list[dict]:
    """Cases from one or more results files, pooled. Errored cases are dropped."""
    cases = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cases += [c for c in data["cases"] if not c.get("error") and c.get("agreement") is not None]
    return cases


def escalates(case: dict, t: Thresholds) -> bool:
    p = case.get("verdict_probability")
    return (
        case.get("hard_fail", False)
        or (p is not None and p < t.verdict_probability)
        or case["agreement"] < t.agreement
        or case["groundedness"] < t.groundedness
    )


def evaluate(cases: list[dict], t: Thresholds) -> Point:
    auto = [c for c in cases if not escalates(c, t)]
    wrong = [c for c in cases if c["expected"] != c["got"]]
    correct_auto = sum(c["expected"] == c["got"] for c in auto)
    return Point(
        thresholds=t,
        escalation_rate=1 - len(auto) / len(cases),
        auto_accuracy=correct_auto / len(auto) if auto else None,
        auto_cases=len(auto),
        wrong_caught=sum(1 for c in wrong if escalates(c, t)),
        wrong_total=len(wrong),
    )


def _candidates(cases: list[dict], key: str) -> list[float]:
    """Thresholds only matter at observed values: 0 (never escalate on this signal),
    and each distinct value seen (escalate everything strictly below it)."""
    seen = {round(c[key], 4) for c in cases if c.get(key) is not None}
    return sorted({0.0} | seen)


def sweep(cases: list[dict]) -> list[Point]:
    grid = product(
        _candidates(cases, "verdict_probability"),
        _candidates(cases, "agreement"),
        _candidates(cases, "groundedness"),
    )
    return [evaluate(cases, Thresholds(*t)) for t in grid]


def frontier(points: list[Point]) -> list[Point]:
    """For each escalation rate, the best auto accuracy; then keep only points that
    beat every point with a lower escalation rate."""
    best: dict[float, Point] = {}
    for pt in points:
        if pt.auto_accuracy is None:
            continue
        key = round(pt.escalation_rate, 6)
        cur = best.get(key)
        if cur is None or pt.auto_accuracy > cur.auto_accuracy:
            best[key] = pt
    out, top = [], -1.0
    for key in sorted(best):
        if best[key].auto_accuracy > top:
            out.append(best[key])
            top = best[key].auto_accuracy
    return out


def recommend(points: list[Point], target: float) -> Point | None:
    ok = [p for p in points if p.auto_accuracy is not None and p.auto_accuracy >= target]
    if not ok:
        return None
    # least escalation; then the loosest thresholds, so nothing is stricter than needed
    return min(
        ok,
        key=lambda p: (
            round(p.escalation_rate, 6),
            -p.auto_accuracy,
            p.thresholds.verdict_probability + p.thresholds.agreement + p.thresholds.groundedness,
        ),
    )


def render_svg(points: list[Point], front: list[Point], marks: dict[str, Point]) -> str:
    """Escalation rate (x) against accuracy on auto-decided cases (y), axes fitted to
    the data so the frontier is readable."""
    w, h, pad, left = 560, 360, 52, 76
    scored = [p for p in points if p.auto_accuracy is not None]
    lo = max(0.0, math.floor((min(p.auto_accuracy for p in scored) - 0.02) * 100) / 100)
    xmax = min(1.0, math.ceil((max(p.escalation_rate for p in scored) + 0.05) * 10) / 10)

    def xy(p: Point) -> tuple[float, float]:
        x = left + p.escalation_rate / xmax * (w - left - pad)
        y = h - pad - (p.auto_accuracy - lo) / (1 - lo) * (h - 2 * pad)
        return x, y

    dots = "".join(
        f'<circle cx="{xy(p)[0]:.1f}" cy="{xy(p)[1]:.1f}" r="2.5" fill="#9aa3ad" opacity=".45"/>'
        for p in scored
    )
    line = " ".join(f"{xy(p)[0]:.1f},{xy(p)[1]:.1f}" for p in front)
    colours = {"current": "#a35b00", "recommended": "#0f6b5c"}
    labels = ""
    for i, (name, p) in enumerate(marks.items()):
        x, y = xy(p)
        c = colours.get(name, "#14171c")
        dy = -12 if i % 2 == 0 else 22  # alternate above / below so labels never collide
        labels += (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="none" stroke="{c}" stroke-width="2.5"/>'
            f'<text x="{x:.1f}" y="{y + dy:.1f}" fill="{c}" font-size="12" text-anchor="middle">{name}</text>'
        )
    ticks = ""
    for i in range(6):
        fx = xmax * i / 5
        x = left + i / 5 * (w - left - pad)
        ticks += f'<text x="{x:.0f}" y="{h - pad + 18}" font-size="11" text-anchor="middle">{fx:.0%}</text>'
    for i in range(5):
        fy = lo + i * (1 - lo) / 4
        y = h - pad - i / 4 * (h - 2 * pad)
        ticks += (
            f'<text x="{left - 8}" y="{y + 4:.0f}" font-size="11" text-anchor="end">{fy:.1%}</text>'
        )
        ticks += f'<line x1="{left}" x2="{w - pad}" y1="{y:.0f}" y2="{y:.0f}" stroke="#e3e6ea"/>'
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="system-ui, sans-serif" fill="#14171c">
<rect width="{w}" height="{h}" fill="#ffffff"/>
{ticks}
<line x1="{left}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#5d6672"/>
<line x1="{left}" y1="{pad}" x2="{left}" y2="{h - pad}" stroke="#5d6672"/>
{dots}
<polyline points="{line}" fill="none" stroke="#0f6b5c" stroke-width="2"/>
{labels}
<text x="{(left + w - pad) / 2}" y="{h - 10}" font-size="12" text-anchor="middle">share of reviews escalated to a human</text>
<text x="14" y="{h / 2}" font-size="12" text-anchor="middle" transform="rotate(-90 14 {h / 2})">accuracy on auto-decided reviews</text>
<text x="{left}" y="26" font-size="13" font-weight="600">Every threshold combination (grey) and the best trade-off (line)</text>
</svg>"""


def report(
    points: list[Point], front: list[Point], current: Point, rec: Point | None, target: float
) -> str:
    def row(p: Point) -> str:
        t = p.thresholds
        acc = "n/a" if p.auto_accuracy is None else f"{p.auto_accuracy:.1%}"
        return (
            f"| {t.verdict_probability:.2f} | {t.agreement:.2f} | {t.groundedness:.2f} | "
            f"{p.escalation_rate:.0%} | {acc} ({p.auto_cases}) | {p.wrong_caught}/{p.wrong_total} |"
        )

    head = (
        "| min verdict prob. | min agreement | min groundedness | escalated | "
        "auto accuracy (n) | wrong caught |\n| --- | --- | --- | --- | --- | --- |"
    )
    lines = [f"Combinations tried: {len(points)}", "", "**Current thresholds**", head, row(current)]
    lines += ["", f"**Recommended** (least escalation with auto accuracy ≥ {target:.0%})", head]
    lines.append(row(rec) if rec else "| no combination reaches the target | | | | | |")
    lines += ["", "**Frontier**", head] + [row(p) for p in front]
    return "\n".join(lines)
