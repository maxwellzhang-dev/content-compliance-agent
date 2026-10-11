"""Markdown shown in the chat UI. Open WebUI renders <details> as collapsible blocks, so
the verdict and route sit on top and the evidence folds away underneath."""

from __future__ import annotations

from .schemas import Review, Verdict

VERDICT_LABEL = {
    Verdict.ALLOWED: "✅ Allowed",
    Verdict.CONDITIONAL: "⚠️ Allowed with changes",
    Verdict.NOT_ALLOWED: "⛔ Not allowed",
}


def _clip(text: str, limit: int = 500) -> str:
    return f"“{text}”" if len(text) <= limit else f"“{text[:limit].rstrip()}…”"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def render_markdown(review: Review, queue_url: str | None = None) -> str:
    d, c = review.decision, review.confidence
    lines = [f"### {VERDICT_LABEL[d.verdict]}", ""]
    if review.route == "auto":
        lines.append("**Route:** auto — all confidence signals above threshold")
    else:
        lines.append("**Route:** 🧑‍⚖️ sent to human review")
        lines += [f"- {r}" for r in review.route_reasons]
    lines += ["", d.rationale, ""]
    o = review.observation
    if o is not None:
        heading = (
            "What the agent saw and heard" if review.media == "video" else "What the agent saw"
        )
        lines += [f"**{heading}**", o.description]
        rows = [
            ("On-screen text", [f"“{t}”" for t in o.visible_text]),
            ("People", o.people),
            ("Products", o.products),
            ("Disclosures", o.disclosures or ["none visible"]),
        ]
        if review.media == "video":
            heard = _clip(review.transcript) if review.transcript else "no speech recognised"
            rows.append(("Voice-over", [heard]))
        for label, items in rows:
            if items:
                lines.append(f"- *{label}:* {'; '.join(items)}")
        lines.append("")
    if d.required_changes:
        lines.append("**Required changes**")
        lines += [f"{i}. {ch}" for i, ch in enumerate(d.required_changes, 1)]
        lines.append("")

    by_id = {r.id: r for r in review.retrieved}
    if d.cited_clauses:
        lines.append("**Cited clauses**")
        for cid in d.cited_clauses:
            r = by_id.get(cid)
            lines.append(
                f"> **{cid}** {r.title} — {r.text}" if r else f"> **{cid}** (not retrieved)"
            )
            lines.append(">")
        lines.pop()
        lines.append("")

    samples = ", ".join(v.value for v in c.sample_verdicts)
    lines += [
        "<details>",
        "<summary>Confidence signals</summary>",
        "",
        "| Signal | Value |",
        "| --- | --- |",
        f"| Verdict probability (logprobs) | {_pct(c.verdict_probability)} |",
        f"| Agreement across {len(c.sample_verdicts)} samples | {_pct(c.agreement)} |",
        f"| Groundedness of rationale | {_pct(c.groundedness)} |",
        "",
        f"Sample verdicts: {samples}",
        "",
        "</details>",
        "",
        "<details>",
        "<summary>Retrieved clauses</summary>",
        "",
        "| Clause | Score | Found via |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {r.id} {r.title} | {r.score} | {', '.join(r.sources)} |" for r in review.retrieved
    ]
    if review.concepts:
        lines += ["", f"Query keywords: {', '.join(review.concepts)}"]
    lines += ["", "</details>"]
    if review.id is not None:
        where = (
            f" · in the [review queue]({queue_url})" if queue_url and review.route != "auto" else ""
        )
        lines += ["", f"<sub>Review #{review.id}{where}</sub>"]
    return "\n".join(lines)
