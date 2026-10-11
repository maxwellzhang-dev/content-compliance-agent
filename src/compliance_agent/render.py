"""Markdown shown in the chat UI. Open WebUI renders <details> as collapsible blocks, so
the verdict and route sit on top and the evidence folds away underneath.

Labels follow the language the reviewer wrote in (English or Simplified Chinese). Policy
clauses are always quoted in English, the language they are written in."""

from __future__ import annotations

import re

from .schemas import Review, Verdict

TEXT = {
    "en": {
        Verdict.ALLOWED: "✅ Allowed",
        Verdict.CONDITIONAL: "⚠️ Allowed with changes",
        Verdict.NOT_ALLOWED: "⛔ Not allowed",
        "route": "Route",
        "auto": "auto — all confidence signals above threshold",
        "human": "🧑‍⚖️ sent to human review",
        "saw": "What the agent saw",
        "saw_heard": "What the agent saw and heard",
        "on_screen": "On-screen text",
        "people": "People",
        "products": "Products",
        "disclosures": "Disclosures",
        "none_visible": "none visible",
        "voice": "Voice-over",
        "no_speech": "no speech recognised",
        "changes": "Required changes",
        "cited": "Cited clauses",
        "not_retrieved": "(not retrieved)",
        "signals": "Confidence signals",
        "signal": "Signal",
        "value": "Value",
        "p": "Verdict probability (logprobs)",
        "agree": "Agreement across {n} samples",
        "ground": "Groundedness of rationale",
        "samples": "Sample verdicts",
        "retrieved": "Retrieved clauses",
        "clause": "Clause",
        "score": "Score",
        "via": "Found via",
        "keywords": "Query keywords",
        "review": "Review",
        "in_queue": "in the [review queue]({url})",
    },
    "zh": {
        Verdict.ALLOWED: "✅ 可以播出",
        Verdict.CONDITIONAL: "⚠️ 修改后可播出",
        Verdict.NOT_ALLOWED: "⛔ 不可播出",
        "route": "路由",
        "auto": "自动通过 — 三项置信度信号均达标",
        "human": "🧑‍⚖️ 已转人工复核",
        "saw": "Agent 看到的内容",
        "saw_heard": "Agent 看到和听到的内容",
        "on_screen": "画面文字",
        "people": "人物",
        "products": "产品",
        "disclosures": "标识",
        "none_visible": "未见标识",
        "voice": "口播",
        "no_speech": "未识别到语音",
        "changes": "需要修改",
        "cited": "引用条款（英文原文）",
        "not_retrieved": "（未检索到）",
        "signals": "置信度信号",
        "signal": "信号",
        "value": "数值",
        "p": "结论概率（logprobs）",
        "agree": "{n} 次采样一致性",
        "ground": "理由 groundedness",
        "samples": "各次采样结论",
        "retrieved": "检索到的条款",
        "clause": "条款",
        "score": "得分",
        "via": "来源",
        "keywords": "检索关键词",
        "review": "审核记录",
        "in_queue": "见[复核队列]({url})",
    },
}

# Routing reasons are produced in English by confidence.route; these render them in Chinese.
REASONS_ZH = [
    (re.compile(r"no policy clause was retrieved"), "没有检索到相关政策条款"),
    (re.compile(r"verdict probability ([\d.]+) < ([\d.]+)"), r"结论概率 \1 低于阈值 \2"),
    (re.compile(r"samples disagree \(agreement ([\d.]+)\)"), r"多次采样结论不一致（一致性 \1）"),
    (re.compile(r"rationale weakly grounded \(([\d.]+)\)"), r"理由缺少条款支撑（groundedness \1）"),
    (re.compile(r"cites clauses that were not retrieved: (.+)"), r"引用了未检索到的条款：\1"),
    (re.compile(r"reported: (.+)"), r"被报告误判：\1"),
]


def _reason(reason: str, lang: str) -> str:
    if lang != "zh":
        return reason
    for pattern, repl in REASONS_ZH:
        if pattern.search(reason):
            return pattern.sub(repl, reason)
    return reason


def _clip(text: str, limit: int = 500) -> str:
    return f"“{text}”" if len(text) <= limit else f"“{text[:limit].rstrip()}…”"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def render_markdown(review: Review, queue_url: str | None = None) -> str:
    t = TEXT.get(review.language, TEXT["en"])
    d, c = review.decision, review.confidence
    lines = [f"### {t[d.verdict]}", ""]
    if review.route == "auto":
        lines.append(f"**{t['route']}:** {t['auto']}")
    else:
        lines.append(f"**{t['route']}:** {t['human']}")
        lines += [f"- {_reason(r, review.language)}" for r in review.route_reasons]
    lines += ["", d.rationale, ""]
    o = review.observation
    if o is not None:
        heading = t["saw_heard"] if review.media == "video" else t["saw"]
        lines += [f"**{heading}**", o.description]
        rows = [
            (t["on_screen"], [f"“{x}”" for x in o.visible_text]),
            (t["people"], o.people),
            (t["products"], o.products),
            (t["disclosures"], o.disclosures or [t["none_visible"]]),
        ]
        if review.media == "video":
            heard = _clip(review.transcript) if review.transcript else t["no_speech"]
            rows.append((t["voice"], [heard]))
        for label, items in rows:
            if items:
                lines.append(f"- *{label}:* {'; '.join(items)}")
        lines.append("")
    if d.required_changes:
        lines.append(f"**{t['changes']}**")
        lines += [f"{i}. {ch}" for i, ch in enumerate(d.required_changes, 1)]
        lines.append("")

    by_id = {r.id: r for r in review.retrieved}
    if d.cited_clauses:
        lines.append(f"**{t['cited']}**")
        for cid in d.cited_clauses:
            r = by_id.get(cid)
            lines.append(
                f"> **{cid}** {r.title} — {r.text}" if r else f"> **{cid}** {t['not_retrieved']}"
            )
            lines.append(">")
        lines.pop()
        lines.append("")

    samples = ", ".join(v.value for v in c.sample_verdicts)
    lines += [
        "<details>",
        f"<summary>{t['signals']}</summary>",
        "",
        f"| {t['signal']} | {t['value']} |",
        "| --- | --- |",
        f"| {t['p']} | {_pct(c.verdict_probability)} |",
        f"| {t['agree'].format(n=len(c.sample_verdicts))} | {_pct(c.agreement)} |",
        f"| {t['ground']} | {_pct(c.groundedness)} |",
        "",
        f"{t['samples']}: {samples}",
        "",
        "</details>",
        "",
        "<details>",
        f"<summary>{t['retrieved']}</summary>",
        "",
        f"| {t['clause']} | {t['score']} | {t['via']} |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {r.id} {r.title} | {r.score} | {', '.join(r.sources)} |" for r in review.retrieved
    ]
    if review.concepts:
        lines += ["", f"{t['keywords']}: {', '.join(review.concepts)}"]
    lines += ["", "</details>"]
    if review.id is not None:
        where = (
            f" · {t['in_queue'].format(url=queue_url)}"
            if queue_url and review.route != "auto"
            else ""
        )
        lines += ["", f"<sub>{t['review']} #{review.id}{where}</sub>"]
    return "\n".join(lines)
