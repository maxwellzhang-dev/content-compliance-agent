"""The human-review queue as a small standalone page. Open WebUI has no view for this,
so the agent serves it itself; the Open WebUI banner links here."""

from __future__ import annotations

from html import escape

VERDICT = {
    "allowed": "Allowed",
    "conditional": "Allowed with changes",
    "not_allowed": "Not allowed",
}

STYLE = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#14171c;--muted:#5d6672;--line:#e3e6ea;--accent:#0f6b5c;
--warn:#a35b00;--bad:#b42318;--ok:#067647}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a20;--ink:#e8eaed;--muted:#9aa3ad;
--line:#2a2f37;--accent:#3cc6a8;--warn:#f0a43a;--bad:#f97066;--ok:#47cd89}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;
margin-bottom:14px}
.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.badge{font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;border:1px solid}
.allowed{color:var(--ok)}.conditional{color:var(--warn)}.not_allowed{color:var(--bad)}
time{color:var(--muted);font-size:13px;margin-left:auto}
.scenario{font-weight:600;margin:4px 0 8px}ul{margin:6px 0 10px;padding-left:20px}
.reasons li{color:var(--warn)}.meta{color:var(--muted);font-size:13px}
.empty{color:var(--muted);text-align:center;padding:48px 0}a{color:var(--accent)}
"""


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def render_queue(items: list[dict]) -> str:
    cards = []
    for it in items:
        d, c = it["decision"], it["confidence"]
        v = d["verdict"]
        reasons = "".join(f"<li>{escape(r)}</li>" for r in it["route_reasons"])
        obs = it.get("observation")
        seen = f'<p class="meta">Image: {escape(obs["description"])}</p>' if obs else ""
        cards.append(
            f"""<article class="card">
<div class="top"><span class="badge {v}">{VERDICT.get(v, v)}</span>
<span class="meta">cites {escape(", ".join(d["cited_clauses"]) or "nothing")}</span>
<time>{escape(it["queued_at"][:19].replace("T", " "))} UTC</time></div>
<div class="scenario">{escape(it["scenario"])}</div>
{seen}<ul class="reasons">{reasons}</ul>
<div>{escape(d["rationale"])}</div>
<p class="meta">verdict probability {_pct(c["verdict_probability"])} · agreement
{_pct(c["agreement"])} · groundedness {_pct(c["groundedness"])}</p>
</article>"""
        )
    body = "".join(cards) or '<p class="empty">Nothing waiting for review.</p>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Review Queue</title><style>{STYLE}</style></head><body><main>
<h1>Human review queue</h1>
<p class="sub">Reviews the agent was not confident enough to decide alone, newest first.
The reasons say which signal fell short.</p>
{body}</main></body></html>"""
