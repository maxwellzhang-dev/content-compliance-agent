"""The human-review queue as a small standalone page. Open WebUI has no view for this,
so the agent serves it itself; the Open WebUI banner links here.

Plain HTML forms, no JavaScript: a reviewer approves the agent's verdict or overrides
it with their own, optionally naming the clauses that decide the case."""

from __future__ import annotations

from html import escape

from .schemas import Verdict

VERDICT = {
    "allowed": "Allowed",
    "conditional": "Allowed with changes",
    "not_allowed": "Not allowed",
}
TABS = [("pending", "Waiting"), ("resolved", "Resolved"), ("auto", "Auto-decided")]

STYLE = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#14171c;--muted:#5d6672;--line:#e3e6ea;--accent:#0f6b5c;
--accent-ink:#fff;--warn:#a35b00;--bad:#b42318;--ok:#067647;--field:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a20;--ink:#e8eaed;
--muted:#9aa3ad;--line:#2a2f37;--accent:#3cc6a8;--accent-ink:#0f1115;--warn:#f0a43a;
--bad:#f97066;--ok:#47cd89;--field:#0f1115}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:960px;margin:0 auto;padding:32px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 18px}
nav{display:flex;gap:6px;margin-bottom:18px;flex-wrap:wrap}
nav a{padding:6px 12px;border-radius:999px;border:1px solid var(--line);color:var(--ink);
text-decoration:none;font-size:14px}nav a.on{background:var(--accent);color:var(--accent-ink);
border-color:var(--accent)}nav span{opacity:.7;margin-left:4px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;
margin-bottom:14px;display:grid;grid-template-columns:1fr;gap:6px}
.card.has-img{grid-template-columns:140px 1fr;column-gap:16px}
.card img{width:140px;border-radius:6px;border:1px solid var(--line);grid-row:span 9}
.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.badge{font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;border:1px solid}
.allowed{color:var(--ok)}.conditional{color:var(--warn)}.not_allowed{color:var(--bad)}
time{color:var(--muted);font-size:13px;margin-left:auto}
.scenario{font-weight:600}ul{margin:2px 0;padding-left:20px}
.reasons li{color:var(--warn)}.meta{color:var(--muted);font-size:13px;margin:0}
form{display:flex;gap:8px;flex-wrap:wrap;align-items:center;border-top:1px solid var(--line);
padding-top:10px;margin-top:4px}
select,input[type=text]{font:inherit;font-size:14px;padding:6px 8px;border-radius:6px;
border:1px solid var(--line);background:var(--field);color:var(--ink)}
input[name=note]{flex:1;min-width:160px}input[name=clauses]{width:130px}
button{font:inherit;font-size:14px;padding:6px 12px;border-radius:6px;cursor:pointer;
border:1px solid var(--line);background:var(--card);color:var(--ink)}
button.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.outcome{font-size:14px}.empty{color:var(--muted);text-align:center;padding:48px 0}
@media (max-width:600px){.card.has-img{grid-template-columns:1fr}.card img{width:100%;
grid-row:auto}}
"""


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _verdict_options(exclude: str) -> str:
    return "".join(
        f'<option value="{v.value}">{VERDICT[v.value]}</option>'
        for v in Verdict
        if v.value != exclude
    )


def _card(rec: dict) -> str:
    rv = rec["review"]
    d, c = rv["decision"], rv["confidence"]
    v = d["verdict"]
    first = rec["images"][0] if rec["images"] else ""
    img = (
        f'<img src="{escape(first)}" alt="Attached content">'
        if first.startswith(("data:image/", "https://", "http://"))
        else ""
    )
    reasons = list(rv["route_reasons"])
    if rec.get("flag_reason"):
        reasons.insert(0, f"reported: {rec['flag_reason']}")
    reasons_html = "".join(f"<li>{escape(r)}</li>" for r in reasons)
    obs = rv.get("observation")
    seen = f'<p class="meta">Image: {escape(obs["description"])}</p>' if obs else ""
    if rec["status"] == "pending":
        action = f"""<form method="post" action="/review/{rec["id"]}">
<button class="primary" name="action" value="approve">Approve</button>
<span class="meta">or</span>
<select name="verdict" aria-label="Correct verdict">{_verdict_options(v)}</select>
<input type="text" name="clauses" placeholder="Clauses, e.g. PRV-2" aria-label="Deciding clauses">
<input type="text" name="note" placeholder="Why (optional)" aria-label="Note">
<button name="action" value="override">Override</button></form>"""
    elif rec["status"] in ("approved", "overridden"):
        rvv = rec["reviewer_verdict"]
        note = f" — {escape(rec['reviewer_note'])}" if rec["reviewer_note"] else ""
        clauses = (
            f" · deciding clauses {escape(', '.join(rec['reviewer_clauses']))}"
            if rec["reviewer_clauses"]
            else ""
        )
        action = (
            f'<p class="outcome">Reviewer {rec["status"]}: '
            f'<span class="{rvv}">{VERDICT.get(rvv, rvv)}</span>{clauses}{note}</p>'
        )
    else:
        action = ""
    cites = escape(", ".join(d["cited_clauses"]) or "nothing")
    created = escape(rec["created_at"][:19].replace("T", " "))
    return f"""<article class="card{" has-img" if img else ""}">{img}
<div class="top"><span class="badge {v}">{VERDICT.get(v, v)}</span>
<span class="meta">#{rec["id"]} · cites {cites}</span>
<time>{created} UTC</time></div>
<div class="scenario">{escape(rec["scenario"])}</div>
{seen}<ul class="reasons">{reasons_html}</ul>
<div>{escape(d["rationale"])}</div>
<p class="meta">verdict probability {_pct(c["verdict_probability"])} · agreement
{_pct(c["agreement"])} · groundedness {_pct(c["groundedness"])}</p>
{action}</article>"""


def render_queue(records: list[dict], tab: str, counts: dict[str, int]) -> str:
    tab_counts = {
        "pending": counts.get("pending", 0),
        "resolved": counts.get("approved", 0) + counts.get("overridden", 0),
        "auto": counts.get("auto", 0),
    }
    nav = "".join(
        f'<a href="/review?tab={key}"{" class=on" if key == tab else ""}>'
        f"{label}<span>{tab_counts[key]}</span></a>"
        for key, label in TABS
    )
    body = "".join(_card(r) for r in records) or '<p class="empty">Nothing here.</p>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Review Queue</title><style>{STYLE}</style></head><body><main>
<h1>Human review queue</h1>
<p class="sub">Reviews the agent was not confident enough to decide alone, and decisions
someone reported as wrong. Approve the agent's verdict or override it; resolved reviews
become evaluation cases.</p>
<nav>{nav}</nav>
{body}</main></body></html>"""
