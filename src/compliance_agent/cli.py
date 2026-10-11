"""Command line: index the policies, review one scenario, run the eval, or serve."""

from __future__ import annotations

import argparse
import asyncio
import json
import os


def _agent(settings):
    from .api import build_agent

    return build_agent(settings)


async def _index(settings) -> None:
    from .lightrag_store import LightRAGRetriever
    from .policies import load_corpus

    retriever = LightRAGRetriever(load_corpus(), settings)
    try:
        diff, counts = await retriever.index()
        print(
            f"policy version {retriever.corpus.version}: "
            f"{len(diff.added)} added, {len(diff.changed)} changed, {len(diff.removed)} removed"
        )
        for label, ids in (
            ("added", diff.added),
            ("changed", diff.changed),
            ("removed", diff.removed),
        ):
            if ids:
                print(f"  {label}: {', '.join(ids)}")
        print(f"document status in {settings.rag_dir}: {counts}")
        left = retriever.pending_changes()
        if not left.empty:
            missing = ", ".join(left.added + left.changed)
            raise SystemExit(f"not indexed yet: {missing}; run `compliance-agent index` again")
    finally:
        await retriever.stop()


async def _retrieve(scenario: str, settings) -> None:
    from .policies import load_corpus

    if os.getenv("RETRIEVER") == "offline":
        from .embeddings import HashingEmbedder
        from .retrieval import OfflineHybridRetriever

        retriever = OfflineHybridRetriever(load_corpus(), HashingEmbedder())
    else:
        from .lightrag_store import LightRAGRetriever

        retriever = LightRAGRetriever(load_corpus(), settings)
    try:
        result = await retriever.retrieve(scenario)
    finally:
        await retriever.stop()
    for c in result.clauses:
        print(f"{c.id:6} {c.score:<8} {','.join(c.sources):14} {c.title}")
    if result.keywords:
        print("keywords:", ", ".join(result.keywords))


async def _ask(scenario: str, settings, as_json: bool) -> None:
    from .render import render_markdown

    agent = _agent(settings)
    try:
        review = await agent.review(scenario)
    finally:
        await agent.retriever.stop()
    print(review.model_dump_json(indent=2) if as_json else render_markdown(review))


async def _eval(settings, out: str | None, scenarios: str | None) -> None:
    from .evaluate import DEFAULT_SCENARIOS, load_scenarios, run, summarise

    agent = _agent(settings)
    try:
        results = await run(agent, load_scenarios(scenarios or DEFAULT_SCENARIOS))
    finally:
        await agent.retriever.stop()
    for r in results:
        mark = "ok " if r.correct else "BAD"
        print(
            f"{mark} {r.id} expected={r.expected:12} got={r.got:12} {r.route:13} recall={r.recall}"
        )
    summary = summarise(results)
    print(json.dumps(summary, indent=2))
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "cases": [r.__dict__ for r in results]}, f, indent=2)


def main() -> None:
    from .config import Settings

    p = argparse.ArgumentParser(prog="compliance-agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="sync the LightRAG index with policies/standards.yaml")
    sub.add_parser("policy-diff", help="show policy changes not yet indexed (no API calls)")
    r = sub.add_parser("retrieve", help="show which clauses a scenario retrieves")
    r.add_argument("scenario")
    a = sub.add_parser("ask", help="review one scenario")
    a.add_argument("scenario")
    a.add_argument("--json", action="store_true")
    e = sub.add_parser("eval", help="run the labelled scenarios")
    e.add_argument("--out")
    e.add_argument("--scenarios", help="YAML file of cases (default eval/scenarios.yaml)")
    k = sub.add_parser("calibrate", help="choose thresholds from recorded eval results")
    k.add_argument("results", nargs="+", help="results JSON files from `eval --out`, pooled")
    k.add_argument("--target", type=float, default=0.95, help="minimum auto accuracy")
    k.add_argument("--svg", help="write the trade-off chart here")
    m = sub.add_parser("compare", help="verdict consistency between two runs (e.g. en vs zh)")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--prefix", default="zh-", help="ID prefix to ignore when matching")
    f = sub.add_parser("export-feedback", help="turn resolved reviews into eval cases")
    f.add_argument("--out", default="eval/feedback.yaml")
    s = sub.add_parser("serve", help="run the API (OpenAI-compatible + JSON)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    settings = Settings.from_env()
    if args.cmd == "index":
        asyncio.run(_index(settings))
    elif args.cmd == "policy-diff":
        from .lightrag_store import LightRAGRetriever
        from .policies import load_corpus

        retriever = LightRAGRetriever(load_corpus(), settings)
        d = retriever.pending_changes()
        print(f"policy version {retriever.corpus.version}")
        print("index is up to date" if d.empty else json.dumps(d.__dict__, indent=2))
    elif args.cmd == "retrieve":
        asyncio.run(_retrieve(args.scenario, settings))
    elif args.cmd == "ask":
        asyncio.run(_ask(args.scenario, settings, args.json))
    elif args.cmd == "eval":
        asyncio.run(_eval(settings, args.out, args.scenarios))
    elif args.cmd == "calibrate":
        from pathlib import Path

        from . import calibrate as cal

        cases = cal.load_cases(args.results)
        points = cal.sweep(cases)
        front = cal.frontier(points)
        current = cal.evaluate(
            cases,
            cal.Thresholds(
                settings.min_verdict_probability, settings.min_agreement, settings.min_groundedness
            ),
        )
        rec = cal.recommend(points, args.target)
        print(f"{len(cases)} cases from {len(args.results)} file(s)\n")
        print(cal.report(points, front, current, rec, args.target))
        if args.svg:
            marks = {"current": current} | ({"recommended": rec} if rec else {})
            Path(args.svg).parent.mkdir(parents=True, exist_ok=True)
            Path(args.svg).write_text(cal.render_svg(points, front, marks), encoding="utf-8")
            print(f"\nchart written to {args.svg}")
    elif args.cmd == "compare":
        from .evaluate import compare

        print(json.dumps(compare(args.a, args.b, args.prefix), indent=2, ensure_ascii=False))
    elif args.cmd == "export-feedback":
        from .evaluate import export_feedback
        from .store import ReviewStore

        n = export_feedback(ReviewStore(settings.review_db_path), args.out)
        print(f"wrote {n} cases to {args.out}")
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("compliance_agent.api:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
