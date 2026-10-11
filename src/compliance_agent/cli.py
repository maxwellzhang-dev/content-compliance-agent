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
        counts = await retriever.index()
        print(f"document status in {settings.rag_dir}: {counts}")
        if any(k.lower() != "processed" and v for k, v in counts.items()):
            raise SystemExit("some clauses did not index; run `compliance-agent index` again")
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


async def _eval(settings, out: str | None) -> None:
    from .evaluate import load_scenarios, run, summarise

    agent = _agent(settings)
    try:
        results = await run(agent, load_scenarios())
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
    sub.add_parser("index", help="build the LightRAG index from policies/standards.yaml")
    r = sub.add_parser("retrieve", help="show which clauses a scenario retrieves")
    r.add_argument("scenario")
    a = sub.add_parser("ask", help="review one scenario")
    a.add_argument("scenario")
    a.add_argument("--json", action="store_true")
    e = sub.add_parser("eval", help="run the labelled scenarios")
    e.add_argument("--out")
    s = sub.add_parser("serve", help="run the API (OpenAI-compatible + JSON)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    settings = Settings.from_env()
    if args.cmd == "index":
        asyncio.run(_index(settings))
    elif args.cmd == "retrieve":
        asyncio.run(_retrieve(args.scenario, settings))
    elif args.cmd == "ask":
        asyncio.run(_ask(args.scenario, settings, args.json))
    elif args.cmd == "eval":
        asyncio.run(_eval(settings, args.out))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("compliance_agent.api:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
