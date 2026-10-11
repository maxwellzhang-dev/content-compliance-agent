"""LightRAG-backed retrieval.

Each policy clause is inserted as its own document with file path `<clause id>.md`, so
whatever LightRAG returns (chunks, entities, relations) can be traced back to clause
IDs. Retrieval uses `mix` mode: knowledge-graph traversal (entities and relations that
LightRAG extracted at indexing time) combined with vector search over chunks. We call
`aquery_data`, which stops before generation; the decision itself is made by our own
judge so we control the output schema and get token logprobs.
"""

from __future__ import annotations

import asyncio
import os
import re
from functools import partial
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from .config import Settings
from .policies import PolicyCorpus
from .retrieval import RetrievalResult
from .schemas import RetrievedClause
from .usage import LightRAGTracker, UsageMeter
from .usage import current as current_meter

_CLAUSE_FILE = re.compile(r"([A-Z]{2,3}-\d+)\.md")


def clause_ids_in(file_path: str | None) -> list[str]:
    """LightRAG joins merged sources with a separator, so one entity can carry several."""
    return _CLAUSE_FILE.findall(file_path or "")


class LightRAGRetriever:
    def __init__(self, corpus: PolicyCorpus, settings: Settings, top_k: int = 5) -> None:
        self.corpus = corpus
        self.settings = settings
        self.top_k = top_k
        Path(settings.rag_dir).mkdir(parents=True, exist_ok=True)

        self._sink: UsageMeter | None = None  # meter charged for LightRAG's own calls
        self._lock: asyncio.Lock | None = None
        llm_tracker = LightRAGTracker(settings.index_model, lambda: self._sink)
        embed_tracker = LightRAGTracker(settings.embedding_model, lambda: self._sink)
        azure = settings.provider == "azure"
        conn = (
            {
                "use_azure": True,
                "base_url": settings.azure_endpoint,
                "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                "api_version": settings.api_version,
            }
            if azure
            else {"base_url": settings.openai_base_url}
        )

        async def llm(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
            return await openai_complete_if_cache(
                settings.index_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                **({"azure_deployment": settings.index_model} if azure else {}),
                **conn,
                **{**kwargs, "token_tracker": llm_tracker},
            )

        embed = EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            model_name=settings.embedding_model,
            func=partial(
                openai_embed.func,
                model=settings.embedding_model,
                token_tracker=embed_tracker,
                **({"azure_deployment": settings.embedding_model} if azure else {}),
                **conn,
            ),
        )

        self.rag = LightRAG(
            working_dir=settings.rag_dir,
            llm_model_func=llm,
            llm_model_name=settings.index_model,
            embedding_func=embed,
            llm_model_max_async=4,
        )
        self._ready = False

    async def start(self) -> None:
        if not self._ready:
            await self.rag.initialize_storages()
            self._ready = True

    async def stop(self) -> None:
        if self._ready:
            await self.rag.finalize_storages()
            self._ready = False

    async def index(self) -> dict[str, int]:
        """Insert every clause and return document counts by status.

        LightRAG skips IDs it already holds and retries failed documents on the next
        run, so re-running `index` after a rate-limit failure finishes the job."""
        await self.start()
        await self.rag.ainsert(
            [c.render() for c in self.corpus.clauses],
            ids=[f"clause-{c.id}" for c in self.corpus.clauses],
            file_paths=[f"{c.id}.md" for c in self.corpus.clauses],
        )
        return await self.rag.doc_status.get_status_counts()

    async def retrieve(self, scenario: str) -> RetrievalResult:
        await self.start()
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:  # one retrieval at a time, so its tokens go to one meter
            self._sink = current_meter()
            try:
                result = await self.rag.aquery_data(
                    scenario, QueryParam(mode="mix", top_k=20, chunk_top_k=8, enable_rerank=False)
                )
            finally:
                self._sink = None
        data = result.get("data") or {}
        ranked: dict[str, tuple[float, set[str]]] = {}

        def add(cid: str, score: float, source: str) -> None:
            if cid not in self.corpus.clause_ids:
                return
            prev, sources = ranked.get(cid, (0.0, set()))
            ranked[cid] = (prev + score, sources | {source})

        for rank, chunk in enumerate(data.get("chunks", [])):
            for cid in clause_ids_in(chunk.get("file_path")):
                add(cid, 1.0 / (rank + 1), "chunk")
        for rank, rel in enumerate(data.get("relationships", [])):
            for cid in clause_ids_in(rel.get("file_path")):
                add(cid, 0.5 / (rank + 1), "graph")
        for rank, ent in enumerate(data.get("entities", [])):
            for cid in clause_ids_in(ent.get("file_path")):
                add(cid, 0.25 / (rank + 1), "graph")

        top = sorted(ranked.items(), key=lambda kv: (-kv[1][0], kv[0]))[: self.top_k]
        clauses = []
        for cid, (score, sources) in top:
            c = self.corpus.clause(cid)
            clauses.append(
                RetrievedClause(
                    id=cid,
                    title=c.title,
                    text=c.text,
                    score=round(score, 4),
                    sources=sorted(sources),
                )
            )
        facts = [
            f"{r.get('src_id')} -> {r.get('tgt_id')}: {r.get('description', '')}".strip()
            for r in data.get("relationships", [])[:8]
        ]
        keywords = (result.get("metadata") or {}).get("keywords") or {}
        return RetrievalResult(
            clauses=clauses,
            graph_facts=facts,
            keywords=list(keywords.get("high_level", [])) + list(keywords.get("low_level", [])),
        )
