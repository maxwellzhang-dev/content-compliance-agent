"""Offline hybrid graph-vector retrieval, plus the result type every retriever returns.

Production retrieval is LightRAG (`lightrag_store.py`). This module is a small,
no-network stand-in with the same shape, used by the tests, CI and `RETRIEVER=offline`.

Two retrievers look at the same scenario from different angles:

* the **graph** side maps the scenario to policy concepts (alcohol, minors, AI-generated,
  real person ...) and walks concept -> clause edges, plus one hop to related concepts.
  It catches clauses that share no wording with the scenario, e.g. a beer advert placed
  in a cartoon block reaches the alcohol clause *and* the children's clause.
* the **vector** side ranks clauses by embedding similarity, which catches paraphrase the
  concept vocabulary does not cover.

The two rankings are fused with reciprocal rank fusion, so neither score scale has to be
calibrated against the other.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .embeddings import Embedder
from .policies import PolicyCorpus
from .schemas import RetrievedClause

RELATED_HOP_WEIGHT = 0.5
RRF_K = 60


@dataclass
class RetrievalResult:
    clauses: list[RetrievedClause]
    graph_facts: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


class KeywordConceptExtractor:
    """Match concept aliases on word boundaries. Cheap, offline, and predictable."""

    def __init__(self, corpus: PolicyCorpus) -> None:
        self._patterns = {
            name: [re.compile(rf"(?<!\w){re.escape(a)}(?!\w)") for a in c.aliases]
            for name, c in corpus.concepts.items()
        }

    def extract(self, scenario: str) -> list[str]:
        text = scenario.lower()
        return sorted(
            name for name, pats in self._patterns.items() if any(p.search(text) for p in pats)
        )


class PolicyGraph:
    def __init__(self, corpus: PolicyCorpus) -> None:
        self.corpus = corpus
        g = nx.Graph()
        for concept in corpus.concepts.values():
            g.add_node(f"concept:{concept.name}", kind="concept")
            for rel in concept.related:
                g.add_edge(f"concept:{concept.name}", f"concept:{rel}", kind="related")
        for clause in corpus.clauses:
            g.add_node(f"clause:{clause.id}", kind="clause")
            for concept in clause.concepts:
                g.add_edge(f"clause:{clause.id}", f"concept:{concept}", kind="governs")
        self.g = g

    def search(self, concepts: list[str]) -> list[tuple[str, float]]:
        direct = {f"concept:{c}" for c in concepts if f"concept:{c}" in self.g}
        hop = {
            n
            for c in direct
            for n in self.g.neighbors(c)
            if self.g.nodes[n]["kind"] == "concept" and n not in direct
        }
        scores: dict[str, float] = defaultdict(float)
        for node, weight in [(c, 1.0) for c in direct] + [(c, RELATED_HOP_WEIGHT) for c in hop]:
            for n in self.g.neighbors(node):
                if self.g.nodes[n]["kind"] == "clause":
                    scores[n.removeprefix("clause:")] += weight
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class VectorIndex:
    def __init__(self, corpus: PolicyCorpus, embedder: Embedder) -> None:
        self.ids = [c.id for c in corpus.clauses]
        self.embedder = embedder
        self.matrix = embedder.embed([c.render() for c in corpus.clauses])

    def search(self, scenario: str) -> list[tuple[str, float]]:
        q = self.embedder.embed([scenario])[0]
        sims = self.matrix @ q
        order = np.argsort(-sims)
        return [(self.ids[i], float(sims[i])) for i in order]


class OfflineHybridRetriever:
    """No-network stand-in for LightRAG: a hand-specified concept graph plus a lexical
    vector index. Used by the tests, CI and `--offline` demos."""

    def __init__(
        self,
        corpus: PolicyCorpus,
        embedder: Embedder,
        concept_extractor=None,
        top_k: int = 5,
        vector_depth: int = 8,
    ) -> None:
        self.corpus = corpus
        self.graph = PolicyGraph(corpus)
        self.vectors = VectorIndex(corpus, embedder)
        self.keywords = KeywordConceptExtractor(corpus)
        self.concept_extractor = concept_extractor  # optional LLM extractor
        self.top_k = top_k
        self.vector_depth = vector_depth

    def concepts_for(self, scenario: str) -> list[str]:
        found = set(self.keywords.extract(scenario))
        if self.concept_extractor is not None:
            found |= set(self.concept_extractor(scenario, sorted(self.corpus.concepts)))
        return sorted(found & self.corpus.concepts.keys())

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def retrieve(self, scenario: str) -> RetrievalResult:
        concepts = self.concepts_for(scenario)
        rankings = {
            "graph": self.graph.search(concepts),
            "vector": self.vectors.search(scenario)[: self.vector_depth],
        }
        fused: dict[str, float] = defaultdict(float)
        sources: dict[str, list[str]] = defaultdict(list)
        for name, ranking in rankings.items():
            for rank, (cid, _) in enumerate(ranking):
                fused[cid] += 1.0 / (RRF_K + rank + 1)
                sources[cid].append(name)
        top = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[: self.top_k]
        out = []
        for cid, score in top:
            clause = self.corpus.clause(cid)
            out.append(
                RetrievedClause(
                    id=cid,
                    title=clause.title,
                    text=clause.text,
                    score=round(score, 5),
                    sources=sources[cid],
                )
            )
        facts = [f"scenario concept: {c}" for c in concepts]
        return RetrievalResult(clauses=out, graph_facts=facts, keywords=concepts)
