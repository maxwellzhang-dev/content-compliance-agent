"""Load the policy corpus: clauses plus the concept vocabulary that links them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "policies" / "standards.yaml"


@dataclass(frozen=True)
class Clause:
    id: str
    title: str
    text: str
    concepts: tuple[str, ...]

    def render(self) -> str:
        return f"[{self.id}] {self.title}\n{self.text}"


@dataclass(frozen=True)
class Concept:
    name: str
    aliases: tuple[str, ...]
    related: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PolicyCorpus:
    clauses: tuple[Clause, ...]
    concepts: dict[str, Concept]

    def clause(self, clause_id: str) -> Clause:
        for c in self.clauses:
            if c.id == clause_id:
                return c
        raise KeyError(clause_id)

    @property
    def clause_ids(self) -> set[str]:
        return {c.id for c in self.clauses}


def load_corpus(path: Path | str = DEFAULT_POLICY_PATH) -> PolicyCorpus:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    concepts = {
        name: Concept(
            name=name,
            aliases=tuple(str(a).lower() for a in spec.get("aliases", [])),
            related=tuple(spec.get("related", [])),
        )
        for name, spec in raw["concepts"].items()
    }
    clauses = tuple(
        Clause(
            id=c["id"],
            title=c["title"],
            text=" ".join(c["text"].split()),
            concepts=tuple(c["concepts"]),
        )
        for c in raw["clauses"]
    )
    _validate(clauses, concepts)
    return PolicyCorpus(clauses=clauses, concepts=concepts)


def _validate(clauses: tuple[Clause, ...], concepts: dict[str, Concept]) -> None:
    ids = [c.id for c in clauses]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate clause ids in policy corpus")
    for c in clauses:
        unknown = set(c.concepts) - concepts.keys()
        if unknown:
            raise ValueError(f"{c.id} references unknown concepts: {sorted(unknown)}")
    for concept in concepts.values():
        unknown = set(concept.related) - concepts.keys()
        if unknown:
            raise ValueError(f"concept {concept.name} relates to unknown: {sorted(unknown)}")
