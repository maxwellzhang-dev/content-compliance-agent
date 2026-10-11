"""Load the policy corpus: clauses plus the concept vocabulary that links them."""

from __future__ import annotations

import hashlib
import json
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

    @property
    def fingerprint(self) -> str:
        """Short hash of everything that defines the clause. Any edit changes it."""
        payload = json.dumps([self.id, self.title, self.text, list(self.concepts)])
        return hashlib.sha256(payload.encode()).hexdigest()[:10]


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

    @property
    def versions(self) -> dict[str, str]:
        return {c.id: c.fingerprint for c in self.clauses}

    @property
    def version(self) -> str:
        """Version of the whole corpus: changes when any clause is added, edited or removed."""
        payload = json.dumps(sorted(self.versions.items()))
        return hashlib.sha256(payload.encode()).hexdigest()[:10]


@dataclass(frozen=True)
class PolicyDiff:
    added: list[str]
    changed: list[str]
    removed: list[str]

    @property
    def empty(self) -> bool:
        return not (self.added or self.changed or self.removed)


def diff_versions(old: dict[str, str] | None, new: dict[str, str]) -> PolicyDiff:
    """What changed between an indexed manifest and the current corpus. With no manifest,
    everything counts as added."""
    old = old or {}
    return PolicyDiff(
        added=sorted(new.keys() - old.keys()),
        changed=sorted(k for k in new.keys() & old.keys() if new[k] != old[k]),
        removed=sorted(old.keys() - new.keys()),
    )


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
