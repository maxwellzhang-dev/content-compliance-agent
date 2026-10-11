"""The LLM calls: a vision pass over attached images, one decision with logprobs, extra
samples for agreement, and a grounding check. Kept behind a small protocol so tests can
swap in a fake."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings
from .retrieval import RetrievalResult
from .schemas import ClaimCheck, Decision, GroundingReport, MediaObservation

DECIDE_SYSTEM = """You are a content compliance reviewer for a broadcaster.
Decide whether the scenario is allowed under the policy clauses provided, and only those.
- "allowed": nothing in the clauses is breached as the content stands, including any
  labels or disclosures it already carries.
- "conditional": the content breaches a clause, but a concrete edit to this same content
  would fix it: blur a face, add a label, update a figure, reword a claim, move the
  placement. List those edits.
- "not_allowed": no edit to this content can make it compliant, because what it is or
  does is prohibited outright (for example a product category that is not accepted, or
  a real person depicted in an advert without a signed release). Leave required changes
  empty.
Before choosing "not_allowed", check whether one of the edits above would cure the
breach; if one would, the verdict is "conditional".
When images are attached, they are the content under review: judge what they show, and
use the written observation only as a reading aid. For a video, the images are frames in
time order and the audio transcript is part of the content: a breach in the voice-over
counts the same as one on screen.
Cite clause IDs exactly as written, e.g. "AI-1". Do not cite clauses you were not given.
If the clauses do not cover the scenario, say so in the rationale and choose the verdict
the clauses best support."""

GROUND_SYSTEM = """You check whether a compliance rationale is supported by policy text.
Split the rationale and required changes into individual claims and label each:
- "scenario": restates facts about the content under review (taken as given, not checked);
- "policy": says what the rules require, allow or forbid, or applies them.
For each policy claim, decide whether the quoted policy clauses directly support it.
Mark a claim supported only if a clause says it; reasonable-sounding claims the
clauses do not state are unsupported.
Return the clause ID that supports each supported claim."""

OBSERVE_SYSTEM = """You prepare attached media for a broadcaster's compliance review.
Describe only what is visible; do not judge compliance. Transcribe on-screen text exactly.
For people, give apparent age group and role. Name a person only if they appear to be a
well-known public figure, and say "appears to be". Never identify private individuals.
List any visible labels such as "AI-generated", "Sponsored" or "Ad"; if none, leave empty."""


@dataclass
class Case:
    """The thing under review: what the user wrote, plus any attached images
    (http(s) or data: URLs) and what the vision pass saw in them."""

    text: str
    images: list[str] = field(default_factory=list)
    observation: MediaObservation | None = None
    transcript: str | None = None  # audio track of a video, if any
    is_video: bool = False  # images are frames sampled from one video, in time order
    language: str = "en"  # language of the reviewer's text; answers come back in it
    english_text: str | None = None  # translation used for retrieval when not English

    def query_text(self) -> str:
        """Text used for retrieval, which LightRAG requires to be text only. The policy
        corpus and its graph are English, so a non-English note is searched by its
        translation."""
        parts = [self.english_text or self.text]
        if self.transcript:
            parts.append(f"Audio / voice-over: {self.transcript}")
        if self.observation is None:
            return "\n".join(parts)
        o = self.observation
        parts.append(f"Content shows: {o.description}")
        if o.visible_text:
            parts.append("On-screen text: " + " | ".join(o.visible_text))
        if o.people:
            parts.append("People: " + "; ".join(o.people))
        if o.products:
            parts.append("Products: " + "; ".join(o.products))
        parts.append("Disclosures: " + ("; ".join(o.disclosures) or "none visible"))
        return "\n".join(parts)

    def judge_text(self) -> str:
        """What the judge reads: the retrieval text, plus the original wording when the
        note was translated, so nothing is lost in translation."""
        if self.english_text is None:
            return self.query_text()
        name = LANGUAGE_NAMES.get(self.language, self.language)
        return f"Original note ({name}): {self.text}\n{self.query_text()}"


LANGUAGE_NAMES = {"en": "English", "zh": "Simplified Chinese"}


def detect_language(text: str) -> str:
    """'zh' when CJK characters make up a fifth or more of the letters, else 'en'."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "en"
    cjk = sum(1 for ch in letters if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk / len(letters) >= 0.2 else "en"


TRANSLATE_SYSTEM = """Translate the user's note about a piece of broadcast or advertising
content into plain English, for searching an English policy corpus. Keep brand names,
numbers and quoted on-screen text as they are. Return only the translation."""


def _image_parts(images: list[str]) -> list[dict]:
    return [{"type": "image_url", "image_url": {"url": url}} for url in images]


def build_context(retrieval: RetrievalResult) -> str:
    lines = ["POLICY CLAUSES:"]
    lines += [f"[{c.id}] {c.title}: {c.text}" for c in retrieval.clauses] or ["(none retrieved)"]
    if retrieval.graph_facts:
        lines += ["", "RELATED POLICY GRAPH FACTS (for orientation, not citable):"]
        lines += [f"- {f}" for f in retrieval.graph_facts]
    return "\n".join(lines)


class Judge(Protocol):
    async def to_english(self, text: str) -> str: ...

    async def observe(self, case: Case) -> MediaObservation: ...

    async def decide(
        self, case: Case, retrieval: RetrievalResult
    ) -> tuple[Decision, float | None]: ...

    async def sample(self, case: Case, retrieval: RetrievalResult, n: int) -> list[Decision]: ...

    async def ground(self, decision: Decision, retrieval: RetrievalResult) -> list[ClaimCheck]: ...


def verdict_probability(tokens: list[tuple[str, float]]) -> float | None:
    """Probability the model assigned to the verdict value it emitted.

    `tokens` is the (token, logprob) stream of a JSON object whose first field is
    `verdict`. We rebuild the text, find the character span of the verdict value, and sum
    the logprobs of every token overlapping that span.
    """
    text = "".join(t for t, _ in tokens)
    key = text.find('"verdict"')
    if key < 0:
        return None
    start = text.find('"', text.find(":", key) + 1) + 1
    end = text.find('"', start)
    if start <= 0 or end < 0:
        return None
    pos, total = 0, 0.0
    for tok, lp in tokens:
        tok_end = pos + len(tok)
        if tok_end > start and pos < end:
            total += lp
        pos = tok_end
    return math.exp(total)


class OpenAIJudge:
    def __init__(self, client, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def _messages(self, case: Case, retrieval: RetrievalResult) -> list[dict]:
        text = f"{build_context(retrieval)}\n\nSCENARIO:\n{case.judge_text()}"
        if case.language != "en":
            name = LANGUAGE_NAMES.get(case.language, case.language)
            text += (
                f"\n\nANSWER LANGUAGE: write the rationale and required changes in {name}. "
                "Keep clause IDs exactly as written."
            )
        content = [{"type": "text", "text": text}, *_image_parts(case.images)]
        return [
            {"role": "system", "content": DECIDE_SYSTEM},
            {"role": "user", "content": content if case.images else text},
        ]

    async def _parse(self, messages, schema, **extra):
        return await self.client.chat.completions.parse(
            model=self.settings.judge_model, messages=messages, response_format=schema, **extra
        )

    async def to_english(self, text):
        extra = {} if self.settings.judge_is_reasoning else {"temperature": 0}
        resp = await self.client.chat.completions.create(
            model=self.settings.judge_model,
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM},
                {"role": "user", "content": text},
            ],
            **extra,
        )
        return (resp.choices[0].message.content or "").strip()

    async def observe(self, case):
        note = f"Reviewer's note: {case.judge_text() if case.english_text else case.text}"
        if case.is_video:
            note += (
                f"\nThe {len(case.images)} images are frames sampled in time order from one "
                "video. Describe the video as a whole."
            )
        msgs = [
            {"role": "system", "content": OBSERVE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": note},
                    *_image_parts(case.images),
                ],
            },
        ]
        extra = {} if self.settings.judge_is_reasoning else {"temperature": 0}
        resp = await self._parse(msgs, MediaObservation, **extra)
        return resp.choices[0].message.parsed

    async def decide(self, case, retrieval):
        msgs = self._messages(case, retrieval)
        if self.settings.judge_is_reasoning:
            resp = await self._parse(msgs, Decision)
            return resp.choices[0].message.parsed, None
        resp = await self._parse(msgs, Decision, temperature=0, logprobs=True)
        choice = resp.choices[0]
        content = choice.logprobs.content if choice.logprobs else None
        prob = verdict_probability([(t.token, t.logprob) for t in content]) if content else None
        return choice.message.parsed, prob

    async def sample(self, case, retrieval, n):
        if n == 0:
            return []
        msgs = self._messages(case, retrieval)
        if self.settings.judge_is_reasoning:
            # Reasoning deployments take neither temperature nor n: one call per sample.
            resps = await asyncio.gather(*(self._parse(msgs, Decision) for _ in range(n)))
            return [r.choices[0].message.parsed for r in resps]
        # One request with n choices keeps the request rate down on small quotas.
        resp = await self._parse(msgs, Decision, temperature=1.0, n=n)
        return [c.message.parsed for c in resp.choices]

    async def ground(self, decision, retrieval):
        clauses = "\n".join(f"[{c.id}] {c.text}" for c in retrieval.clauses)
        payload = json.dumps(
            {"rationale": decision.rationale, "required_changes": decision.required_changes}
        )
        msgs = [
            {"role": "system", "content": GROUND_SYSTEM},
            {"role": "user", "content": f"POLICY CLAUSES:\n{clauses}\n\nRATIONALE:\n{payload}"},
        ]
        extra = {} if self.settings.judge_is_reasoning else {"temperature": 0}
        resp = await self._parse(msgs, GroundingReport, **extra)
        return resp.choices[0].message.parsed.claims
