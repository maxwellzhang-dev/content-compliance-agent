import json
import math

import pytest

from compliance_agent.confidence import agreement, groundedness
from compliance_agent.judge import verdict_probability
from compliance_agent.schemas import ClaimCheck, Decision, Verdict


def test_verdict_probability_sums_only_the_value_tokens():
    tokens = [
        ('{"', -0.0),
        ("verdict", -0.0),
        ('":"', -0.0),
        ("not", -0.1),
        ("_allowed", -0.2),
        ('","', -0.0),
        ("rationale", -0.5),
    ]
    assert verdict_probability(tokens) == pytest.approx(math.exp(-0.3))


def test_verdict_probability_handles_missing_field():
    assert verdict_probability([("{}", 0.0)]) is None


def test_agreement_bounds():
    assert agreement([Verdict.ALLOWED] * 5) == pytest.approx(1.0)
    assert agreement([Verdict.ALLOWED, Verdict.CONDITIONAL, Verdict.NOT_ALLOWED]) == pytest.approx(
        0.0
    )
    assert 0 < agreement([Verdict.ALLOWED] * 4 + [Verdict.NOT_ALLOWED]) < 1


def test_stray_citation_counts_against_groundedness():
    d = Decision(
        verdict=Verdict.ALLOWED, rationale="x", cited_clauses=["AI-1", "ZZ-9"], required_changes=[]
    )
    claims = [ClaimCheck(claim="x", kind="policy", supported=True, clause_id="AI-1")]
    score, stray = groundedness(claims, d, {"AI-1"})
    assert stray == ["ZZ-9"] and score == pytest.approx(0.5)


async def _review(agent, text):
    return await agent.review(text)


def test_confident_case_is_auto_routed(make_agent):
    import asyncio

    review = asyncio.run(_review(make_agent(), "A beer advert during cartoons"))
    assert review.route == "auto" and review.route_reasons == []


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"prob": 0.6}, "verdict probability"),
        (
            {
                "samples": [
                    Verdict.ALLOWED,
                    Verdict.CONDITIONAL,
                    Verdict.ALLOWED,
                    Verdict.CONDITIONAL,
                ]
            },
            "samples disagree",
        ),
        ({"supported": False}, "weakly grounded"),
        ({"cite": ["ZZ-9"]}, "not retrieved"),
    ],
)
def test_each_weak_signal_escalates_and_is_queued(make_agent, settings, kwargs, reason):
    import asyncio

    review = asyncio.run(_review(make_agent(**kwargs), "A beer advert during cartoons"))
    assert review.route == "human_review"
    assert any(reason in r for r in review.route_reasons)
    queued = [json.loads(x) for x in open(settings.review_queue_path)]
    assert queued[-1]["scenario"] == "A beer advert during cartoons"


def test_missing_logprobs_are_skipped_not_faked(make_agent):
    import asyncio

    review = asyncio.run(_review(make_agent(prob=None), "A beer advert during cartoons"))
    assert review.confidence.verdict_probability is None
    assert review.route == "auto"


def test_scenario_restatements_are_not_scored():
    d = Decision(
        verdict=Verdict.NOT_ALLOWED, rationale="x", cited_clauses=["AI-2"], required_changes=[]
    )
    claims = [
        ClaimCheck(
            claim="The advert clones a singer's voice",
            kind="scenario",
            supported=False,
            clause_id=None,
        ),
        ClaimCheck(
            claim="Cloned voices need consent", kind="policy", supported=True, clause_id="AI-2"
        ),
    ]
    assert groundedness(claims, d, {"AI-2"})[0] == pytest.approx(1.0)


def test_eval_records_failed_cases_instead_of_aborting(make_agent):
    import asyncio

    from compliance_agent.evaluate import run, summarise

    class Boom:
        retriever = None

        async def review(self, scenario, images=None):
            raise RuntimeError("429")

    cases = [{"id": "x", "scenario": "s", "expected": "allowed", "clauses": ["AI-1"]}]
    results = asyncio.run(run(Boom(), cases))
    assert results[0].error == "RuntimeError: 429"
    assert summarise(results)["errors"] == 1 and summarise(results)["verdict_accuracy"] == 0
