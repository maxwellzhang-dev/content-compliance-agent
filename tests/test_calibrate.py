import json

from compliance_agent import calibrate as cal


def case(expected, got, p, a, g, hard_fail=False):
    return {
        "expected": expected,
        "got": got,
        "verdict_probability": p,
        "agreement": a,
        "groundedness": g,
        "hard_fail": hard_fail,
        "error": None,
    }


CASES = [
    case("allowed", "allowed", 0.99, 1.0, 1.0),
    case("allowed", "allowed", 0.98, 1.0, 1.0),
    case("not_allowed", "not_allowed", 0.97, 1.0, 0.8),
    case("conditional", "not_allowed", 0.70, 0.54, 1.0),  # wrong, low confidence
    case("conditional", "not_allowed", 0.99, 1.0, 1.0),  # wrong, confident: uncatchable
    case("allowed", "allowed", 0.99, 1.0, 1.0, hard_fail=True),
]


def test_escalation_follows_each_signal():
    t = cal.Thresholds(0.9, 0.6, 0.9)
    flags = [cal.escalates(c, t) for c in CASES]
    assert flags == [False, False, True, True, False, True]


def test_missing_logprob_never_escalates_on_that_signal():
    c = case("allowed", "allowed", None, 1.0, 1.0)
    assert not cal.escalates(c, cal.Thresholds(0.99, 0.0, 0.0))


def test_recommend_picks_least_escalation_meeting_target():
    points = cal.sweep(CASES)
    rec = cal.recommend(points, target=0.75)
    assert rec is not None and rec.auto_accuracy >= 0.75
    assert all(
        p.escalation_rate >= rec.escalation_rate
        for p in points
        if p.auto_accuracy is not None and p.auto_accuracy >= 0.75
    )
    # the confident wrong case has the same signals as correct ones: no threshold
    # separates it, so a 100% target is unreachable and nothing is recommended
    assert cal.recommend(points, target=1.0) is None


def test_frontier_is_monotone():
    front = cal.frontier(cal.sweep(CASES))
    rates = [p.escalation_rate for p in front]
    accs = [p.auto_accuracy for p in front]
    assert rates == sorted(rates) and accs == sorted(accs)


def test_load_cases_pools_files_and_drops_errors(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"cases": CASES[:2]}))
    b.write_text(json.dumps({"cases": [CASES[2], {**CASES[3], "error": "429"}]}))
    assert len(cal.load_cases([a, b])) == 3


def test_svg_renders(tmp_path):
    points = cal.sweep(CASES)
    svg = cal.render_svg(points, cal.frontier(points), {"current": points[0]})
    assert svg.startswith("<svg") and "current" in svg
