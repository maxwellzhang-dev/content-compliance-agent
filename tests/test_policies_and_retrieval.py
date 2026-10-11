import asyncio

from compliance_agent.evaluate import load_scenarios


def test_corpus_loads_with_unique_ids(corpus):
    assert len(corpus.clauses) == 20
    assert len(corpus.clause_ids) == 20


def test_graph_reaches_clause_with_no_shared_wording(retriever):
    result = asyncio.run(retriever.retrieve("A beer commercial in the Saturday cartoon block"))
    ids = [c.id for c in result.clauses]
    assert "ALC-1" in ids
    assert "graph" in result.clauses[ids.index("ALC-1")].sources
    assert {"alcohol", "minors"} <= set(result.keywords)


def test_offline_recall_on_eval_scenarios(retriever):
    scenarios = [c for c in load_scenarios() if "images" not in c]  # text-only retriever
    hits = 0
    for case in scenarios:
        ids = {c.id for c in asyncio.run(retriever.retrieve(case["scenario"])).clauses}
        hits += bool(set(case["clauses"]) & ids)
    assert hits / len(scenarios) >= 0.85


def test_image_cases_load_as_data_urls():
    cases = [c for c in load_scenarios() if "image" in c]
    assert len(cases) == 6
    assert all(c["images"][0].startswith("data:image/png;base64,") for c in cases)
