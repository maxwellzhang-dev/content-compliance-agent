import asyncio
import json

from compliance_agent.evaluate import compare
from compliance_agent.judge import Case, OpenAIJudge, detect_language
from compliance_agent.render import render_markdown
from compliance_agent.retrieval import RetrievalResult


def test_detect_language():
    assert detect_language("一条啤酒广告在卡通时段播出") == "zh"
    assert detect_language("A beer advert in the cartoon block") == "en"
    assert detect_language("Brand X 啤酒广告") == "zh"  # brand names in Latin script
    assert detect_language("") == "en"


def test_chinese_note_is_translated_for_retrieval_and_answered_in_chinese(make_agent):
    agent = make_agent(prob=0.5)
    review = asyncio.run(agent.review("一条啤酒广告想在周六早上的卡通时段播出"))
    judge = agent.judge
    assert judge.translated.startswith("一条啤酒广告")
    case = judge.decided
    assert case.query_text().startswith("A beer advert")  # LightRAG searches in English
    assert "Original note (Simplified Chinese): 一条啤酒广告" in case.judge_text()
    assert review.language == "zh"
    text = render_markdown(review)
    assert "⛔ 不可播出" in text and "已转人工复核" in text
    assert "结论概率 0.50 低于阈值 0.85" in text  # routing reason rendered in Chinese


def test_answer_language_instruction_reaches_the_judge(settings):
    case = Case(text="一条啤酒广告", language="zh", english_text="A beer advert")
    msgs = OpenAIJudge(client=None, settings=settings)._messages(case, RetrievalResult([]))
    user = msgs[1]["content"]
    assert "ANSWER LANGUAGE" in user and "Simplified Chinese" in user


def test_english_note_is_not_translated(make_agent):
    agent = make_agent()
    asyncio.run(agent.review("A beer advert during cartoons"))
    assert not hasattr(agent.judge, "translated")


def test_compare_matches_ids_across_languages(tmp_path):
    def write(name, rows):
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {"cases": [{"id": i, "got": g, "expected": e, "error": None} for i, g, e in rows]}
            )
        )
        return path

    a = write("en.json", [("h01", "allowed", "allowed"), ("h02", "conditional", "conditional")])
    b = write(
        "zh.json", [("zh-h01", "allowed", "allowed"), ("zh-h02", "not_allowed", "conditional")]
    )
    out = compare(a, b)
    assert out["matched_cases"] == 2 and out["consistency"] == 0.5
    assert out["differences"] == [
        {"id": "h02", "a": "conditional", "b": "not_allowed", "expected": "conditional"}
    ]
