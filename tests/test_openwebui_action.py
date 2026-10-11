import asyncio
import importlib.util
from pathlib import Path

ACTION = Path(__file__).resolve().parents[1] / "openwebui" / "actions" / "report_wrong_verdict.py"
spec = importlib.util.spec_from_file_location("report_wrong_verdict", ACTION)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def run(body, answer=None):
    events = []

    async def emit(event):
        events.append(event)

    async def ask(_):
        return answer

    asyncio.run(mod.Action().action(body, __event_emitter__=emit, __event_call__=ask))
    return events


def test_review_number_is_found_in_both_languages():
    assert mod.REVIEW_ID.search("*Review #12 · in the review queue*").group(1) == "12"
    assert mod.REVIEW_ID.search("*审核记录 #7*").group(1) == "7"


def test_answer_without_review_number_warns():
    events = run({"id": "m1", "messages": [{"id": "m1", "role": "assistant", "content": "hi"}]})
    assert events[0]["data"]["type"] == "warning"


def test_cancelled_dialog_sends_nothing():
    body = {"id": "m1", "messages": [{"id": "m1", "role": "assistant", "content": "*Review #3*"}]}
    assert run(body, answer=False) == []


def test_unreachable_agent_reports_an_error():
    body = {"id": "m1", "messages": [{"id": "m1", "role": "assistant", "content": "*Review #3*"}]}
    action = mod.Action()
    action.valves.agent_url = "http://127.0.0.1:9"  # nothing listens on the discard port
    events = []

    async def emit(e):
        events.append(e)

    async def ask(_):
        return "too strict"

    asyncio.run(action.action(body, __event_emitter__=emit, __event_call__=ask))
    assert events[-1]["data"]["type"] == "error"
