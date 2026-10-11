"""
title: Report wrong verdict
author: Zhang Dingpu
version: 0.1.0
required_open_webui_version: 0.6.0
description: Sends a compliance-agent answer to the human review queue with a reason.
icon_url: data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNNCAxNXMxLTEgNC0xIDUgMiA4IDIgNC0xIDQtMVYzcy0xIDEtNCAxLTUtMi04LTItNCAxLTQgMXoiLz48bGluZSB4MT0iNCIgeTE9IjIyIiB4Mj0iNCIgeTI9IjE1Ii8+PC9zdmc+
"""

# Runs inside Open WebUI, not in the agent. Open WebUI shows it as a button under each
# answer. It finds the review number in the answer ("Review #12" / "审核记录 #12"), asks
# for a reason, and calls the agent's flag endpoint so the review lands in /review.

import re

import aiohttp
from pydantic import BaseModel, Field

REVIEW_ID = re.compile(r"(?:Review|审核记录) #(\d+)")


class Action:
    class Valves(BaseModel):
        agent_url: str = Field(
            default="http://agent:8000", description="Agent base URL as seen from Open WebUI"
        )
        agent_api_key: str = Field(default="local-dev-key", description="AGENT_API_KEY")

    def __init__(self):
        self.valves = self.Valves()

    async def action(self, body: dict, __event_emitter__=None, __event_call__=None):
        messages = body.get("messages", [])
        target = next((m for m in messages if m.get("id") == body.get("id")), None)
        if target is None:
            target = next((m for m in reversed(messages) if m.get("role") == "assistant"), {})
        match = REVIEW_ID.search(target.get("content", ""))

        async def notify(kind: str, text: str):
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "notification", "data": {"type": kind, "content": text}}
                )

        if not match:
            await notify("warning", "This answer has no review number to report.")
            return
        review_id = match.group(1)

        reason = ""
        if __event_call__:
            reason = await __event_call__(
                {
                    "type": "input",
                    "data": {
                        "title": f"Report review #{review_id} as wrong",
                        "message": "What is wrong with this verdict? It goes to the review queue.",
                        "placeholder": "e.g. blurring the face would fix it, so it should be conditional",
                    },
                }
            )
            if reason is None or reason is False:  # dialog cancelled
                return

        url = f"{self.valves.agent_url.rstrip('/')}/api/reviews/{review_id}/flag"
        headers = {"Authorization": f"Bearer {self.valves.agent_api_key}"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json={"reason": str(reason or "")}, headers=headers, timeout=10
                ) as resp:
                    if resp.status == 200:
                        await notify("success", f"Review #{review_id} sent to the review queue.")
                    elif resp.status == 409:
                        await notify("info", f"Review #{review_id} is already resolved.")
                    else:
                        await notify("error", f"Agent answered {resp.status}.")
        except aiohttp.ClientError as exc:
            await notify("error", f"Could not reach the agent: {exc}")
