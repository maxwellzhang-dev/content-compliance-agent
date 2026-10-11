"""Install (or update) the Open WebUI action functions in openwebui/actions/.

Usage: python openwebui/install_actions.py [--url http://localhost:3001]

Authenticates with OPENWEBUI_TOKEN if set (an admin API key from Settings > Account),
otherwise signs in as the built-in admin, which works when WEBUI_AUTH=False as in this
repo's docker-compose. Each action is enabled and made global, so the button appears
under every answer. Standard library only.
"""

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

ACTIONS = Path(__file__).parent / "actions"


def call(url: str, token: str | None, payload: dict | None = None, method: str = "POST"):
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=30) as resp:
        return json.loads(resp.read() or b"null")


def token_for(base: str) -> str:
    if os.getenv("OPENWEBUI_TOKEN"):
        return os.environ["OPENWEBUI_TOKEN"]
    return call(f"{base}/api/v1/auths/signin", None, {"email": "", "password": ""})["token"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3001")
    base = ap.parse_args().url.rstrip("/")
    token = token_for(base)
    for path in sorted(ACTIONS.glob("*.py")):
        fid = path.stem
        content = path.read_text(encoding="utf-8")
        title = re.search(r"^title:\s*(.+)$", content, re.M).group(1).strip()
        desc = re.search(r"^description:\s*(.+)$", content, re.M)
        form = {
            "id": fid,
            "name": title,
            "content": content,
            "meta": {"description": desc.group(1).strip() if desc else ""},
        }
        try:
            call(f"{base}/api/v1/functions/id/{fid}", token, method="GET")
            call(f"{base}/api/v1/functions/id/{fid}/update", token, form)
            action = "updated"
        except urllib.error.HTTPError as err:
            if err.code not in (401, 404):
                raise
            call(f"{base}/api/v1/functions/create", token, form)
            action = "created"
        state = call(f"{base}/api/v1/functions/id/{fid}", token, method="GET")
        if not state.get("is_active"):
            call(f"{base}/api/v1/functions/id/{fid}/toggle", token)
        if not state.get("is_global"):
            call(f"{base}/api/v1/functions/id/{fid}/toggle/global", token)
        print(f"{action} {fid}: active and global")


if __name__ == "__main__":
    main()
