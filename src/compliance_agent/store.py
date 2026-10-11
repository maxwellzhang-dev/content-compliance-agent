"""Every review the agent makes, in SQLite.

A review is stored whatever its route, so a decision that went out automatically can
still be reported and pulled into the queue later. Status moves:

    auto ───────(reported)──────► pending ──► approved | overridden
    human_review route ─────────► pending ──┘

Resolved reviews are the feedback set: `compliance-agent export-feedback` turns them
into eval cases, the reviewer's verdict becoming the expected answer.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .schemas import Review, Verdict

STATUSES = ("auto", "pending", "approved", "overridden")

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    scenario TEXT NOT NULL,
    route TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,           -- the Review as JSON
    images TEXT NOT NULL,            -- JSON list of image URLs (data: or http)
    flag_reason TEXT,                -- why an auto review was pulled into the queue
    reviewer_verdict TEXT,
    reviewer_clauses TEXT,           -- JSON list of clause IDs the reviewer says decide it
    reviewer_note TEXT,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS reviews_status ON reviews(status);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ReviewStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def add(self, review: Review, images: list[str]) -> int:
        status = "pending" if review.route == "human_review" else "auto"
        with self._conn() as db:
            cur = db.execute(
                "INSERT INTO reviews (created_at, scenario, route, status, payload, images)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    review.scenario,
                    review.route,
                    status,
                    review.model_dump_json(),
                    json.dumps(images),
                ),
            )
            return int(cur.lastrowid)

    def get(self, review_id: int) -> dict | None:
        with self._conn() as db:
            row = db.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        return _row(row) if row else None

    def list(self, status: str | None = None, limit: int = 50) -> list[dict]:
        sql, args = "SELECT * FROM reviews", []
        if status:
            sql += " WHERE status = ?"
            args.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as db:
            return [_row(r) for r in db.execute(sql, args).fetchall()]

    def counts(self) -> dict[str, int]:
        with self._conn() as db:
            rows = db.execute("SELECT status, COUNT(*) n FROM reviews GROUP BY status").fetchall()
        return {s: 0 for s in STATUSES} | {r["status"]: r["n"] for r in rows}

    def flag(self, review_id: int, reason: str) -> bool:
        """Pull a review into the queue. Only reviews nobody has resolved can be flagged."""
        with self._conn() as db:
            cur = db.execute(
                "UPDATE reviews SET status = 'pending', flag_reason = ?"
                " WHERE id = ? AND status IN ('auto', 'pending')",
                (reason.strip() or "reported as wrong", review_id),
            )
            return cur.rowcount == 1

    def resolve(
        self,
        review_id: int,
        *,
        approve: bool,
        verdict: Verdict | None = None,
        clauses: list[str] | None = None,
        note: str = "",
    ) -> bool:
        record = self.get(review_id)
        if record is None or record["status"] != "pending":
            return False
        model_verdict = Verdict(record["review"]["decision"]["verdict"])
        if approve:
            final, status = model_verdict, "approved"
        else:
            if verdict is None or verdict == model_verdict:
                raise ValueError("an override needs a verdict different from the agent's")
            final, status = verdict, "overridden"
        with self._conn() as db:
            db.execute(
                "UPDATE reviews SET status = ?, reviewer_verdict = ?, reviewer_clauses = ?,"
                " reviewer_note = ?, resolved_at = ? WHERE id = ?",
                (status, final.value, json.dumps(clauses or []), note.strip(), _now(), review_id),
            )
        return True


def _row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["review"] = json.loads(d.pop("payload"))
    d["images"] = json.loads(d["images"])
    d["reviewer_clauses"] = json.loads(d["reviewer_clauses"] or "[]")
    return d
