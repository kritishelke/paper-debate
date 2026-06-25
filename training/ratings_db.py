from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.pipeline.models import DebateRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS debates (
    session_id TEXT PRIMARY KEY,
    paper_title TEXT,
    paper_abstract TEXT,
    original_prompt TEXT,
    debate_transcript TEXT,
    trigger_log TEXT,
    sycophancy_pct REAL,
    rounds_to_consensus INTEGER,
    final_answer TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    rater_id TEXT,
    debate_quality INTEGER,
    argument_novelty INTEGER,
    claim_coverage INTEGER,
    consensus_quality INTEGER,
    notes TEXT,
    created_at TEXT,
    FOREIGN KEY (session_id) REFERENCES debates(session_id)
);
"""


class RatingsDB:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.getenv("RATINGS_DB_PATH", "training/ratings.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_debate(self, record: DebateRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO debates (
                    session_id, paper_title, paper_abstract, original_prompt, debate_transcript,
                    trigger_log, sycophancy_pct, rounds_to_consensus, final_answer, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.paper.title,
                    record.paper.abstract,
                    record.original_prompt,
                    json.dumps(record.to_transcript_json()),
                    json.dumps([trigger.to_dict() for trigger in record.trigger_log]),
                    record.metrics.sycophancy_pct,
                    record.metrics.rounds_to_consensus,
                    record.final_answer.winning_answer if record.final_answer else "",
                    now_iso(),
                ),
            )

    def add_rating(self, session_id: str, rating: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ratings (
                    session_id, rater_id, debate_quality, argument_novelty,
                    claim_coverage, consensus_quality, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    rating["rater_id"],
                    rating["debate_quality"],
                    rating["argument_novelty"],
                    rating["claim_coverage"],
                    rating["consensus_quality"],
                    rating.get("notes", ""),
                    now_iso(),
                ),
            )

    def rated_debates(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT d.*, r.rater_id, r.debate_quality, r.argument_novelty,
                       r.claim_coverage, r.consensus_quality, r.notes
                FROM debates d JOIN ratings r ON d.session_id = r.session_id
                ORDER BY d.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def rating_count(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0])

    def training_status(self) -> dict[str, Any]:
        n_ratings = self.rating_count()
        output = Path("training/po_training_data.jsonl")
        n_pairs = 0
        n_synthetic = 0
        if output.exists():
            for line in output.read_text().splitlines():
                if line.strip():
                    n_pairs += 1
                    if '"synthetic": true' in line:
                        n_synthetic += 1
        thresholds = [10, 30, 100]
        next_threshold = next((t for t in thresholds if n_ratings < t), None)
        return {
            "n_ratings": n_ratings,
            "n_training_pairs": n_pairs,
            "n_synthetic_pairs": n_synthetic,
            "next_retrain_threshold": next_threshold,
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
