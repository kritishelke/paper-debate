from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.pipeline.text_utils import cosine_similarity
from training.ratings_db import RatingsDB


SYSTEM = (
    "You are a prompt optimization model for multi-agent scientific debate systems. "
    "Given a prompt that led to low-quality debate (sycophancy, stalling, shallow reasoning), "
    "rewrite it to produce more rigorous, independent, evidence-based debate. Return only the improved prompt."
)


def quality_score(row: dict[str, Any]) -> float:
    return (
        row["debate_quality"] * 0.40
        + row["argument_novelty"] * 0.30
        + row["claim_coverage"] * 0.20
        + row["consensus_quality"] * 0.10
    )


def is_good(row: dict[str, Any]) -> bool:
    return quality_score(row) >= 4.0 and row["debate_quality"] >= 4


def is_bad(row: dict[str, Any]) -> bool:
    return quality_score(row) <= 2.5 or row["debate_quality"] <= 2 or float(row["sycophancy_pct"] or 0) > 30


def prepare_training_data(
    *,
    db_path: str | Path | None = None,
    output_path: str | Path = "training/po_training_data.jsonl",
) -> Path:
    db = RatingsDB(db_path)
    rows = db.rated_debates()
    good = [row for row in rows if is_good(row)]
    bad = [row for row in rows if is_bad(row)]
    pairs: list[dict[str, Any]] = []
    for bad_row in bad:
        match = most_similar_good(bad_row, good)
        if match:
            optimized = match["original_prompt"]
            synthetic = False
        else:
            optimized = synthetic_rewrite(bad_row)
            synthetic = True
        pairs.append(training_pair(bad_row["original_prompt"], optimized, synthetic=synthetic))
    if len(pairs) < 30:
        existing_bad_ids = {row["session_id"] for row in bad}
        for row in rows:
            if row["session_id"] in existing_bad_ids:
                continue
            if len(pairs) >= 30:
                break
            if not is_good(row):
                pairs.append(training_pair(row["original_prompt"], synthetic_rewrite(row), synthetic=True))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(pair) for pair in pairs) + ("\n" if pairs else ""))
    return output


def most_similar_good(bad_row: dict[str, Any], good_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not good_rows:
        return None
    return max(good_rows, key=lambda row: cosine_similarity(bad_row.get("paper_abstract") or "", row.get("paper_abstract") or ""))


def synthetic_rewrite(row: dict[str, Any]) -> str:
    transcript = row.get("debate_transcript") or "[]"
    return (
        "Evaluate the paper's central claims with independent role-specific reasoning. "
        "Advocate must cite concrete paper evidence for each supported claim; Skeptic must name hidden assumptions, "
        "statistical weaknesses, and physics or ML validity concerns; Explainer must synthesize without voting. "
        "Do not copy another agent's answer unless you identify the exact new evidence or logical error that changed your mind. "
        f"Under-addressed context from prior low-quality transcript: {transcript[:1000]}"
    )


def training_pair(original_prompt: str, optimized_prompt: str, *, synthetic: bool) -> dict[str, Any]:
    pair = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": original_prompt},
            {"role": "assistant", "content": optimized_prompt},
        ]
    }
    if synthetic:
        pair["synthetic"] = True
    return pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare prompt-optimizer fine-tuning data.")
    parser.add_argument("--db", default=None)
    parser.add_argument("--output", default="training/po_training_data.jsonl")
    args = parser.parse_args()
    print(prepare_training_data(db_path=args.db, output_path=args.output))


if __name__ == "__main__":
    main()
