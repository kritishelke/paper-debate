from __future__ import annotations

import argparse
import json

from training.ratings_db import RatingsDB


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a human rating for a stored PaperDebate session.")
    parser.add_argument("session_id")
    parser.add_argument("--rater-id", required=True)
    args = parser.parse_args()
    fields = [
        "debate_quality",
        "argument_novelty",
        "claim_coverage",
        "consensus_quality",
    ]
    rating = {"rater_id": args.rater_id}
    for field in fields:
        rating[field] = ask_score(field)
    rating["notes"] = input("notes (optional): ").strip()
    db = RatingsDB()
    db.add_rating(args.session_id, rating)
    print(json.dumps({"status": "ok", "session_id": args.session_id}, indent=2))


def ask_score(field: str) -> int:
    while True:
        value = input(f"{field} (1-5): ").strip()
        if value.isdigit() and 1 <= int(value) <= 5:
            return int(value)
        print("Enter an integer from 1 to 5.")


if __name__ == "__main__":
    main()
