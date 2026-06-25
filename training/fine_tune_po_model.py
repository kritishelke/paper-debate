from __future__ import annotations

import argparse
import time
from pathlib import Path

from backend.pipeline.prompt_optimizer import save_po_model_id


def start_fine_tune(training_path: str = "training/po_training_data.jsonl") -> str:
    from openai import OpenAI

    client = OpenAI()
    path = Path(training_path)
    file = client.files.create(file=path.open("rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(
        training_file=file.id,
        model="gpt-4o-2024-08-06",
        hyperparameters={"n_epochs": 3},
    )
    return job.id


def poll_and_store(job_id: str, interval: int = 30) -> str:
    from openai import OpenAI

    client = OpenAI()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        if job.status in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(interval)
    if job.status != "succeeded" or not job.fine_tuned_model:
        raise RuntimeError(f"Fine-tune did not succeed: {job.status}")
    save_po_model_id(job.fine_tuned_model, version=job.id)
    return job.fine_tuned_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the prompt optimizer model.")
    parser.add_argument("--training-data", default="training/po_training_data.jsonl")
    parser.add_argument("--poll", action="store_true")
    args = parser.parse_args()
    job_id = start_fine_tune(args.training_data)
    print(job_id)
    if args.poll:
        print(poll_and_store(job_id))


if __name__ == "__main__":
    main()

