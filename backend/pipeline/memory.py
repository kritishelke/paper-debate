from __future__ import annotations

import asyncio
import uuid
from typing import Any

from backend.pipeline.models import AgentResponse, DebateRecord, TriggerResult
from backend.pipeline.text_utils import cosine_similarity, mean_pairwise_diversity, normalize_answer


class DebateMemory:
    def __init__(self) -> None:
        self.records: dict[str, DebateRecord] = {}
        self.queues: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}

    def create(self, record: DebateRecord) -> None:
        self.records[record.session_id] = record
        self.queues[record.session_id] = asyncio.Queue()

    def get(self, session_id: str) -> DebateRecord:
        return self.records[session_id]

    async def emit(self, session_id: str, event: dict[str, Any]) -> None:
        record = self.get(session_id)
        record.events.append(event)
        await self.queues[session_id].put(event)

    async def close(self, session_id: str) -> None:
        if session_id in self.queues:
            await self.queues[session_id].put(None)

    async def stream(self, session_id: str):
        record = self.get(session_id)
        for event in record.events:
            yield event
        queue = self.queues[session_id]
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event


STORE = DebateMemory()


def new_session_id() -> str:
    return str(uuid.uuid4())


def update_metrics_after_round(record: DebateRecord, round_no: int) -> float:
    responses = record.rounds.get(round_no, [])
    all_responses = record.all_responses()
    record.metrics.claim_coverage_rate = claim_coverage(record)
    record.metrics.position_drift = position_drift(record)
    record.metrics.sycophancy_pct = sycophancy_pct(record)
    record.metrics.novelty_rate = novelty_rate(all_responses)
    health = debate_health(responses, record.metrics.novelty_rate, record.metrics.sycophancy_pct)
    record.metrics.debate_health_score.append(health)
    record.metrics.trigger_log = record.trigger_log
    return health


def claim_coverage(record: DebateRecord) -> float:
    if not record.paper.claims:
        return 1.0
    text = " ".join(r.explanation.lower() for r in record.all_responses())
    addressed = 0
    for idx, claim in enumerate(record.paper.claims, start=1):
        keywords = set(claim.lower().split()[:8])
        if f"claim {idx}" in text or len(keywords & set(text.split())) >= 3:
            addressed += 1
    return addressed / len(record.paper.claims)


def position_drift(record: DebateRecord) -> dict[str, dict[str, int]]:
    drift: dict[str, dict[str, int]] = {}
    prior: dict[str, AgentResponse] = {}
    for response in record.all_responses():
        bucket = drift.setdefault(response.agent_id, {"justified": 0, "unjustified": 0})
        old = prior.get(response.agent_id)
        if old and normalize_answer(old.answer) != normalize_answer(response.answer):
            if response.new_evidence:
                bucket["justified"] += 1
            else:
                bucket["unjustified"] += 1
        prior[response.agent_id] = response
    return drift


def novelty_rate(responses: list[AgentResponse]) -> float:
    if not responses:
        return 1.0
    novel = 0
    prior: list[str] = []
    for response in responses:
        if not prior or max(cosine_similarity(response.explanation, text) for text in prior) < 0.85:
            novel += 1
        prior.append(response.explanation)
    return novel / len(responses)


def sycophancy_pct(record: DebateRecord) -> float:
    non_trivial_rounds = [r for r in record.rounds if r > 0]
    possible = len(non_trivial_rounds) * max(1, len({"claude", "gpt4o", "gemini"}))
    if possible == 0:
        return 0.0
    syco = sum(1 for trigger in record.trigger_log if any(t in {"t1", "t2"} for t in trigger.trigger_ids))
    return (syco / possible) * 100.0


def debate_health(responses: list[AgentResponse], current_novelty_rate: float, current_sycophancy_pct: float) -> float:
    diversity = mean_pairwise_diversity([r.explanation for r in responses])
    denominator = max(1.0, current_sycophancy_pct / 100.0)
    return max(0.0, min(1.0, (diversity * current_novelty_rate) / denominator))


def consensus_answer(responses: list[AgentResponse]) -> str | None:
    if not responses:
        return None
    answers = {normalize_answer(response.answer) for response in responses}
    return responses[0].answer if len(answers) == 1 else None

