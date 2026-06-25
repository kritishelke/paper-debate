from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import mean

from backend.llm.openai_client import OpenAIClient
from backend.pipeline.models import AgentResponse, TeamAnswer
from backend.pipeline.text_utils import normalize_answer


async def build_team_answer(
    responses: list[AgentResponse],
    *,
    openai_client: OpenAIClient | None = None,
) -> TeamAnswer:
    if not responses:
        return TeamAnswer("", 0.0, {}, {}, "No responses were available.", 0.0)
    final_by_agent = latest_by_agent(responses)
    grouped: dict[str, list[AgentResponse]] = defaultdict(list)
    for response in final_by_agent.values():
        grouped[normalize_answer(response.answer)].append(response)

    total_agents = max(1, len(final_by_agent))
    all_rounds = sorted({r.round for r in responses})
    consistency_scores: dict[str, float] = {}
    group_scores: dict[str, float] = {}
    for answer_key, group in grouped.items():
        n_g = len(group)
        c_bar = mean(r.confidence for r in responses if normalize_answer(r.answer) == answer_key)
        s_r = sum(
            1 for round_no in all_rounds if any(r.round == round_no and normalize_answer(r.answer) == answer_key for r in responses)
        ) / max(1, len(all_rounds))
        consistency_scores[answer_key] = s_r
        group_scores[answer_key] = (c_bar / total_agents) * math.log(1 + n_g) * (1 + s_r)
    winning_key = max(group_scores, key=group_scores.get)
    winning_answer = grouped[winning_key][0].answer
    judge = await judge_answer(openai_client or OpenAIClient(), responses, winning_answer)
    if not judge.get("validated", True):
        winning_answer = str(judge.get("final_answer") or winning_answer)
    return TeamAnswer(
        winning_answer=winning_answer,
        score=group_scores[winning_key],
        per_agent_breakdown={agent: response.to_dict() for agent, response in final_by_agent.items()},
        consistency_scores=consistency_scores,
        judge_explanation=str(judge.get("judge_explanation", "Judge accepted the selected team answer.")),
        judge_confidence=float(judge.get("confidence", 0.0) or 0.0),
    )


def latest_by_agent(responses: list[AgentResponse]) -> dict[str, AgentResponse]:
    latest: dict[str, AgentResponse] = {}
    for response in responses:
        if response.agent_id not in latest or response.round >= latest[response.agent_id].round:
            latest[response.agent_id] = response
    return latest


async def judge_answer(client: OpenAIClient, responses: list[AgentResponse], winning_answer: str) -> dict:
    transcript = json.dumps([r.to_dict() for r in responses], indent=2)
    try:
        return json.loads(await client.judge(transcript, winning_answer))
    except Exception:
        return {
            "validated": True,
            "final_answer": winning_answer,
            "judge_explanation": "Judge fallback accepted the highest-scoring answer.",
            "confidence": 0.5,
        }

