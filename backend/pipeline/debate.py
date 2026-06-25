from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from backend.llm.openai_client import OpenAIClient
from backend.pipeline.agents import DebateAgents
from backend.pipeline.gnn import ClaimGNN
from backend.pipeline.graph import build_claim_graph, cytoscape_json, structural_context, update_edge_weights_from_round
from backend.pipeline.ingest import ingest_input
from backend.pipeline.memory import STORE, consensus_answer, new_session_id, update_metrics_after_round
from backend.pipeline.models import AgentResponse, DebateRecord
from backend.pipeline.prompt_optimizer import optimize_prompt
from backend.pipeline.team_answer import build_team_answer
from backend.pipeline.trigger import check_triggers


AGENT_ORDER = ["claude", "gpt4o", "gemini"]
DEFAULT_PROMPT = (
    "Evaluate the paper's central claims. Maintain independent reasoning, cite specific claims, "
    "and update positions only for new evidence or a precise logical rebuttal."
)


async def create_debate_session(
    *,
    pdf_bytes: str | None = None,
    abstract: str | None = None,
    focus: str = "",
    agents: DebateAgents | None = None,
    openai_client: OpenAIClient | None = None,
) -> str:
    session_id = new_session_id()
    paper = await ingest_input(pdf_bytes=pdf_bytes, abstract=abstract, focus=focus)
    record = DebateRecord(session_id=session_id, paper=paper, focus=focus, original_prompt=DEFAULT_PROMPT, status="queued")
    STORE.create(record)
    asyncio.create_task(run_debate(session_id, agents=agents, openai_client=openai_client))
    return session_id


async def run_debate(
    session_id: str,
    *,
    agents: DebateAgents | None = None,
    openai_client: OpenAIClient | None = None,
    n_max: int = 5,
) -> DebateRecord:
    record = STORE.get(session_id)
    record.status = "running"
    agents = agents or DebateAgents()
    openai_client = openai_client or OpenAIClient()
    graph = await build_claim_graph(record.paper, openai_client=openai_client)
    ClaimGNN().forward(graph)
    record.graph = cytoscape_json(graph)
    await STORE.emit(session_id, {"type": "graph", "payload": record.graph})

    context = structural_context(graph)
    round_zero = await asyncio.gather(
        *[
            agents.ask(
                agent_id,
                paper=record.paper,
                focus=record.focus,
                structural_context=context,
                round_no=0,
                optimized_prompt=record.original_prompt,
            )
            for agent_id in AGENT_ORDER
        ]
    )
    for response in round_zero:
        record.add_response(response)
        await STORE.emit(session_id, {"type": "agent_turn", "round": 0, "agent_id": response.agent_id, "response": response.to_dict()})
    health = update_metrics_after_round(record, 0)
    if health < 0.4:
        await STORE.emit(session_id, {"type": "health_warning", "debate_health_score": health, "round": 0})

    optimized_prompt = record.original_prompt
    phase3_used = False
    stop_after_round = False
    for round_no in range(1, n_max + 1):
        current_round: list[AgentResponse] = []
        mapping = blind_mapping()
        record.blind_mappings.append(mapping)
        for agent_id in AGENT_ORDER:
            prior_context = blind_prior_context(record.rounds.get(round_no - 1, []), mapping, exclude=agent_id)
            prior_context += "\n" + blind_prior_context(current_round, mapping, exclude=agent_id)
            own_prior = latest_for_agent(record, agent_id)
            extra = anti_sycophancy_directives(record, agent_id)
            response = await agents.ask(
                agent_id,
                paper=record.paper,
                focus=record.focus,
                structural_context=structural_context(graph),
                round_no=round_no,
                prior_context=prior_context,
                own_prior=own_prior,
                extra_directive=extra,
                optimized_prompt=optimized_prompt,
            )
            current_round.append(response)
            record.add_response(response)
            await STORE.emit(session_id, {"type": "agent_turn", "round": round_no, "agent_id": agent_id, "response": response.to_dict()})

        delta = update_edge_weights_from_round(graph, current_round)
        record.metrics.gnn_edge_confidence_delta += delta
        ClaimGNN().forward(graph)
        record.graph = cytoscape_json(graph)
        previous_round = record.rounds.get(round_no - 1, [])
        trigger = check_triggers(round_no, previous_round, current_round)
        if trigger.fired:
            record.trigger_log.append(trigger)
            await STORE.emit(
                session_id,
                {
                    "type": "trigger",
                    "trigger_id": ",".join(trigger.trigger_ids),
                    "round": round_no,
                    "agents_involved": trigger.agents_involved,
                    "description": trigger.description,
                },
            )
            if not phase3_used:
                update = await optimize_prompt(
                    optimized_prompt,
                    transcript=json.dumps(record.to_transcript_json()),
                    openai_client=openai_client,
                )
                optimized_prompt = update["optimized"]
                record.prompt_updated = update
                record.metrics.po_model_version = update["po_model_version"]
                record.metrics.po_fallback_used = update["fallback_used"]
                phase3_used = True
                await STORE.emit(session_id, {"type": "prompt_updated", **update})
            else:
                stop_after_round = True

        answer = consensus_answer(current_round)
        health = update_metrics_after_round(record, round_no)
        await STORE.emit(session_id, {"type": "metrics", "payload": record.metrics.to_dict()})
        if health < 0.4:
            await STORE.emit(session_id, {"type": "health_warning", "debate_health_score": health, "round": round_no})
        if answer and not any(t in {"t1", "t2"} for t in trigger.trigger_ids):
            record.metrics.rounds_to_consensus = round_no
            stop_after_round = True
        if stop_after_round:
            break

    record.final_answer = await build_team_answer(record.all_responses(), openai_client=openai_client)
    record.metrics.team_answer_score = record.final_answer.score
    record.status = "completed"
    await STORE.emit(session_id, {"type": "consensus", "answer": record.final_answer.winning_answer, "score": record.final_answer.score})
    await STORE.emit(session_id, {"type": "metrics", "payload": record.metrics.to_dict()})
    await STORE.close(session_id)
    return record


def blind_mapping() -> dict[str, str]:
    labels = ["Agent A", "Agent B", "Agent C"]
    shuffled = labels[:]
    random.shuffle(shuffled)
    return dict(zip(AGENT_ORDER, shuffled))


def blind_prior_context(responses: list[AgentResponse], mapping: dict[str, str], exclude: str) -> str:
    lines: list[str] = []
    visible = [response for response in responses if response.agent_id != exclude]
    random.shuffle(visible)
    for response in visible:
        label = mapping.get(response.agent_id, "Agent")
        lines.append(
            f"{label} round {response.round}: answer={response.answer}; confidence={response.confidence}; "
            f"argument_type={response.argument_type}; explanation={response.explanation}"
        )
    return "\n".join(lines)


def latest_for_agent(record: DebateRecord, agent_id: str) -> AgentResponse | None:
    matches = [response for response in record.all_responses() if response.agent_id == agent_id]
    return matches[-1] if matches else None


def anti_sycophancy_directives(record: DebateRecord, agent_id: str) -> str:
    responses = [r for r in record.all_responses() if r.agent_id == agent_id]
    directives: list[str] = []
    if len(responses) >= 2:
        prev, curr = responses[-2], responses[-1]
        if curr.position_changed and not curr.new_evidence:
            directives.append(
                "In the last round you changed your position without citing new evidence. "
                "Either maintain your original position with a rebuttal, or explicitly identify the new logical reason."
            )
    if responses:
        latest = responses[-1]
        prior = [r.explanation for r in record.all_responses() if r is not latest]
        from backend.pipeline.text_utils import cosine_similarity

        if prior and max(cosine_similarity(latest.explanation, text) for text in prior) > 0.85:
            directives.append(
                "Your last argument was very similar to a prior argument in this debate. "
                "Introduce a new line of reasoning, or explicitly concede the point and state precisely why."
            )
    return "\n".join(directives)

