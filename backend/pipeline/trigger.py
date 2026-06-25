from __future__ import annotations

from backend.pipeline.models import AgentResponse, TriggerResult
from backend.pipeline.text_utils import cosine_similarity, normalize_answer


def check_triggers(round_no: int, previous: list[AgentResponse], current: list[AgentResponse]) -> TriggerResult:
    trigger_ids: list[str] = []
    agents: set[str] = set()
    descriptions: list[str] = []
    by_prev = {r.agent_id: r for r in previous}
    by_curr = {r.agent_id: r for r in current}

    stalled = []
    for agent_id, curr in by_curr.items():
        prev = by_prev.get(agent_id)
        if prev and normalize_answer(curr.answer) == normalize_answer(prev.answer):
            if cosine_similarity(curr.explanation, prev.explanation) > 0.85:
                stalled.append(agent_id)
    if len(stalled) >= 2:
        trigger_ids.append("t0")
        agents.update(stalled)
        descriptions.append("Majority of agents repeated prior answers with highly similar explanations.")

    ids = list(by_curr)
    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1 :]:
            left_prev = by_prev.get(left_id)
            right_prev = by_prev.get(right_id)
            if not left_prev or not right_prev:
                continue
            left_curr = by_curr[left_id]
            right_curr = by_curr[right_id]
            if (
                normalize_answer(left_curr.answer) == normalize_answer(right_prev.answer)
                and normalize_answer(right_curr.answer) == normalize_answer(left_prev.answer)
                and normalize_answer(left_prev.answer) != normalize_answer(right_prev.answer)
            ):
                trigger_ids.append("t1")
                agents.update([left_id, right_id])
                descriptions.append(f"{left_id} and {right_id} swapped answers from the prior round.")

    for curr in current:
        prev = by_prev.get(curr.agent_id)
        if not prev or normalize_answer(curr.answer) == normalize_answer(prev.answer):
            continue
        for source in previous:
            if source.agent_id == curr.agent_id:
                continue
            if normalize_answer(curr.answer) == normalize_answer(source.answer):
                if cosine_similarity(curr.explanation, source.explanation) > 0.8:
                    trigger_ids.append("t2")
                    agents.update([curr.agent_id, source.agent_id])
                    descriptions.append(f"{curr.agent_id} copied {source.agent_id}'s answer without enough independent reasoning.")

    unique_ids = sorted(set(trigger_ids), key=trigger_ids.index)
    result = TriggerResult(
        fired=bool(unique_ids),
        trigger_ids=unique_ids,
        agents_involved=sorted(agents),
        round=round_no,
        description=" ".join(descriptions) if descriptions else "No trigger fired.",
    )
    if any(t in {"t1", "t2"} for t in unique_ids):
        involved = set(result.agents_involved)
        for response in current:
            if response.agent_id in involved:
                response.sycophancy_flag = True
    return result

