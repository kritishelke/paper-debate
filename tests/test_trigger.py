from backend.pipeline.models import AgentResponse
from backend.pipeline.trigger import check_triggers


def response(agent: str, answer: str, explanation: str, round_no: int) -> AgentResponse:
    return AgentResponse(agent, answer, explanation, 0.7, "methodological_critique", False, False, round_no, "{}")


def test_t0_stall_trigger() -> None:
    previous = [
        response("claude", "yes", "same detailed explanation", 0),
        response("gpt4o", "no", "same skeptical explanation", 0),
        response("gemini", "mixed", "synthesis", 0),
    ]
    current = [
        response("claude", "yes", "same detailed explanation", 1),
        response("gpt4o", "no", "same skeptical explanation", 1),
        response("gemini", "mixed", "different synthesis", 1),
    ]
    result = check_triggers(1, previous, current)
    assert result.fired
    assert "t0" in result.trigger_ids


def test_t1_swap_trigger() -> None:
    previous = [response("claude", "yes", "a", 0), response("gpt4o", "no", "b", 0)]
    current = [response("claude", "no", "c", 1), response("gpt4o", "yes", "d", 1)]
    result = check_triggers(1, previous, current)
    assert "t1" in result.trigger_ids


def test_t2_copy_trigger_marks_sycophancy() -> None:
    previous = [
        response("claude", "support", "the same copied explanation", 0),
        response("gpt4o", "reject", "skeptic explanation", 0),
    ]
    current = [response("gpt4o", "support", "the same copied explanation", 1)]
    result = check_triggers(1, previous, current)
    assert "t2" in result.trigger_ids
    assert current[0].sycophancy_flag

