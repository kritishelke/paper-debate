import asyncio

from backend.pipeline.models import AgentResponse
from backend.pipeline.team_answer import build_team_answer


class JudgeClient:
    async def judge(self, transcript: str, winning_answer: str) -> str:
        return '{"validated": true, "final_answer": "%s", "judge_explanation": "ok", "confidence": 0.9}' % winning_answer


def r(agent: str, answer: str, confidence: float, round_no: int) -> AgentResponse:
    return AgentResponse(agent, answer, "explanation", confidence, "x", True, False, round_no, "{}")


def test_team_answer_prefers_high_confidence_consistent_group() -> None:
    result = asyncio.run(build_team_answer(
        [
            r("claude", "accept", 0.9, 0),
            r("gpt4o", "reject", 0.4, 0),
            r("gemini", "accept", 0.8, 0),
            r("claude", "accept", 0.9, 1),
            r("gpt4o", "accept", 0.7, 1),
            r("gemini", "accept", 0.8, 1),
        ],
        openai_client=JudgeClient(),
    ))
    assert result.winning_answer == "accept"
    assert result.score > 0
    assert result.judge_confidence == 0.9
