import asyncio

from backend.pipeline.agents import DebateAgents
from backend.pipeline.debate import run_debate
from backend.pipeline.memory import STORE
from backend.pipeline.models import DebateRecord, PaperStructure


class SequenceClient:
    def __init__(self, payloads: list[str]) -> None:
        self.payloads = payloads
        self.index = 0

    async def complete_json(self, system: str, user: str) -> str:
        payload = self.payloads[min(self.index, len(self.payloads) - 1)]
        self.index += 1
        return payload


class MockOpenAI(SequenceClient):
    async def detect_edge(self, claim_a: str, claim_b: str) -> dict:
        return {"supports": True, "contradicts": False, "requires": False, "confidence": 0.8}

    async def complete_text(self, system: str, user: str, model: str | None = None) -> str:
        return "Optimized prompt requiring independent evidence and distinct roles."

    async def judge(self, transcript: str, winning_answer: str) -> str:
        return '{"validated": true, "final_answer": "%s", "judge_explanation": "validated", "confidence": 0.8}' % winning_answer


def payload(agent: str, answer: str, explanation: str, arg: str = "empirical_support") -> str:
    return (
        '{"answer":"%s","explanation":"%s","confidence":0.8,'
        '"argument_type":"%s","new_evidence":true,"position_changed":false,'
        '"intuition_explanation":"simple","technical_explanation":"technical","frontier_explanation":"frontier"}'
    ) % (answer, explanation, arg)


def test_full_debate_loop_with_t2_phase3_and_team_answer() -> None:
    paper = PaperStructure(
        claims=["Claim 1 says the method improves turbulent transport modeling."],
        methods=["Physics-informed neural operator."],
        results=["Improves extrapolation across Reynolds numbers."],
        assumptions=["Conservation constraints are correctly encoded."],
        section_tags=["intro"],
        abstract="Physics-informed neural operator abstract.",
    )
    record = DebateRecord("integration-session", paper, "Assess support.", "Original prompt")
    STORE.create(record)
    claude = SequenceClient([
        payload("claude", "support", "copied support explanation"),
        payload("claude", "support", "advocate adds Claim 1 support"),
        payload("claude", "support", "advocate final Claim 1 support"),
    ])
    gpt = SequenceClient([
        payload("gpt4o", "reject", "skeptic starts separate", "methodological_critique"),
        payload("gpt4o", "support", "copied support explanation", "methodological_critique"),
        payload("gpt4o", "support", "skeptic accepts after rebuttal", "methodological_critique"),
    ])
    gemini = SequenceClient([
        payload("gemini", "mixed", "synthesis starts", "synthesis"),
        payload("gemini", "mixed", "synthesis after copy", "synthesis"),
        payload("gemini", "support", "synthesis final", "synthesis"),
    ])
    result = asyncio.run(run_debate(
        "integration-session",
        agents=DebateAgents(claude=claude, openai=gpt, gemini=gemini),
        openai_client=MockOpenAI([]),
        n_max=2,
    ))
    assert result.prompt_updated is not None
    assert result.final_answer is not None
    assert result.metrics.team_answer_score > 0
    assert any("t2" in trigger.trigger_ids for trigger in result.trigger_log)
