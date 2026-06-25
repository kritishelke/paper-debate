import asyncio

from backend.pipeline.agents import DebateAgents
from backend.pipeline.models import PaperStructure


class RetryClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete_json(self, system: str, user: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return "not json"
        return '{"answer":"ok","explanation":"Valid parsed answer with enough detail.","confidence":0.5,"argument_type":"empirical_support","new_evidence":true,"position_changed":false}'


def test_agent_json_retry() -> None:
    client = RetryClient()
    agents = DebateAgents(claude=client, openai=client, gemini=client)
    paper = PaperStructure(["claim one"], [], [], [], ["intro"])
    response = asyncio.run(agents.ask("claude", paper=paper, focus="", structural_context="", round_no=0))
    assert response.answer == "ok"
    assert client.calls == 2
