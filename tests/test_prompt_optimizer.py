import asyncio

from backend.pipeline.prompt_optimizer import optimize_prompt


class SimilarThenFallbackClient:
    async def complete_text(self, system: str, user: str, model: str | None = None) -> str:
        if model:
            return "Evaluate claims independently."
        return "Evaluate claims independently, require role-specific rebuttals, and forbid copying without new evidence."


def test_prompt_optimizer_uses_fallback_when_too_similar(monkeypatch) -> None:
    monkeypatch.setenv("PO_MODEL_ID", "ft:test")
    result = asyncio.run(optimize_prompt("Evaluate claims independently.", openai_client=SimilarThenFallbackClient()))
    assert result["fallback_used"] is True
    assert "role-specific" in result["optimized"]
