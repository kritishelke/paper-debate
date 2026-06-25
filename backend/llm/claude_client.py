from __future__ import annotations

import json
import os


class ClaudeClient:
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    async def complete_json(self, system: str, user: str) -> str:
        if self.api_key:
            try:
                import anthropic

                client = anthropic.AsyncAnthropic(api_key=self.api_key)
                response = await client.messages.create(
                    model=self.model,
                    max_tokens=1800,
                    temperature=0.2,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return response.content[0].text
            except Exception as exc:
                return json.dumps({"answer": "api_error", "explanation": str(exc), "confidence": 0.0})
        return json.dumps(
            {
                "answer": "support_with_caveats",
                "explanation": "The paper's internal evidence gives a reasonable basis for its main claims. The strongest case is that multiple reported results point in the same direction rather than relying on a single observation. Remaining limits should be framed as boundaries, not fatal flaws, unless a specific contradiction is shown.",
                "confidence": 0.72,
                "argument_type": "empirical_support",
                "new_evidence": True,
                "position_changed": False,
            }
        )

