from __future__ import annotations

import json
import os
from typing import Any


class OpenAIClient:
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    async def complete_json(self, system: str, user: str, model: str | None = None) -> str:
        if self.api_key:
            try:
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=self.api_key)
                response = await client.chat.completions.create(
                    model=model or self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                return response.choices[0].message.content or "{}"
            except Exception as exc:
                return json.dumps({"answer": "api_error", "explanation": str(exc), "confidence": 0.0})
        return json.dumps(self._offline_json(user))

    async def complete_text(self, system: str, user: str, model: str | None = None) -> str:
        if self.api_key:
            try:
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=self.api_key)
                response = await client.chat.completions.create(
                    model=model or self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.3,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                return f"API error: {exc}"
        return "Clarify the target claims, require independent evidence, and ask each agent to justify any position change."

    async def judge(self, transcript: str, winning_answer: str) -> str:
        system = "You are a scientific judge. Return only valid JSON."
        user = (
            "Given this debate transcript and the proposed winning answer, assess whether it is valid.\n\n"
            f"Transcript:\n{transcript}\n\nWinning answer:\n{winning_answer}\n\n"
            'Return JSON: {"validated": bool, "final_answer": str, "judge_explanation": str, "confidence": float}'
        )
        return await self.complete_json(system, user)

    async def detect_edge(self, claim_a: str, claim_b: str) -> dict[str, Any]:
        system = "You classify scientific claim relationships. Return only valid JSON."
        user = (
            f"Claim A: {claim_a}\nClaim B: {claim_b}\n"
            "Does claim A provide evidence for claim B? Does claim A contradict claim B? "
            'Is claim B only valid if claim A is true? Return JSON: {"supports": bool, '
            '"contradicts": bool, "requires": bool, "confidence": float}'
        )
        try:
            return json.loads(await self.complete_json(system, user))
        except json.JSONDecodeError:
            return self._offline_edge(claim_a, claim_b)

    def _offline_json(self, user: str) -> dict[str, Any]:
        lower = user.lower()
        if "return json: {supports" in lower:
            return self._offline_edge(user[:120], user[-120:])
        if "intuition_explanation" in lower or "explainer" in lower:
            return {
                "answer": "mixed",
                "explanation": "The paper's core claim is plausible but still depends on assumptions that deserve testing. The strongest support is the reported result pattern, while the main weakness is whether the method isolates the claimed mechanism. The debate should keep separating evidence from interpretation.",
                "confidence": 0.66,
                "argument_type": "synthesis",
                "new_evidence": True,
                "position_changed": False,
                "intuition_explanation": "The paper may be onto something, but some pieces still need checking.",
                "technical_explanation": "The methods and results are directionally supportive, with caveats around assumptions and evaluation design.",
                "frontier_explanation": "The contribution is promising if the claimed mechanism survives stronger ablations and independent validation.",
            }
        return {
            "answer": "mixed",
            "explanation": "The evidence supports a cautious interpretation rather than full acceptance. Several claims need stronger links between method, result, and assumption. A defensible answer should distinguish what the paper demonstrates from what it suggests.",
            "confidence": 0.62,
            "argument_type": "methodological_critique",
            "new_evidence": True,
            "position_changed": False,
        }

    def _offline_edge(self, claim_a: str, claim_b: str) -> dict[str, Any]:
        a = set(claim_a.lower().split())
        b = set(claim_b.lower().split())
        overlap = len(a & b) / max(1, len(a | b))
        return {
            "supports": overlap > 0.08,
            "contradicts": any(word in a for word in {"not", "fails", "contradict"}) and overlap > 0.04,
            "requires": any(word in b for word in {"requires", "depends", "assumes"}),
            "confidence": min(0.9, 0.45 + overlap * 2),
        }
