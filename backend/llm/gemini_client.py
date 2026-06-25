from __future__ import annotations

import json
import os


class GeminiClient:
    def __init__(self, model: str = "gemini-1.5-pro", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

    async def complete_json(self, system: str, user: str) -> str:
        if self.api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.model, system_instruction=system)
                response = await model.generate_content_async(user)
                return response.text
            except Exception as exc:
                return json.dumps({"answer": "api_error", "explanation": str(exc), "confidence": 0.0})
        return json.dumps(
            {
                "answer": "mixed",
                "explanation": "Both the advocate and skeptic have useful points. The paper appears to make a plausible claim, but the debate should track which claims are directly evidenced and which depend on assumptions. The most useful synthesis is cautious support pending stronger validation.",
                "confidence": 0.68,
                "argument_type": "synthesis",
                "new_evidence": True,
                "position_changed": False,
                "intuition_explanation": "The idea looks promising, but some proof is still missing.",
                "technical_explanation": "The methods support the headline result while leaving open questions about robustness and causal interpretation.",
                "frontier_explanation": "The work may be a meaningful contribution if the central mechanism remains under ablations, scaling checks, and independent replication.",
            }
        )
