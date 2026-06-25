from __future__ import annotations

import json
import re
from typing import Protocol

from backend.llm.claude_client import ClaudeClient
from backend.llm.gemini_client import GeminiClient
from backend.llm.openai_client import OpenAIClient
from backend.pipeline.models import AgentResponse, PaperStructure


JSON_DIRECTIVE = """Return your response as a JSON object with exactly these keys:
answer (string), explanation (string, 3-5 sentences),
confidence (float 0.0-1.0), argument_type (string),
new_evidence (bool), position_changed (bool)."""

CLAUDE_SYSTEM = """You are a rigorous scientific advocate analyzing a research paper. Construct
the strongest possible case for the paper's claims using its own evidence and
standard scientific reasoning. Be specific: cite which evidence supports which
claim. Do not concede to other agents unless they introduce genuinely new
evidence or identify a specific logical error you cannot rebut.
argument_type must be one of:
empirical_support / logical_defense / evidence_citation"""

GPT4O_SYSTEM = """You are a physicist and ML scientist acting as a rigorous skeptic of a research
paper. Stress-test claims from first principles. Look for hidden assumptions,
statistical overclaims, physically inconsistent scaling arguments, symmetry
constraints asserted but not enforced, and decorative physics framing. Be
specific: reference claim numbers. Do not concede unless given a direct logical
rebuttal to your specific objection.
argument_type must be one of:
assumption_challenge / statistical_objection / physics_objection /
methodological_critique"""

GEMINI_SYSTEM = """You are a science communicator and educator. Your role is not to take a side
but to synthesize the strongest arguments from both sides and produce three
layered explanations of what this paper claims and what remains contested after
this debate round:
intuition_explanation: intuition only, no jargon
technical_explanation: methods and key caveats
frontier_explanation: frontier contribution and open questions
Update your synthesis after each round to incorporate new arguments.
argument_type must always be: synthesis

Add three additional keys to your JSON output:
intuition_explanation (string), technical_explanation (string),
frontier_explanation (string)"""


class JsonLLM(Protocol):
    async def complete_json(self, system: str, user: str) -> str: ...


class DebateAgents:
    def __init__(
        self,
        claude: JsonLLM | None = None,
        openai: JsonLLM | None = None,
        gemini: JsonLLM | None = None,
    ) -> None:
        self.clients: dict[str, JsonLLM] = {
            "claude": claude or ClaudeClient(),
            "gpt4o": openai or OpenAIClient(),
            "gemini": gemini or GeminiClient(),
        }

    async def ask(
        self,
        agent_id: str,
        *,
        paper: PaperStructure,
        focus: str,
        structural_context: str,
        round_no: int,
        prior_context: str = "",
        own_prior: AgentResponse | None = None,
        extra_directive: str = "",
        optimized_prompt: str | None = None,
    ) -> AgentResponse:
        system = system_prompt(agent_id)
        prompt = build_agent_prompt(
            agent_id=agent_id,
            paper=paper,
            focus=focus,
            structural_context=structural_context,
            prior_context=prior_context,
            own_prior=own_prior,
            extra_directive=extra_directive,
            optimized_prompt=optimized_prompt,
        )
        raw = await self.clients[agent_id].complete_json(system, prompt)
        try:
            parsed = parse_agent_json(raw)
        except ValueError:
            retry_prompt = prompt + "\n\nYour prior response was not valid JSON. Return only the required JSON object."
            raw = await self.clients[agent_id].complete_json(system, retry_prompt)
            try:
                parsed = parse_agent_json(raw)
            except ValueError:
                parsed = {
                    "answer": "unparsed",
                    "explanation": "The model returned output that could not be parsed as JSON.",
                    "confidence": 0.0,
                    "argument_type": "parse_failure",
                    "new_evidence": False,
                    "position_changed": False,
                }
        return AgentResponse(
            agent_id=agent_id,
            answer=str(parsed.get("answer", "")),
            explanation=str(parsed.get("explanation", "")),
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            argument_type=str(parsed.get("argument_type", "")),
            new_evidence=bool(parsed.get("new_evidence", False)),
            position_changed=bool(parsed.get("position_changed", False)),
            round=round_no,
            raw_response=raw,
            intuition_explanation=parsed.get("intuition_explanation"),
            technical_explanation=parsed.get("technical_explanation"),
            frontier_explanation=parsed.get("frontier_explanation"),
        )


def system_prompt(agent_id: str) -> str:
    base = {"claude": CLAUDE_SYSTEM, "gpt4o": GPT4O_SYSTEM, "gemini": GEMINI_SYSTEM}[agent_id]
    return base + "\n\n" + JSON_DIRECTIVE


def build_agent_prompt(
    *,
    agent_id: str,
    paper: PaperStructure,
    focus: str,
    structural_context: str,
    prior_context: str,
    own_prior: AgentResponse | None,
    extra_directive: str,
    optimized_prompt: str | None,
) -> str:
    prompt = optimized_prompt or "Evaluate this paper's central scientific claims through debate."
    own = own_prior.to_dict() if own_prior else "No prior answer."
    return f"""{prompt}

Original paper focus / question:
{focus or "Assess the paper's central claims."}

Paper structure:
{paper.summary()}

Structural context:
{structural_context}

Prior blind-ordered agent responses:
{prior_context or "No inter-agent visibility for round 0."}

This agent's own prior answer:
{own}

{extra_directive}

Update your position ONLY if the other agents have introduced new evidence
or identified a specific logical error. Do not agree simply because another
agent is more confident or states their answer more forcefully.
"""


def parse_agent_json(raw: str) -> dict:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("No JSON object found.")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON object.") from exc
