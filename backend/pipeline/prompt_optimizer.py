from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from backend.llm.openai_client import OpenAIClient
from backend.pipeline.text_utils import cosine_similarity


CONFIG_PATH = Path("config.yaml")
SYSTEM_PROMPT = (
    "You are a prompt optimization model for multi-agent scientific debate systems. "
    "Given a prompt that led to low-quality debate, rewrite it to produce more rigorous, "
    "independent, evidence-based debate. Return only the improved prompt."
)


async def optimize_prompt(
    original_prompt: str,
    *,
    transcript: str = "",
    openai_client: OpenAIClient | None = None,
) -> dict[str, Any]:
    client = openai_client or OpenAIClient()
    model_id = load_po_model_id()
    fallback_used = False
    optimized = ""
    if model_id:
        optimized = await client.complete_text(SYSTEM_PROMPT, original_prompt, model=model_id)
    if not optimized or cosine_similarity(optimized, original_prompt) >= 0.95:
        fallback_used = True
        optimized = await synthetic_prompt_rewrite(client, original_prompt, transcript)
    return {
        "original": original_prompt,
        "optimized": optimized.strip(),
        "po_model_version": model_id or "synthetic-fallback",
        "fallback_used": fallback_used,
    }


async def synthetic_prompt_rewrite(client: OpenAIClient, original_prompt: str, transcript: str) -> str:
    user = f"""This multi-agent debate was rated low quality or triggered sycophancy.

Original prompt given to the agents:
{original_prompt}

Debate transcript showing what went wrong:
{transcript[:6000]}

Rewrite the original prompt to:
1. Remove ambiguity in what agents are being asked to evaluate
2. Add explicit guiding steps specific to each agent's role
3. Specify output format and argument_type constraints clearly
4. Make agents less likely to copy each other's reasoning by emphasizing their distinct epistemic roles
5. Highlight the specific claims that were under-addressed

Return only the improved prompt, nothing else."""
    return await client.complete_text(SYSTEM_PROMPT, user)


def load_po_model_id() -> str | None:
    if os.getenv("PO_MODEL_ID"):
        return os.getenv("PO_MODEL_ID")
    if not CONFIG_PATH.exists():
        return None
    text = CONFIG_PATH.read_text()
    for line in text.splitlines():
        if line.startswith("po_model_id:"):
            return line.split(":", 1)[1].strip().strip('"') or None
    return None


def save_po_model_id(model_id: str, version: str | None = None) -> None:
    data = {"po_model_id": model_id, "po_model_version": version or model_id}
    CONFIG_PATH.write_text("\n".join(f"{k}: {json.dumps(v)}" for k, v in data.items()) + "\n")

