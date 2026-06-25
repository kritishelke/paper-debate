from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


AgentId = Literal["claude", "gpt4o", "gemini"]


@dataclass
class PaperStructure:
    claims: list[str]
    methods: list[str]
    results: list[str]
    assumptions: list[str]
    section_tags: list[str]
    title: str = "Untitled paper"
    abstract: str = ""
    raw_text: str = ""

    def summary(self) -> str:
        parts = [
            "Claims:\n" + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(self.claims)),
            "Methods:\n" + "\n".join(f"- {m}" for m in self.methods),
            "Results:\n" + "\n".join(f"- {r}" for r in self.results),
            "Assumptions:\n" + "\n".join(f"- {a}" for a in self.assumptions),
        ]
        return "\n\n".join(parts)


@dataclass
class AgentResponse:
    agent_id: str
    answer: str
    explanation: str
    confidence: float
    argument_type: str
    new_evidence: bool
    position_changed: bool
    round: int
    raw_response: str
    intuition_explanation: str | None = None
    technical_explanation: str | None = None
    frontier_explanation: str | None = None
    sycophancy_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriggerResult:
    fired: bool
    trigger_ids: list[str]
    agents_involved: list[str]
    round: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DebateMetrics:
    claim_coverage_rate: float = 0.0
    sycophancy_pct: float = 0.0
    position_drift: dict[str, dict[str, int]] = field(default_factory=dict)
    novelty_rate: float = 1.0
    rounds_to_consensus: int | None = None
    trigger_log: list[TriggerResult] = field(default_factory=list)
    gnn_edge_confidence_delta: float = 0.0
    team_answer_score: float = 0.0
    debate_health_score: list[float] = field(default_factory=list)
    human_ratings: list[dict[str, Any]] = field(default_factory=list)
    po_model_version: str | None = None
    po_fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["trigger_log"] = [t.to_dict() for t in self.trigger_log]
        return data


@dataclass
class TeamAnswer:
    winning_answer: str
    score: float
    per_agent_breakdown: dict[str, Any]
    consistency_scores: dict[str, float]
    judge_explanation: str
    judge_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DebateRecord:
    session_id: str
    paper: PaperStructure
    focus: str
    original_prompt: str
    rounds: dict[int, list[AgentResponse]] = field(default_factory=dict)
    trigger_log: list[TriggerResult] = field(default_factory=list)
    graph: dict[str, Any] = field(default_factory=dict)
    metrics: DebateMetrics = field(default_factory=DebateMetrics)
    final_answer: TeamAnswer | None = None
    prompt_updated: dict[str, Any] | None = None
    status: str = "created"
    events: list[dict[str, Any]] = field(default_factory=list)
    blind_mappings: list[dict[str, str]] = field(default_factory=list)

    def add_response(self, response: AgentResponse) -> None:
        self.rounds.setdefault(response.round, []).append(response)

    def all_responses(self) -> list[AgentResponse]:
        return [response for round_no in sorted(self.rounds) for response in self.rounds[round_no]]

    def to_transcript_json(self) -> list[dict[str, Any]]:
        return [
            {"round": round_no, "responses": [r.to_dict() for r in responses]}
            for round_no, responses in sorted(self.rounds.items())
        ]
