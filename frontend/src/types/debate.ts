export type AgentResponse = {
  agent_id: "claude" | "gpt4o" | "gemini";
  answer: string;
  explanation: string;
  confidence: number;
  argument_type: string;
  new_evidence: boolean;
  position_changed: boolean;
  round: number;
  raw_response: string;
  intuition_explanation?: string | null;
  technical_explanation?: string | null;
  frontier_explanation?: string | null;
  sycophancy_flag?: boolean;
};

export type DebateEvent =
  | { type: "agent_turn"; round: number; agent_id: string; response: AgentResponse }
  | { type: "trigger"; trigger_id: string; round: number; agents_involved: string[]; description?: string }
  | { type: "prompt_updated"; original: string; optimized: string; po_model_version: string; fallback_used: boolean }
  | { type: "consensus"; answer: string; score: number }
  | { type: "health_warning"; debate_health_score: number; round: number }
  | { type: "metrics"; payload: DebateMetrics }
  | { type: "graph"; payload: ClaimGraphPayload };

export type DebateMetrics = {
  claim_coverage_rate: number;
  sycophancy_pct: number;
  novelty_rate: number;
  rounds_to_consensus?: number | null;
  team_answer_score: number;
  debate_health_score: number[];
  po_model_version?: string | null;
  po_fallback_used: boolean;
  trigger_log: Array<{ trigger_ids: string[]; round: number; description: string }>;
};

export type ClaimGraphPayload = {
  nodes: Array<{ data: { id: string; label: string; text: string; kind: string; pagerank: number } }>;
  edges: Array<{ data: { id: string; source: string; target: string; type: string; weight: number } }>;
};
