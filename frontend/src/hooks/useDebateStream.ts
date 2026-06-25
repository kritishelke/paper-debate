import { useEffect, useMemo, useState } from "react";
import type { AgentResponse, ClaimGraphPayload, DebateEvent, DebateMetrics } from "../types/debate";

export function useDebateStream(sessionId: string | null) {
  const [events, setEvents] = useState<DebateEvent[]>([]);
  const [responses, setResponses] = useState<AgentResponse[]>([]);
  const [metrics, setMetrics] = useState<DebateMetrics | null>(null);
  const [graph, setGraph] = useState<ClaimGraphPayload>({ nodes: [], edges: [] });
  const [finalAnswer, setFinalAnswer] = useState<{ answer: string; score: number } | null>(null);
  const [status, setStatus] = useState<"idle" | "running" | "completed" | "error">("idle");

  useEffect(() => {
    if (!sessionId) return;
    setEvents([]);
    setResponses([]);
    setMetrics(null);
    setFinalAnswer(null);
    setStatus("running");
    const source = new EventSource(`/debate/${sessionId}/stream`);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as DebateEvent;
      setEvents((current) => [...current, event]);
      if (event.type === "agent_turn") setResponses((current) => [...current, event.response]);
      if (event.type === "metrics") setMetrics(event.payload);
      if (event.type === "graph") setGraph(event.payload);
      if (event.type === "consensus") {
        setFinalAnswer({ answer: event.answer, score: event.score });
        setStatus("completed");
        source.close();
      }
    };
    source.onerror = () => {
      setStatus((current) => (current === "completed" ? current : "error"));
      source.close();
    };
    return () => source.close();
  }, [sessionId]);

  const latestGemini = useMemo(
    () => [...responses].reverse().find((response) => response.agent_id === "gemini"),
    [responses]
  );

  return { events, responses, metrics, graph, finalAnswer, latestGemini, status };
}

