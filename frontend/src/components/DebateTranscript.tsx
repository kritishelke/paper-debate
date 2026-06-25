import type { AgentResponse, DebateEvent } from "../types/debate";

const agentColors: Record<string, string> = {
  claude: "border-claude bg-teal-50",
  gpt4o: "border-gpt4o bg-red-50",
  gemini: "border-gemini bg-amber-50"
};

export function DebateTranscript({ responses, events }: { responses: AgentResponse[]; events: DebateEvent[] }) {
  const rounds = responses.reduce<Record<number, AgentResponse[]>>((acc, response) => {
    acc[response.round] = [...(acc[response.round] ?? []), response];
    return acc;
  }, {});
  return (
    <section className="min-h-0 rounded-lg border border-slate-200 bg-white">
      <div className="border-b border-slate-200 px-4 py-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-600">Transcript</h2>
      </div>
      <div className="max-h-[720px] space-y-4 overflow-auto p-4">
        {Object.keys(rounds).length === 0 && <p className="text-sm text-slate-500">Agent turns will stream here.</p>}
        {Object.entries(rounds).map(([round, items]) => (
          <div key={round}>
            <div className="mb-2 text-xs font-semibold uppercase text-slate-500">Round {round}</div>
            <div className="space-y-3">
              {items.map((response, index) => (
                <article
                  key={`${round}-${response.agent_id}-${index}`}
                  className={`rounded-lg border-l-4 p-3 ${agentColors[response.agent_id] ?? "border-slate-300 bg-slate-50"} ${
                    response.sycophancy_flag ? "ring-2 ring-red-300" : ""
                  }`}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold capitalize">{response.agent_id}</span>
                    <span className="rounded bg-white px-2 py-0.5 text-xs text-slate-600">{response.argument_type}</span>
                    {response.new_evidence && <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700">new evidence</span>}
                    {response.position_changed && <span className="rounded bg-violet-100 px-2 py-0.5 text-xs text-violet-700">position changed</span>}
                  </div>
                  <div className="mt-2 text-sm font-medium">{response.answer}</div>
                  <div className="mt-2 h-2 overflow-hidden rounded bg-white">
                    <div className="h-full rounded bg-slate-800" style={{ width: `${Math.round(response.confidence * 100)}%` }} />
                  </div>
                  <details className="mt-2">
                    <summary className="cursor-pointer text-xs text-slate-500">Explanation</summary>
                    <p className="mt-2 text-sm leading-6 text-slate-700">{response.explanation}</p>
                  </details>
                </article>
              ))}
            </div>
          </div>
        ))}
        {events
          .filter((event) => event.type === "trigger")
          .map((event, index) =>
            event.type === "trigger" ? (
              <div key={`trigger-${index}`} className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
                Trigger {event.trigger_id} fired in round {event.round}: {event.description}
              </div>
            ) : null
          )}
      </div>
    </section>
  );
}

