import type { DebateEvent, DebateMetrics } from "../types/debate";

export function MetricsPanel({ metrics, events }: { metrics: DebateMetrics | null; events: DebateEvent[] }) {
  const healthScores = metrics?.debate_health_score ?? [];
  const health = healthScores.length ? healthScores[healthScores.length - 1] : 1;
  const healthClass = health > 0.7 ? "text-emerald-700" : health >= 0.4 ? "text-amber-700" : "text-red-700";
  const promptUpdate = [...events].reverse().find((event) => event.type === "prompt_updated");
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-600">Metrics</h2>
      {health < 0.4 && <div className="mt-3 rounded-md bg-red-100 p-2 text-sm text-red-800">Debate health warning</div>}
      <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
        <Metric label="Sycophancy" value={`${(metrics?.sycophancy_pct ?? 0).toFixed(1)}%`} />
        <Metric label="Coverage" value={`${Math.round((metrics?.claim_coverage_rate ?? 0) * 100)}%`} />
        <Metric label="Novelty" value={`${Math.round((metrics?.novelty_rate ?? 0) * 100)}%`} />
        <Metric label="Team score" value={(metrics?.team_answer_score ?? 0).toFixed(2)} />
      </div>
      <div className={`mt-3 rounded-md bg-slate-50 p-3 text-sm font-semibold ${healthClass}`}>
        Health score {(health ?? 0).toFixed(2)}
      </div>
      <div className="mt-4 text-xs text-slate-500">
        <div>PO model: {metrics?.po_model_version ?? "not used"}</div>
        <div>Fallback: {metrics?.po_fallback_used ? "yes" : "no"}</div>
        {promptUpdate && promptUpdate.type === "prompt_updated" && <div className="mt-2 text-slate-700">Prompt optimizer activated.</div>}
      </div>
      <div className="mt-4 space-y-2">
        <div className="text-xs font-semibold uppercase text-slate-500">Trigger log</div>
        {(metrics?.trigger_log ?? []).map((trigger, index) => (
          <div key={index} className="rounded bg-slate-50 p-2 text-xs text-slate-600">
            Round {trigger.round}: {trigger.trigger_ids.join(", ")}
          </div>
        ))}
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-slate-50 p-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-900">{value}</div>
    </div>
  );
}
