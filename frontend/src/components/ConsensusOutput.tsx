import type { AgentResponse } from "../types/debate";

type FinalAnswer = {
  answer: string;
  score: number;
} | null;

const columns = [
  {
    key: "intuition_explanation",
    label: "Intuition",
    fallback: "Plain-language consensus will appear here."
  },
  {
    key: "technical_explanation",
    label: "Technical",
    fallback: "Technical consensus will appear here."
  },
  {
    key: "frontier_explanation",
    label: "Frontier",
    fallback: "Open questions and research-level implications will appear here."
  }
] as const;

export function ConsensusOutput({ finalAnswer, latestSynthesis }: { finalAnswer: FinalAnswer; latestSynthesis?: AgentResponse }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-600">Final Output</h2>
        {finalAnswer && (
          <span className="rounded bg-emerald-100 px-2 py-1 text-xs font-medium text-emerald-800">
            Consensus score {finalAnswer.score.toFixed(3)}
          </span>
        )}
      </div>

      <div className="mt-3 rounded-md border border-emerald-100 bg-emerald-50 p-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-emerald-700">General Consensus</div>
        <p className="mt-2 text-sm leading-6 text-emerald-950">
          {finalAnswer?.answer ?? "The final consensus will appear after the debate completes."}
        </p>
      </div>

      <div className="mt-3 grid gap-3 md:grid-cols-3">
        {columns.map((column) => (
          <article key={column.key} className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-600">{column.label}</h3>
            <p className="mt-2 text-sm leading-6 text-slate-700">{latestSynthesis?.[column.key] ?? column.fallback}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
