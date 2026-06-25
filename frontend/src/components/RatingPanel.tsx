import { useState } from "react";

const fields = [
  ["debate_quality", "Debate quality"],
  ["argument_novelty", "Argument novelty"],
  ["claim_coverage", "Claim coverage"],
  ["consensus_quality", "Consensus quality"]
] as const;

export function RatingPanel({ sessionId, completed }: { sessionId: string | null; completed: boolean }) {
  const [scores, setScores] = useState<Record<string, number>>(Object.fromEntries(fields.map(([field]) => [field, 3])));
  const [notes, setNotes] = useState("");
  const [message, setMessage] = useState("");

  async function submit() {
    if (!sessionId) return;
    const response = await fetch(`/debate/${sessionId}/rating`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rater_id: "local-rater", ...scores, notes })
    });
    setMessage(response.ok ? "Rating saved" : "Rating failed");
  }

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-600">Rating</h2>
      <div className="mt-3 space-y-3">
        {fields.map(([field, label]) => (
          <label key={field} className="block text-sm">
            <div className="flex justify-between text-slate-600">
              <span>{label}</span>
              <span>{scores[field]}</span>
            </div>
            <input
              className="w-full accent-slate-900"
              type="range"
              min={1}
              max={5}
              value={scores[field]}
              disabled={!completed}
              onChange={(event) => setScores((current) => ({ ...current, [field]: Number(event.target.value) }))}
            />
          </label>
        ))}
        <textarea
          className="min-h-24 w-full rounded-md border border-slate-200 p-2 text-sm"
          placeholder="Notes"
          value={notes}
          disabled={!completed}
          onChange={(event) => setNotes(event.target.value)}
        />
        <button
          disabled={!completed}
          onClick={submit}
          className="w-full rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white disabled:bg-slate-300"
        >
          Submit rating
        </button>
        {message && <div className="text-sm text-slate-600">{message}</div>}
      </div>
    </section>
  );
}
