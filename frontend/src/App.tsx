import { Activity, FileText, Play } from "lucide-react";
import { FormEvent, useState } from "react";
import { ClaimGraph } from "./components/ClaimGraph";
import { ConsensusOutput } from "./components/ConsensusOutput";
import { DebateTranscript } from "./components/DebateTranscript";
import { MetricsPanel } from "./components/MetricsPanel";
import { RatingPanel } from "./components/RatingPanel";
import { useDebateStream } from "./hooks/useDebateStream";

const sampleAbstract =
  "This paper proposes a physics-informed neural operator for modeling turbulent transport. The method embeds conservation constraints into the architecture and reports improved extrapolation across Reynolds numbers. Results suggest better stability than black-box baselines, but the causal role of the physics prior and the robustness of scaling claims require careful scrutiny.";

export default function App() {
  const [abstract, setAbstract] = useState(sampleAbstract);
  const [focus, setFocus] = useState("Are the paper's central ML/physics claims supported by its evidence?");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const stream = useDebateStream(sessionId);

  async function startDebate(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    const response = await fetch("/debate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ abstract, focus })
    });
    const payload = await response.json();
    setSessionId(payload.session_id);
    setSubmitting(false);
  }

  return (
    <main className="min-h-screen">
      <div className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-slate-900">
              <Activity size={22} />
              <h1 className="text-xl font-semibold">PaperDebate</h1>
            </div>
            <p className="mt-1 text-sm text-slate-500">Multi-agent scientific paper debate with anti-sycophancy triggers.</p>
          </div>
          <div className="rounded-md bg-slate-100 px-3 py-2 text-sm text-slate-600">
            {sessionId ? `Session ${sessionId.slice(0, 8)} · ${stream.status}` : "Ready"}
          </div>
        </div>
      </div>

      <div className="mx-auto grid max-w-7xl gap-4 px-4 py-4 lg:grid-cols-[380px_1fr_360px]">
        <aside className="space-y-4">
          <form onSubmit={startDebate} className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-600">
              <FileText size={16} />
              Paper Input
            </div>
            <label className="mt-3 block text-sm font-medium text-slate-700">Abstract or paper excerpt</label>
            <textarea
              className="mt-1 min-h-48 w-full rounded-md border border-slate-200 p-3 text-sm leading-6"
              value={abstract}
              onChange={(event) => setAbstract(event.target.value)}
            />
            <label className="mt-3 block text-sm font-medium text-slate-700">Focus</label>
            <input
              className="mt-1 w-full rounded-md border border-slate-200 p-2 text-sm"
              value={focus}
              onChange={(event) => setFocus(event.target.value)}
            />
            <button
              type="submit"
              disabled={submitting || !abstract.trim()}
              className="mt-4 flex w-full items-center justify-center gap-2 rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white disabled:bg-slate-300"
            >
              <Play size={16} />
              {submitting ? "Starting..." : "Start debate"}
            </button>
          </form>
          <MetricsPanel metrics={stream.metrics} events={stream.events} />
          <RatingPanel sessionId={sessionId} completed={stream.status === "completed"} />
        </aside>

        <DebateTranscript responses={stream.responses} events={stream.events} />

        <aside className="space-y-4">
          <ClaimGraph graph={stream.graph} />
          <ConsensusOutput finalAnswer={stream.finalAnswer} latestSynthesis={stream.latestGemini} />
        </aside>
      </div>
    </main>
  );
}
