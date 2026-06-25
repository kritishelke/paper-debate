import cytoscape from "cytoscape";
import { useEffect, useRef, useState } from "react";
import type { ClaimGraphPayload } from "../types/debate";

export function ClaimGraph({ graph }: { graph: ClaimGraphPayload }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const graphStyle = [
      {
        selector: "node",
        style: {
          label: "data(label)",
          "background-color": "#334155",
          color: "#0f172a",
          "font-size": 10,
          "text-valign": "bottom",
          "text-margin-y": 6,
          width: "mapData(pagerank, 0, 1, 18, 48)",
          height: "mapData(pagerank, 0, 1, 18, 48)"
        }
      },
      {
        selector: "edge",
        style: {
          width: 2,
          "curve-style": "bezier",
          "target-arrow-shape": "triangle",
          "line-color": "#94a3b8",
          "target-arrow-color": "#94a3b8",
          opacity: "mapData(weight, 0, 1, 0.15, 1)"
        }
      },
      { selector: 'edge[type = "supports"]', style: { "line-color": "#159a73", "target-arrow-color": "#159a73" } },
      { selector: 'edge[type = "contradicts"]', style: { "line-color": "#d84c3f", "target-arrow-color": "#d84c3f" } },
      { selector: 'edge[type = "requires"]', style: { "line-color": "#737b8c", "target-arrow-color": "#737b8c" } }
    ] as any;
    const cy = cytoscape({
      container: containerRef.current,
      elements: [...graph.nodes, ...graph.edges],
      style: graphStyle,
      layout: { name: "cose", animate: true, fit: true, padding: 24 }
    });
    cy.on("tap", "node", (event) => {
      const node = event.target;
      setSelected(`${node.data("label")}: ${node.data("text")}`);
    });
    cyRef.current = cy;
    return () => cy.destroy();
  }, [graph]);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-600">Claim Graph</h2>
      <div ref={containerRef} className="mt-3 h-80 rounded-md border border-slate-100 bg-slate-50" />
      <p className="mt-2 min-h-10 text-xs leading-5 text-slate-600">{selected ?? "Click a node for claim details."}</p>
    </section>
  );
}
