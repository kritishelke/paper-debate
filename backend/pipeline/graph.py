from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

import networkx as nx
import numpy as np

from backend.llm.openai_client import OpenAIClient
from backend.pipeline.models import PaperStructure
from backend.pipeline.text_utils import Embedder


EDGE_TYPES = ("supports", "contradicts", "requires")


async def build_claim_graph(
    paper: PaperStructure,
    *,
    openai_client: OpenAIClient | None = None,
    embedder: Embedder | None = None,
    max_edges: int = 20,
) -> nx.DiGraph:
    client = openai_client or OpenAIClient()
    embedder = embedder or Embedder()
    graph = nx.DiGraph()
    nodes = []
    for kind, values in (("claim", paper.claims), ("result", paper.results), ("assumption", paper.assumptions)):
        for idx, text in enumerate(values):
            node_id = f"{kind}:{idx}"
            nodes.append((node_id, text, kind, idx))
    embeddings = embedder.encode([text for _, text, _, _ in nodes]) if nodes else np.empty((0, 384))
    for i, (node_id, text, kind, idx) in enumerate(nodes):
        graph.add_node(node_id, text=text, kind=kind, index=idx, embedding=embeddings[i].tolist())

    pairs = [
        (left, right)
        for left in graph.nodes
        for right in graph.nodes
        if left != right and graph.nodes[left]["kind"] in {"claim", "result", "assumption"}
    ]
    detections = await asyncio.gather(
        *(client.detect_edge(graph.nodes[a]["text"], graph.nodes[b]["text"]) for a, b in pairs),
        return_exceptions=True,
    )
    candidate_edges: list[tuple[float, str, str, str]] = []
    for (source, target), detected in zip(pairs, detections):
        if isinstance(detected, Exception):
            continue
        confidence = float(detected.get("confidence", 0.0) or 0.0)
        if confidence <= 0.6:
            continue
        for edge_type in EDGE_TYPES:
            if detected.get(edge_type):
                candidate_edges.append((confidence, source, target, edge_type))
    for confidence, source, target, edge_type in sorted(candidate_edges, reverse=True)[:max_edges]:
        graph.add_edge(source, target, type=edge_type, weight=float(max(0.0, min(1.0, confidence))), initial_weight=float(confidence))
    return graph


def structural_context(graph: nx.DiGraph, top_n: int = 3) -> str:
    if not graph.nodes:
        return "No claim graph was available."
    pagerank = nx.pagerank(graph, weight="weight") if graph.edges else {node: 1 / len(graph.nodes) for node in graph.nodes}
    lines: list[str] = []
    for node, _score in sorted(pagerank.items(), key=lambda item: item[1], reverse=True)[:top_n]:
        data = graph.nodes[node]
        downstream = len(nx.descendants(graph, node))
        incoming_support = sum(1 for src, _ in graph.in_edges(node) if graph.edges[src, node].get("type") == "supports")
        contradictions = [
            src for src, _ in graph.in_edges(node) if graph.edges[src, node].get("type") == "contradicts"
        ]
        label = f"{data['kind']} {data['index'] + 1}"
        if downstream:
            lines.append(f"{label} is load-bearing: {downstream} downstream nodes depend on it.")
        elif contradictions and not incoming_support:
            lines.append(f"{label} has no incoming support edges and is contradicted by {contradictions[0]}.")
        else:
            lines.append(f"{label} is central to the structure: {data['text'][:160]}")
    return "\n".join(lines)


def update_edge_weights_from_round(graph: nx.DiGraph, responses: list[Any]) -> float:
    if not graph.edges:
        return 0.0
    before = [float(data.get("weight", 0.0)) for _, _, data in graph.edges(data=True)]
    for response in responses:
        text = f"{response.answer} {response.explanation}".lower()
        claim_indexes = claim_refs(text)
        for idx in claim_indexes:
            node_id = f"claim:{idx - 1}"
            if node_id not in graph:
                continue
            for _, target, data in graph.out_edges(node_id, data=True):
                if response.agent_id == "gpt4o" and "objection" in response.argument_type and data.get("type") == "supports":
                    data["weight"] = max(0.0, float(data.get("weight", 0.0)) - 0.1)
                if response.agent_id == "claude" and data.get("type") == "supports":
                    data["weight"] = min(1.0, float(data.get("weight", 0.0)) + 0.05)
    after = [float(data.get("weight", 0.0)) for _, _, data in graph.edges(data=True)]
    return float(np.mean([abs(a - b) for a, b in zip(after, before)]))


def claim_refs(text: str) -> list[int]:
    import re

    return [int(match) for match in re.findall(r"claim\s*#?(\d+)", text)]


def cytoscape_json(graph: nx.DiGraph) -> dict[str, list[dict[str, Any]]]:
    pagerank = nx.pagerank(graph, weight="weight") if graph.nodes and graph.edges else {node: 1.0 for node in graph.nodes}
    nodes = [
        {
            "data": {
                "id": node,
                "label": f"{data.get('kind')} {data.get('index', 0) + 1}",
                "text": data.get("text", ""),
                "kind": data.get("kind", ""),
                "pagerank": pagerank.get(node, 0.0),
            }
        }
        for node, data in graph.nodes(data=True)
    ]
    edges = [
        {
            "data": {
                "id": f"{source}->{target}",
                "source": source,
                "target": target,
                "type": data.get("type", "supports"),
                "weight": data.get("weight", 0.0),
            }
        }
        for source, target, data in graph.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}

