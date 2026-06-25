from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


class ClaimGNN:
    """Fixed 2-hop message-passing encoder with optional PyG GAT when available."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, output_dim: int = 128) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self._pyg_model = None
        try:
            import torch
            from torch import nn
            from torch_geometric.nn import GATConv

            class _GAT(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.gat1 = GATConv(input_dim, hidden_dim, edge_dim=1)
                    self.gat2 = GATConv(hidden_dim, output_dim, edge_dim=1)

                def forward(self, x: Any, edge_index: Any, edge_attr: Any) -> Any:
                    x = self.gat1(x, edge_index, edge_attr=edge_attr).relu()
                    return self.gat2(x, edge_index, edge_attr=edge_attr)

            self._torch = torch
            self._pyg_model = _GAT()
            self._pyg_model.eval()
        except Exception:
            self._pyg_model = None

    def forward(self, graph: nx.DiGraph) -> dict[str, list[float]]:
        if not graph.nodes:
            return {}
        if self._pyg_model is not None and graph.edges:
            return self._forward_pyg(graph)
        return self._forward_numpy(graph)

    def _forward_pyg(self, graph: nx.DiGraph) -> dict[str, list[float]]:
        nodes = list(graph.nodes)
        node_index = {node: idx for idx, node in enumerate(nodes)}
        x = self._torch.tensor([graph.nodes[n].get("embedding", [0.0] * self.input_dim) for n in nodes], dtype=self._torch.float32)
        edge_index = self._torch.tensor(
            [[node_index[s], node_index[t]] for s, t in graph.edges],
            dtype=self._torch.long,
        ).t().contiguous()
        edge_attr = self._torch.tensor([[float(graph.edges[e].get("weight", 1.0))] for e in graph.edges], dtype=self._torch.float32)
        with self._torch.no_grad():
            out = self._pyg_model(x, edge_index, edge_attr).numpy()
        return {node: out[i].astype(float).tolist() for i, node in enumerate(nodes)}

    def _forward_numpy(self, graph: nx.DiGraph) -> dict[str, list[float]]:
        nodes = list(graph.nodes)
        embeddings = {
            node: np.asarray(graph.nodes[node].get("embedding", [0.0] * self.input_dim), dtype=np.float32)
            for node in nodes
        }
        current = embeddings
        for _ in range(2):
            updated: dict[str, np.ndarray] = {}
            for node in nodes:
                incoming = list(graph.in_edges(node, data=True))
                if not incoming:
                    updated[node] = current[node]
                    continue
                weighted = [current[src] * float(data.get("weight", 1.0)) for src, _, data in incoming]
                updated[node] = 0.5 * current[node] + 0.5 * np.mean(weighted, axis=0)
            current = updated
        return {node: vector[: self.output_dim].astype(float).tolist() for node, vector in current.items()}

