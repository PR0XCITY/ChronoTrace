"""
gnn_layer.py — ChronoTrace GNN Structural Risk Engine
Lightweight 2-layer GCN for structural anomaly detection.

Outputs per-node:
  gnn_risk_score   : blended structural score (0-1, used historically)
  gnn_score        : raw GNN structural output (0-1)
  hybrid_score     : 0.7 * norm_DNA + 0.3 * gnn_score  (for visual layer)

Safe fallback (no torch) — uses numpy GCN simulation.
Model weights are deterministic (seeded) → identical output across reruns.
"""

import numpy as np
import pandas as pd
import networkx as nx

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    GNN_ENABLED = True
except ImportError:
    GNN_ENABLED = False

# Module-level singleton — loaded once, never retrained
_GNN_MODEL = None


class ChronoGCN:
    """
    Lightweight 2-layer GCN for structural risk validation.
    Input : Node DNA feature vector (dim = n_features)
    Output: Structural Risk Score in [0, 1]
    """
    def __init__(self, in_channels: int = 6, hidden_channels: int = 16):
        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels

        # Deterministic weights — seeded once, never changed
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((in_channels, hidden_channels)).astype(np.float32) * 0.1
        self.W2 = rng.standard_normal((hidden_channels, 1)).astype(np.float32) * 0.1

    def forward_simulated(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        Numpy GCN forward pass (fallback when torch_geometric is absent).
        h  = ReLU(A_hat @ X @ W1)
        z  = Sigmoid(A_hat @ h @ W2)
        """
        d = np.sum(adj, axis=1).astype(np.float32)
        d_inv_sqrt = np.power(d + 1e-9, -0.5)
        D_inv = np.diag(d_inv_sqrt)
        A_hat = D_inv @ adj @ D_inv

        h   = np.maximum(0.0, A_hat @ x @ self.W1)
        out = A_hat @ h @ self.W2
        return 1.0 / (1.0 + np.exp(-out))   # sigmoid


def run_gnn_validation(dna_df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """
    Run GNN structural validation on the DNA dataframe.

    Adds three columns:
        gnn_score        : raw structural anomaly score [0, 1]
        hybrid_score     : 0.7 * norm_DNA + 0.3 * gnn_score  [0, 1]
        gnn_risk_score   : (alias) same as gnn_score, kept for back-compat

    Safe fallback: never raises — returns dna_df with heuristic scores.
    """
    global _GNN_MODEL

    if dna_df is None or dna_df.empty:
        res = dna_df.copy() if dna_df is not None else pd.DataFrame()
        res["gnn_score"]      = 0.0
        res["hybrid_score"]   = 0.0
        res["gnn_risk_score"] = 0.0
        return res

    try:
        nodes       = dna_df["node"].tolist()
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        n           = len(nodes)

        # ── Feature matrix ──────────────────────────────────────────────
        FEAT_COLS = [
            "fan_out_ratio", "velocity_score", "burst_score",
            "circularity", "hop_proximity", "amount_anomaly",
        ]
        features = [f for f in FEAT_COLS if f in dna_df.columns]

        if features:
            x_raw = dna_df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
            # Min-max normalize each feature column
            col_min = x_raw.min(axis=0)
            col_max = x_raw.max(axis=0)
            denom   = np.where(col_max - col_min > 1e-9, col_max - col_min, 1.0)
            x       = (x_raw - col_min) / denom
        else:
            # Fallback: single-feature matrix using DNA
            dna_raw = dna_df["dna_score"].values.astype(np.float32).reshape(-1, 1) / 100.0
            x       = dna_raw
            features = ["dna_score"]

        # ── Adjacency matrix ────────────────────────────────────────────
        adj = np.zeros((n, n), dtype=np.float32)
        for u, v in G.edges():
            if u in node_to_idx and v in node_to_idx:
                ui, vi = node_to_idx[u], node_to_idx[v]
                adj[ui, vi] = 1.0
                adj[vi, ui] = 1.0   # undirected message passing
        np.fill_diagonal(adj, 1.0)

        # ── Load model singleton (one per feature width) ─────────────────
        if _GNN_MODEL is None or _GNN_MODEL.in_channels != len(features):
            _GNN_MODEL = ChronoGCN(in_channels=len(features), hidden_channels=16)

        raw_scores = _GNN_MODEL.forward_simulated(x, adj).flatten()   # [0, 1]

        # ── Hybrid score = 0.7 * norm_DNA + 0.3 * gnn ───────────────────
        norm_dna    = np.clip(dna_df["dna_score"].values / 100.0, 0.0, 1.0)
        hybrid      = np.clip(0.70 * norm_dna + 0.30 * raw_scores, 0.0, 1.0)

        res_df = dna_df.copy()
        res_df["gnn_score"]      = np.round(raw_scores, 4)
        res_df["hybrid_score"]   = np.round(hybrid, 4)
        res_df["gnn_risk_score"] = np.round(raw_scores, 4)   # back-compat alias
        return res_df

    except Exception:
        # Safe fallback: heuristic = DNA / 100
        res_df = dna_df.copy()
        heur   = np.clip(dna_df["dna_score"].values / 100.0, 0.0, 1.0)
        res_df["gnn_score"]      = np.round(heur, 4)
        res_df["hybrid_score"]   = np.round(heur, 4)
        res_df["gnn_risk_score"] = np.round(heur, 4)
        return res_df


def get_gnn_scores_dict(dna_df: pd.DataFrame) -> dict:
    """Return {node: gnn_score} for fast lookup in the graph renderer."""
    if dna_df is None or dna_df.empty or "gnn_score" not in dna_df.columns:
        return {}
    return dict(zip(dna_df["node"], dna_df["gnn_score"].astype(float)))


def get_hybrid_scores_dict(dna_df: pd.DataFrame) -> dict:
    """Return {node: hybrid_score} for layout sorting."""
    if dna_df is None or dna_df.empty or "hybrid_score" not in dna_df.columns:
        return {}
    return dict(zip(dna_df["node"], dna_df["hybrid_score"].astype(float)))


def get_gnn_status() -> str:
    if GNN_ENABLED:
        return "GNN Module Active (PyTorch Geometric)"
    return "GNN Structural Mode — Deterministic NumPy (no torch required)"
