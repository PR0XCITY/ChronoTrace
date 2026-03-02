"""
dna_engine.py — ChronoTrace Graph Intelligence Engine
Builds a directed graph from transactions and computes behavioural DNA scores
for each node using pure NetworkX + NumPy (no SciPy required).
"""

import math
from collections import defaultdict
from datetime import timedelta

import networkx as nx
import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ────────────────────────────────────────────────────────────────────────────

def build_graph(transactions: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed weighted graph from the transaction DataFrame.

    Edge attributes:
        weight    : transaction amount
        timestamp : transaction datetime
        tx_type   : type of transaction (normal/layering/cashout…)
        is_suspicious : flag from simulation
        tx_id     : unique transaction identifier
    """
    G = nx.DiGraph()

    for _, row in transactions.iterrows():
        src = row["source"]
        dst = row["target"]

        # Allow multi-edges by accumulating amounts on existing edges
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += row["amount"]
            G[src][dst]["tx_count"] = G[src][dst].get("tx_count", 1) + 1
            # Track earliest/latest timestamps
            G[src][dst]["last_ts"] = max(G[src][dst]["last_ts"], row["timestamp"])
        else:
            G.add_edge(
                src, dst,
                weight=row["amount"],
                tx_count=1,
                first_ts=row["timestamp"],
                last_ts=row["timestamp"],
                tx_type=row.get("tx_type", "normal"),
                is_suspicious=row.get("is_suspicious", False),
                tx_id=row.get("tx_id", ""),
            )

    return G


# ────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL METRIC COMPUTATIONS
# ────────────────────────────────────────────────────────────────────────────

def _compute_fan_out_ratio(G: nx.DiGraph, node: str) -> float:
    """
    Fan-out ratio = out-degree / max(in-degree, 1).
    High fan-out indicates a distributor (mule hub) node.
    """
    out_deg = G.out_degree(node)
    in_deg = max(G.in_degree(node), 1)
    return round(out_deg / in_deg, 4)


def _compute_velocity_score(G: nx.DiGraph, node: str, tx_df: pd.DataFrame) -> float:
    """
    Velocity = number of transactions involving this node in a 10-minute window.
    Normalized to [0, 1] using a reference max of 20 tx/window.
    """
    node_txs = tx_df[(tx_df["source"] == node) | (tx_df["target"] == node)].copy()
    if node_txs.empty:
        return 0.0

    node_txs = node_txs.sort_values("timestamp")
    timestamps = node_txs["timestamp"].tolist()

    if len(timestamps) < 2:
        return 0.0

    window = timedelta(minutes=10)
    max_count = 0
    j = 0
    for i in range(len(timestamps)):
        while timestamps[j] < timestamps[i] - window:
            j += 1
        max_count = max(max_count, i - j + 1)

    return min(max_count / 20.0, 1.0)


def _compute_burst_score(G: nx.DiGraph, node: str, tx_df: pd.DataFrame) -> float:
    """
    Burst score = inter-transaction time anomaly.
    Measures if transactions are suspiciously rapid (< 5 minutes apart on average).
    Returns a score in [0, 1].
    """
    node_txs = tx_df[tx_df["source"] == node].sort_values("timestamp")
    if len(node_txs) < 2:
        return 0.0

    timestamps = node_txs["timestamp"].tolist()
    gaps = [(timestamps[i + 1] - timestamps[i]).total_seconds() / 60
            for i in range(len(timestamps) - 1)]

    avg_gap = np.mean(gaps) if gaps else 999
    # Normalize: gap of 0 min → score 1.0; gap ≥ 60 min → score 0.0
    burst = max(0.0, 1.0 - avg_gap / 60.0)
    return round(burst, 4)


def _compute_hop_count(G: nx.DiGraph, node: str, cashout_nodes: list) -> int:
    """
    Estimate shortest hop count from this node to any known cashout node.
    Returns -1 if no path exists.
    """
    if not cashout_nodes:
        return -1

    min_hops = float("inf")
    for co in cashout_nodes:
        if nx.has_path(G, node, co):
            try:
                path_len = nx.shortest_path_length(G, node, co)
                min_hops = min(min_hops, path_len)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

    return int(min_hops) if min_hops != float("inf") else -1


def _compute_circularity(G: nx.DiGraph, node: str) -> float:
    """
    Circularity score — detects if a node is part of a cycle.
    Uses simple in-/out-neighbour overlap as a proxy.
    Returns a score in [0, 1].
    """
    successors = set(G.successors(node))
    predecessors = set(G.predecessors(node))
    if not successors or not predecessors:
        return 0.0
    overlap = len(successors & predecessors)
    total = len(successors | predecessors)
    return round(overlap / max(total, 1), 4)


def _compute_amount_anomaly(G: nx.DiGraph, node: str, all_amounts: np.ndarray) -> float:
    """
    Amount anomaly = z-score of this node's total outgoing amount vs all edges.
    Clamped to [0, 1].
    """
    if all_amounts.std() == 0 or G.out_degree(node) == 0:
        return 0.0
    node_total = sum(d["weight"] for _, _, d in G.out_edges(node, data=True))
    z = (node_total - all_amounts.mean()) / (all_amounts.std() + 1e-9)
    return round(min(max(z / 10.0, 0.0), 1.0), 4)


# ────────────────────────────────────────────────────────────────────────────
# DNA SCORE COMPUTATION
# ────────────────────────────────────────────────────────────────────────────

# Weights for the composite DNA score (must sum to 1.0)
DNA_WEIGHTS = {
    "fan_out_ratio":   0.20,
    "velocity_score":  0.25,
    "burst_score":     0.20,
    "circularity":     0.10,
    "amount_anomaly":  0.15,
    "hop_proximity":   0.10,
}


def compute_dna_scores(
    G: nx.DiGraph,
    transactions: pd.DataFrame,
    cashout_nodes: list = None,
) -> pd.DataFrame:
    """
    Compute behavioural DNA scores for every node in the graph.

    Returns a DataFrame with columns:
        node, fan_out_ratio, velocity_score, burst_score, circularity,
        amount_anomaly, hop_proximity, dna_score, risk_level
    """
    if cashout_nodes is None:
        cashout_nodes = []

    all_amounts = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    if len(all_amounts) == 0:
        all_amounts = np.array([0.0])

    records = []
    for node in G.nodes():
        fan_out   = _compute_fan_out_ratio(G, node)
        velocity  = _compute_velocity_score(G, node, transactions)
        burst     = _compute_burst_score(G, node, transactions)
        circular  = _compute_circularity(G, node)
        amount_an = _compute_amount_anomaly(G, node, all_amounts)
        hops      = _compute_hop_count(G, node, cashout_nodes)

        # Normalize hop proximity: closer to cashout → higher score
        if hops == -1:
            hop_prox = 0.0
        elif hops == 0:
            hop_prox = 1.0
        else:
            hop_prox = round(1.0 / hops, 4)

        # Fan-out normalization (cap at 10)
        fan_out_norm = min(fan_out / 10.0, 1.0)

        dna_score = (
            DNA_WEIGHTS["fan_out_ratio"]  * fan_out_norm  +
            DNA_WEIGHTS["velocity_score"] * velocity       +
            DNA_WEIGHTS["burst_score"]    * burst          +
            DNA_WEIGHTS["circularity"]    * circular       +
            DNA_WEIGHTS["amount_anomaly"] * amount_an      +
            DNA_WEIGHTS["hop_proximity"]  * hop_prox
        )
        dna_score = round(min(dna_score * 100, 100.0), 2)  # scale to 0–100

        # Risk band
        if dna_score >= 70:
            risk_level = "CRITICAL"
        elif dna_score >= 50:
            risk_level = "HIGH"
        elif dna_score >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        records.append({
            "node": node,
            "fan_out_ratio":  round(fan_out, 4),
            "velocity_score": round(velocity, 4),
            "burst_score":    round(burst, 4),
            "circularity":    round(circular, 4),
            "amount_anomaly": round(amount_an, 4),
            "hop_proximity":  round(hop_prox, 4),
            "hops_to_cashout": hops,
            "dna_score":      dna_score,
            "risk_level":     risk_level,
        })

    df = pd.DataFrame(records)
    df.sort_values("dna_score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ────────────────────────────────────────────────────────────────────────────
# SUSPICIOUS CLUSTER DETECTION
# ────────────────────────────────────────────────────────────────────────────

def detect_suspicious_clusters(
    G: nx.DiGraph,
    dna_df: pd.DataFrame,
    threshold: float = 40.0,
) -> list[list[str]]:
    """
    Identify clusters of high-risk nodes using weakly connected components.

    A cluster is reported if ≥ 2 nodes exceed the DNA score threshold.
    Returns a list of node-lists representing suspicious clusters.
    """
    high_risk_nodes = set(
        dna_df[dna_df["dna_score"] >= threshold]["node"].tolist()
    )

    # Subgraph of only high-risk nodes
    sub = G.subgraph(high_risk_nodes).copy()

    clusters = []
    for component in nx.weakly_connected_components(sub):
        component_list = list(component)
        if len(component_list) >= 2:
            clusters.append(component_list)

    # Sort clusters by size descending
    clusters.sort(key=len, reverse=True)
    return clusters


# ────────────────────────────────────────────────────────────────────────────
# GRAPH LAYOUT (without SciPy — uses spring layout via fruchterman_reingold)
# ────────────────────────────────────────────────────────────────────────────

def compute_layout(G: nx.DiGraph, seed: int = 42) -> dict:
    """
    Compute 2-D positions for graph nodes using Fruchterman-Reingold
    (built into NetworkX — no SciPy dependency).

    Returns a dict {node: (x, y)}.
    """
    # Use a sampled subgraph for display performance (max 300 nodes)
    if len(G.nodes) > 300:
        display_nodes = list(G.nodes)[:300]
        sub = G.subgraph(display_nodes).copy()
    else:
        sub = G

    pos = nx.spring_layout(sub, seed=seed, k=0.5, iterations=50)
    return pos


# ────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ────────────────────────────────────────────────────────────────────────────

def analyse(sim_result: dict) -> dict:
    """
    Full analysis pipeline entry point.

    Args:
        sim_result: dict returned by simulate.run_simulation()

    Returns a dict with keys:
        graph           : nx.DiGraph
        dna_df          : pd.DataFrame of DNA scores
        suspicious_clusters : list of node lists
        layout          : dict of {node: (x, y)}
        top_risks       : pd.DataFrame (top 20 by DNA score)
        summary         : dict of aggregate stats
    """
    transactions  = sim_result["transactions"]
    cashout_nodes = sim_result["cashout_nodes"]

    G        = build_graph(transactions)
    dna_df   = compute_dna_scores(G, transactions, cashout_nodes)
    clusters = detect_suspicious_clusters(G, dna_df)
    layout   = compute_layout(G)

    n_critical = int((dna_df["risk_level"] == "CRITICAL").sum())
    n_high     = int((dna_df["risk_level"] == "HIGH").sum())
    avg_dna    = float(dna_df["dna_score"].mean())
    max_dna    = float(dna_df["dna_score"].max())

    summary = {
        "total_nodes":       G.number_of_nodes(),
        "total_edges":       G.number_of_edges(),
        "n_critical":        n_critical,
        "n_high":            n_high,
        "avg_dna_score":     round(avg_dna, 2),
        "max_dna_score":     round(max_dna, 2),
        "n_clusters":        len(clusters),
        "largest_cluster":   len(clusters[0]) if clusters else 0,
    }

    return {
        "graph":                G,
        "dna_df":               dna_df,
        "suspicious_clusters":  clusters,
        "layout":               layout,
        "top_risks":            dna_df.head(20),
        "summary":              summary,
    }