"""
intervention_engine.py — ChronoTrace Tiered Intervention Engine
Implements stage-aware freeze logic for real-time risk mitigation.

Tier logic (using stage labels, which are tuned to real DNA data ranges):
    HIGH   → stage 3 (Pre-Cashout) or 4 (Exit Imminent) → FREEZE
    MEDIUM → stage 1 (Compromised) or 2 (Layering)      → KYC Required
    LOW    → stage 0 (Normal)                             → Approved

Loss avoided = sum of outgoing edge weights that lead toward cashout nodes
               for all frozen nodes (raw USD, formatted only at display time).
"""

import pandas as pd
import networkx as nx


def _get_tier(stage_label: str, dna_score: float) -> str:
    """
    Determine the risk tier for a node.

    Primary: stage label (tuned to real DNA ranges).
    Secondary: raw DNA thresholds as override for very high scores.

    Returns: 'HIGH', 'MEDIUM', or 'LOW'
    """
    sl = str(stage_label).strip() if stage_label else ""

    # Stage-based primary classification
    if sl in ("Exit Imminent", "Pre-Cashout"):
        return "HIGH"
    if sl in ("Layering", "Compromised"):
        return "MEDIUM"

    # DNA-based secondary override (for cases where stage is unknown)
    dna = float(dna_score) if dna_score is not None else 0.0
    if dna >= 35:
        return "HIGH"
    if dna >= 20:
        return "MEDIUM"
    return "LOW"


def apply_intervention(
    G: nx.DiGraph,
    target_nodes: list,
    dna_lookup: dict = None,
    stage_lookup: dict = None,
    cashout_nodes: list = None,
) -> dict:
    """
    Apply tiered intervention to a set of nodes.

    Args:
        G             : transaction graph (will be mutated — pass a copy)
        target_nodes  : node IDs to evaluate (ring_accounts only)
        dna_lookup    : {node_id: dna_score float}
        stage_lookup  : {node_id: stage_label string}
        cashout_nodes : list of known cashout sink nodes

    Returns dict:
        frozen_nodes      : list[str] — HIGH risk, edges removed
        kyc_nodes         : list[str] — MEDIUM risk, flagged
        approved_nodes    : list[str] — LOW risk, pass-through
        edges_removed     : int
        loss_avoided_usd  : float — raw USD (not formatted)
        tier_breakdown    : {node: tier_label}
    """
    if dna_lookup is None:
        dna_lookup = {}
    if stage_lookup is None:
        stage_lookup = {}
    if cashout_nodes is None:
        cashout_nodes = []

    cashout_set = set(cashout_nodes)

    frozen: list = []
    kyc: list = []
    approved: list = []
    edges_removed = 0
    loss_avoided_usd = 0.0
    tier_breakdown: dict = {}

    for node in target_nodes:
        if node not in G:
            continue

        dna = float(dna_lookup.get(node, G.nodes[node].get("dna_score", 0)))
        stage = stage_lookup.get(node, G.nodes[node].get("stage_label", ""))
        tier = _get_tier(stage, dna)
        tier_breakdown[node] = tier

        if tier == "HIGH":
            frozen.append(node)
            # Compute loss avoided = outgoing edge weights toward cashout or any node
            for _, dst, data in list(G.out_edges(node, data=True)):
                w = float(data.get("weight", 0))
                # Weight it higher if it leads toward cashout
                if dst in cashout_set:
                    loss_avoided_usd += w * 1.0
                else:
                    loss_avoided_usd += w * 0.6
            # Remove outgoing edges (freeze)
            out_edges = list(G.out_edges(node))
            G.remove_edges_from(out_edges)
            edges_removed += len(out_edges)
            G.nodes[node]["status"] = "Frozen"
            G.nodes[node]["frozen"] = True

        elif tier == "MEDIUM":
            kyc.append(node)
            G.nodes[node]["status"] = "KYC Required"
            G.nodes[node]["frozen"] = False

        else:
            approved.append(node)
            G.nodes[node]["status"] = "Approved"
            G.nodes[node]["frozen"] = False

    return {
        "frozen_nodes":     frozen,
        "kyc_nodes":        kyc,
        "approved_nodes":   approved,
        "edges_removed":    edges_removed,
        "loss_avoided_usd": round(loss_avoided_usd, 2),
        "tier_breakdown":   tier_breakdown,
    }


def calculate_loss_avoided(ring_df: pd.DataFrame, frozen_nodes: list) -> float:
    """
    Heuristic fallback: Loss avoided = DNA-weighted outflow for frozen nodes.
    Returns raw USD float. Returns 0.0 safely on empty input.
    """
    if ring_df is None or ring_df.empty or not frozen_nodes:
        return 0.0

    frozen_df = ring_df[ring_df["node"].isin(frozen_nodes)]
    if frozen_df.empty:
        return 0.0

    return round(float(frozen_df["dna_score"].sum()) * 1500.0, 2)
