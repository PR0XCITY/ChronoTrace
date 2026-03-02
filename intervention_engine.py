"""
intervention_engine.py — ChronoTrace Intervention & Loss Reduction Engine
Implements Tiered Freeze Logic for real-time risk mitigation.
"""

import pandas as pd
import networkx as nx

def apply_intervention(G: nx.DiGraph, ring_accounts: list, mode: str = "ring") -> dict:
    """
    Apply intervention to a set of accounts.
    - 'origin': only the primary seed node.
    - 'ring': all participating suspicious nodes.
    
    Returns a dict with:
        frozen_nodes: list
        kyc_nodes: list
        edges_removed: int
    """
    frozen = []
    kyc = []
    removed_count = 0
    
    # We rely on DNA scores already computed and stored in nodes or available via lookups
    # For now, we'll assume the caller gives us the targets and we use their state
    
    for node in ring_accounts:
        if node not in G:
            continue
            
        # Get DNA score from node attributes (assuming they were saved there)
        dna = G.nodes[node].get("dna_score", 0)
        
        if dna >= 70:
            # HIGH RISK -> FREEZE
            frozen.append(node)
            # Remove all outgoing edges to prevent cashout
            out_edges = list(G.out_edges(node))
            G.remove_edges_from(out_edges)
            removed_count += len(out_edges)
            G.nodes[node]["status"] = "Frozen"
        elif dna >= 40:
            # MEDIUM RISK -> KYC
            kyc.append(node)
            G.nodes[node]["status"] = "KYC Required"
        else:
            # LOW RISK -> PASS
            G.nodes[node]["status"] = "Approved"

    return {
        "frozen_nodes": frozen,
        "kyc_nodes": kyc,
        "edges_removed": removed_count
    }

def calculate_loss_avoided(ring_df: pd.DataFrame, frozen_nodes: list) -> float:
    """
    Heuristic: Loss avoided = Sum of DNA-weighted out-flow for frozen nodes.
    """
    if ring_df.empty or not frozen_nodes:
        return 0.0
        
    frozen_df = ring_df[ring_df["node"].isin(frozen_nodes)]
    # Heuristic: $1500 per DNA point (same as loss estimator)
    loss_avoided = float(frozen_df["dna_score"].sum()) * 1500
    return round(loss_avoided, 2)
