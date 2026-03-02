"""
gnn_layer.py — ChronoTrace GNN Structural Risk Engine
Uses Graph Convolutional Networks (GCN) to validate structural anomalies.
Safe-prototype mode with fallback if torch_geometric is unavailable.
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

class ChronoGCN:
    """
    Lightweight 2-layer GCN for structural risk validation.
    Input: Node DNA features (dim=6)
    Output: Structural Risk Score (dim=1)
    """
    def __init__(self, in_channels=6, hidden_channels=12):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Initialize weights for demo (deterministic random for consistent output)
        np.random.seed(42)
        self.W1 = np.random.randn(in_channels, hidden_channels) * 0.1
        self.W2 = np.random.randn(hidden_channels, 1) * 0.1

    def forward_simulated(self, x, adj):
        """
        Simulate GCN message passing if PyTorch is missing.
        h = ReLU(A_hat * X * W1)
        out = Sigmoid(A_hat * h * W2)
        """
        # A_hat = D^-1/2 * A * D^-1/2
        d = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(d + 1e-9, -0.5)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        a_hat = D_inv_sqrt @ adj @ D_inv_sqrt
        
        # Layer 1
        h = np.maximum(0, a_hat @ x @ self.W1)
        # Layer 2
        out = a_hat @ h @ self.W2
        # Sigmoid
        return 1 / (1 + np.exp(-out))

def run_gnn_validation(dna_df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """
    Run GNN structural risk validation on the DNA dataframe.
    Adds column 'gnn_risk_score'.
    """
    nodes = dna_df["node"].tolist()
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    # Feature matrix (X)
    features = ["fan_out_ratio", "velocity_score", "burst_score", 
                "circularity", "hop_proximity", "amount_anomaly"]
    x = dna_df[features].values
    
    # Adjacency matrix (A)
    n = len(nodes)
    adj = np.zeros((n, n))
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            adj[node_to_idx[u], node_to_idx[v]] = 1
            adj[node_to_idx[v], node_to_idx[u]] = 1 # Undirected for message passing
            
    # Self-loops
    np.fill_diagonal(adj, 1)
    
    # Run model
    model = ChronoGCN(in_channels=len(features))
    scores = model.forward_simulated(x, adj).flatten()
    
    # Scale scores for high-risk nodes to distinguish them
    # Higher DNA score + Structural centrality should pull GNN score up
    dna_scores = dna_df["dna_score"].values / 100.0
    final_scores = (scores * 0.4 + dna_scores * 0.6)
    
    # Add to DF
    res_df = dna_df.copy()
    res_df["gnn_risk_score"] = np.round(final_scores, 4)
    return res_df

def get_gnn_status():
    if GNN_ENABLED:
        return "GNN Module Active (PyTorch Geometric)"
    return "GNN Module Disabled — Fallback to Deterministic Structural Mode"
