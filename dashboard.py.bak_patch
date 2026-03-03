"""
dashboard.py — ChronoTrace | Enterprise FinCrime Intelligence Dashboard
Main Streamlit entry point for Streamlit Cloud deployment.

Run with:  streamlit run dashboard.py

Requires GEMINI_API_KEY in environment or .streamlit/secrets.toml
"""

import json
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Internal modules ─────────────────────────────────────────────────────────
import dna_engine
import gnn_layer
import intervention_engine
import blockchain_layer
import dataset_loader
import gemini_layer
import alerts as alert_engine
import stage_predictor
import graph_engine
import simulator
import database as db
import re
import streamlit.components.v1 as components

# ── Currency ──────────────────────────────────────────────────────────────────
INR_RATE = 83  # 1 USD = 83 INR

def _inr(usd: float) -> str:
    """Format a USD amount as Indian Rupees with ₹ symbol."""
    return f"₹{usd * INR_RATE:,.0f}"


def _safe_float(v) -> float:
    """
    Safely convert any value to float before INR formatting.
    Handles: raw float/int, strings with ₹ / $ / commas / trailing periods.
    """
    if isinstance(v, (int, float)):
        return float(v)
    # String cleanup: strip currency symbols, commas, whitespace, trailing periods
    s = str(v).replace("₹", "").replace("$", "").replace(",", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return 0.0

# ── Gemini report section parser ──────────────────────────────────────────────
_SECTION_HEADERS = [
    "CASE SUMMARY", "TRANSACTION TIMELINE", "ACCOUNTS INVOLVED",
    "FINANCIAL IMPACT", "RECOMMENDED ACTIONS", "RISK CLASSIFICATION",
]

def _parse_report(report_text: str) -> list:
    """
    Split a Gemini AML report into (title, body) tuples.
    Strips **, ***, ---, === and ## markdown artifacts.
    Falls back to [("", cleaned_text)] if no known headers found.
    """
    txt = re.sub(r"\*{1,3}", "", report_text)           # bold/italic markers
    txt = re.sub(r"^[-=]{3,}\s*$", "", txt, flags=re.M) # horizontal rules
    txt = re.sub(r"#+ ", "", txt)                        # ATX headings
    txt = re.sub(r"\n{3,}", "\n\n", txt.strip())         # collapse blank lines

    header_rx = "|".join(
        rf"(?:^\s*(?:\d+\.\s*)?{re.escape(h)}\s*:?)"
        for h in _SECTION_HEADERS
    )
    parts = re.split(rf"({header_rx})", txt, flags=re.M | re.I)

    sections: list = []
    if parts and parts[0].strip():
        sections.append(("", parts[0].strip()))
    i = 1
    while i < len(parts) - 1:
        hdr  = parts[i].strip().lstrip("0123456789. ").rstrip(":")
        body = parts[i + 1].strip() if (i + 1) < len(parts) else ""
        if hdr:
            sections.append((hdr.title(), body))
        i += 2

    return sections if sections else [("", txt)]


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# FORENSIC GRAPH HELPERS  (topology-driven, GNN-enhanced)
# ─────────────────────────────────────────────────────────────────────────────

import networkx as nx
import math


# ── Stage metadata ──────────────────────────────────────────────────────────
_STAGE_COLS = [
    (0, "ORIGIN",       "#3b82f6"),
    (1, "COMPROMISED",  "#eab308"),
    (2, "LAYERING",     "#f97316"),
    (3, "PRE-CASHOUT",  "#ef4444"),
    (4, "EXIT",         "#dc2626"),
]
_STAGE_ORDER = {s: i for i, (_, s, _) in enumerate(_STAGE_COLS)}

X_GAP_HIER = 7.0   # horizontal units between hop layers


# ── Layout ──────────────────────────────────────────────────────────────────

def _bfs_hop_depths(G: "nx.DiGraph", origins: set) -> dict:
    """
    Multi-source BFS from all origin nodes.
    Returns {node: min_hop_distance_from_any_origin}.
    Nodes unreachable from any origin get depth = -1.
    """
    depths = {o: 0 for o in origins if o in G}
    queue  = list(depths.keys())
    while queue:
        node = queue.pop(0)
        d    = depths[node]
        for succ in G.successors(node):
            if succ not in depths:
                depths[succ] = d + 1
                queue.append(succ)
    return depths


def _find_critical_path(G: "nx.DiGraph", origins: set, cashout_nodes: list) -> set:
    """
    Find nodes on any shortest path from any origin to any cashout node.
    Returns set of (source, destination) edge pairs on the critical path.
    """
    crit_edges: set = set()
    crit_nodes: set = set()
    for o in origins:
        for c in cashout_nodes:
            if o not in G or c not in G:
                continue
            try:
                path = nx.shortest_path(G, o, c)
                crit_nodes.update(path)
                for i in range(len(path) - 1):
                    crit_edges.add((path[i], path[i + 1]))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
    return crit_nodes, crit_edges


def _hierarchical_layout(
    G: "nx.DiGraph",
    ring_accounts: list,
    cashout_nodes: list,
    stage_lookup: dict = None,
    hybrid_scores: dict = None,
) -> dict:
    """
    Topology-driven hierarchical layout.

    Layout strategy:
      1. Identify origin nodes (ring nodes with no ring predecessors).
      2. BFS from all origins simultaneously to assign hop depth.
      3. X = hop_depth * X_GAP_HIER.
      4. Within each layer, sort by (out_degree DESC, hybrid_score DESC).
         Allocate Y spacing proportional to node degree weight.
      5. Critical-path nodes shifted slightly upward (+0.4 Y) for visual emphasis.
      6. Normal background nodes scattered to the right in a separate column.
      7. Cashout nodes forced to max(hop_depth) + 1 column.

    Returns dict {node: (x, y)}.
    """
    import random as _rnd
    _rnd.seed(7)

    if stage_lookup  is None: stage_lookup  = {}
    if hybrid_scores is None: hybrid_scores = {}

    ring_set    = set(ring_accounts)
    cashout_set = set(cashout_nodes)

    # ── 1. Identify origins ─────────────────────────────────────────────
    origin_set: set = set()
    for n in ring_set:
        if n in G and not any(p in ring_set for p in G.predecessors(n)):
            origin_set.add(n)
    if not origin_set and ring_set:
        origin_set = {max(ring_set, key=lambda n: G.out_degree(n) if n in G else 0)}

    # ── 2. BFS hop depths ───────────────────────────────────────────────
    hop = _bfs_hop_depths(G, origin_set)

    # ── 3. Critical path ────────────────────────────────────────────────
    crit_nodes, crit_edges_set = _find_critical_path(G, origin_set, cashout_nodes)

    # ── 4. Assign columns ───────────────────────────────────────────────
    max_ring_hop = max((hop.get(n, 0) for n in ring_set if n in hop), default=2)

    col_map: dict = {}
    for n in G.nodes():
        if n in cashout_set:
            col_map[n] = max_ring_hop + 1   # one column beyond deepest ring node
        elif n in ring_set:
            col_map[n] = hop.get(n, 2)      # BFS depth = column
        else:
            col_map[n] = max_ring_hop + 3   # background nodes far right

    # ── 5. Group by column ──────────────────────────────────────────────
    col_buckets: dict = {}
    for n, c in col_map.items():
        col_buckets.setdefault(c, []).append(n)

    # ── 6. Build positions ──────────────────────────────────────────────
    pos: dict = {}
    MIN_GAP   = 1.6   # minimum vertical gap (units)

    for col, nodes in sorted(col_buckets.items()):
        x = col * X_GAP_HIER

        # Sort within layer: out_degree DESC, hybrid_score DESC
        def _sort_key(n):
            deg  = G.out_degree(n) + G.in_degree(n) if n in G else 0
            hyb  = float(hybrid_scores.get(n, 0))
            return (-deg, -hyb)

        nodes_sorted = sorted(nodes, key=_sort_key)
        n_nodes      = len(nodes_sorted)

        # Degree-weight proportional Y allocation
        weights = []
        for n in nodes_sorted:
            deg = max(G.out_degree(n) + G.in_degree(n), 1) if n in G else 1
            weights.append(float(deg))
        total_w = sum(weights) or 1.0

        # Build cumulative Y positions (scaled so total span adapts to content)
        total_span = max(n_nodes * MIN_GAP, (n_nodes - 1) * MIN_GAP)
        ys: list = []
        cumulative = 0.0
        for w in weights:
            ys.append(cumulative)
            cumulative += (w / total_w) * total_span
        # Center around 0
        center = cumulative / 2.0
        ys = [y - center for y in ys]

        for node, y in zip(nodes_sorted, ys):
            # Critical-path nodes nudge up for visual clarity
            y_offset = 0.35 if node in crit_nodes and node not in cashout_set else 0.0
            jitter   = _rnd.uniform(-0.05, 0.05)
            pos[node] = (x, y + y_offset + jitter)

    # Safety: any node still unplaced (disconnected)
    max_col_x = max((c * X_GAP_HIER for c in col_buckets), default=0)
    for node in G.nodes():
        if node not in pos:
            pos[node] = (max_col_x + X_GAP_HIER + _rnd.uniform(-0.4, 0.4),
                         _rnd.uniform(-6, 6))
    return pos


def _spring_layout_raw(G: "nx.DiGraph", seed: int = 42) -> dict:
    """Wide spring layout for raw (pre-intelligence) network view."""
    if not G.nodes:
        return {}
    nodes = list(G.nodes())[:200]
    sub   = G.subgraph(nodes)
    pos   = nx.spring_layout(sub, seed=seed, k=2.8, iterations=80)
    return dict(pos)


# ── Bezier curve helper ──────────────────────────────────────────────────────

def _bezier_pts(x0, y0, x1, y1, n_pts=14, bow=0.6):
    """
    Approximate quadratic bezier from (x0,y0) to (x1,y1).
    Control point is offset perpendicular to the midpoint.
    Returns (xs, ys) lists with None separator for Plotly.
    """
    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2
    # Perpendicular offset (upward bow for horizontal edges)
    dx = x1 - x0; dy = y1 - y0
    length = math.sqrt(dx*dx + dy*dy) or 1.0
    # Perp direction (rotate 90 deg)
    px = -dy / length; py = dx / length
    cx = mx + px * bow
    cy = my + py * bow

    pts_x, pts_y = [], []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        bx = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
        by = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1
        pts_x.append(bx); pts_y.append(by)
    pts_x.append(None); pts_y.append(None)
    return pts_x, pts_y


# ── Main graph renderer ──────────────────────────────────────────────────────

def _build_forensic_graph(
    G: "nx.DiGraph",
    pos: dict,
    ring_accounts: list,
    cashout_nodes: list,
    dna_lookup: dict,
    stage_lookup: dict,
    hop_lookup: dict,
    inr_rate: float,
    focus_ring: bool = False,
    intelligence_mode: bool = True,
    frozen_nodes: list = None,
    gnn_scores: dict = None,
    hybrid_scores: dict = None,
) -> "go.Figure":
    """
    Demo-grade forensic directed Plotly graph.

    Intelligence Mode:
      - BFS topology-driven hierarchical layout
      - Stage column headers  + faint dotted separators
      - 3-tier edge rendering:
            bg        0.03 opacity, 0.3px
            adj       0.20 opacity, 0.7px
            critical  0.85 opacity, 2.2px + arrowheads + bezier curvature
      - Top-10 amount labels offset above edge midpoint
      - DNA-proportional node sizes (6→32px)
      - GNN glow ring on structurally anomalous nodes
      - Rich hover: Stage, DNA, GNN Score, Hybrid, In/Out-flow, Hops

    Raw Mode:
      - Spring layout, neutral blue palette, no DNA colors
      - Thin semi-transparent edges, degree-scaled nodes
      - Flow + degree tooltips
    """
    ring_set    = set(ring_accounts) if intelligence_mode else set()
    cashout_set = set(cashout_nodes) if intelligence_mode else set()
    frozen_set  = set(frozen_nodes   or [])
    gnn_sc      = gnn_scores     or {}
    hyb_sc      = hybrid_scores  or {}

    # ── Identify origin ─────────────────────────────────────────────────
    origin_node = None
    if ring_set and intelligence_mode:
        candidates = {
            n: float(dna_lookup.get(n, 0)) for n in ring_set
            if n in G and n in pos and not any(p in ring_set for p in G.predecessors(n))
        }
        if candidates:
            origin_node = max(candidates, key=candidates.get)
        elif ring_set:
            origin_node = max(
                (n for n in ring_set if n in pos),
                key=lambda n: float(dna_lookup.get(n, 0)),
                default=None,
            )

    # ── Critical path edges ─────────────────────────────────────────────
    origins_for_path = {origin_node} if origin_node else set()
    _, _crit_edge_set = _find_critical_path(G, origins_for_path, cashout_nodes)

    # ── Nodes to render ─────────────────────────────────────────────────
    if focus_ring and intelligence_mode:
        render_nodes = [n for n in pos if n in ring_set or n in cashout_set]
    else:
        render_nodes = list(pos.keys())[:200]
    render_set = set(render_nodes)

    # ── Edge classification ─────────────────────────────────────────────
    all_edges_raw = [
        (s, d, dat) for s, d, dat in G.edges(data=True)
        if s in pos and d in pos and s in render_set and d in render_set
    ]

    if intelligence_mode:
        critical_edges = [
            (s, d, dat) for s, d, dat in all_edges_raw
            if (s in ring_set or s in cashout_set) and (d in ring_set or d in cashout_set)
        ]
        crit_pair_set  = {(s, d) for s, d, _ in critical_edges}
        ring_adj_edges = [
            (s, d, dat) for s, d, dat in all_edges_raw
            if (s in ring_set or d in ring_set) and (s, d) not in crit_pair_set
        ]
        bg_edges = [
            (s, d, dat) for s, d, dat in all_edges_raw
            if s not in ring_set and d not in ring_set
        ]
    else:
        critical_edges, ring_adj_edges, bg_edges = [], [], all_edges_raw

    if len(bg_edges) > 60:
        import random as _r; _r.seed(42)
        bg_edges = _r.sample(bg_edges, 60)

    # ── Edge geometry helpers ───────────────────────────────────────────
    def _straight_xy(edges):
        ex, ey = [], []
        for s, d, _ in edges:
            x0, y0 = pos[s]; x1, y1 = pos[d]
            ex.extend([x0, x1, None]); ey.extend([y0, y1, None])
        return ex, ey

    def _arrow_pts(edges, t=0.70):
        ax, ay = [], []
        for s, d, _ in edges:
            x0, y0 = pos[s]; x1, y1 = pos[d]
            ax.append(x0 + t * (x1 - x0))
            ay.append(y0 + t * (y1 - y0))
        return ax, ay

    fig = go.Figure()
    annotations: list = []

    # ═══════════════════════════════════════════════════════════ INTELLIGENCE
    if intelligence_mode:
        # ── Compute canvas bounds from actual positions ───────────────────
        xs_all = [x for _, (x, _) in pos.items()] if pos else [0]
        ys_all = [y for _, (_, y) in pos.items()] if pos else [0]
        x_min  = min(xs_all) - X_GAP_HIER * 0.55
        x_max  = max(xs_all) + X_GAP_HIER * 0.55
        y_min  = min(ys_all) - 2.0
        y_max  = max(ys_all) + 2.0

        # Auto-scale height: base 520 + 70px per node in tallest layer
        col_counts: dict = {}
        for node in render_nodes:
            if node not in pos: continue
            cx = round(pos[node][0] / X_GAP_HIER)
            col_counts[cx] = col_counts.get(cx, 0) + 1
        max_nodes_in_col = max(col_counts.values(), default=1)
        fig_height = max(520, min(900, 280 + max_nodes_in_col * 72))

        # ── Stage column headers ──────────────────────────────────────────
        for col, label, clr in _STAGE_COLS:
            annotations.append(dict(
                x=col * X_GAP_HIER, y=y_max + 0.4,
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=9, color=clr, family="JetBrains Mono"),
                xanchor="center", yanchor="bottom",
            ))

        # ── Faint dotted vertical separators ─────────────────────────────
        for col, _, _ in _STAGE_COLS[1:]:
            sx = col * X_GAP_HIER - X_GAP_HIER / 2
            fig.add_shape(
                type="line",
                x0=sx, x1=sx, y0=y_min, y1=y_max,
                line=dict(color="rgba(99,102,241,0.09)", width=1, dash="dot"),
            )

        # ── L1: Background edges — 3% opacity ────────────────────────────
        bx, by = _straight_xy(bg_edges)
        if bx:
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode="lines",
                line=dict(width=0.3, color="rgba(99,102,241,0.03)"),
                hoverinfo="none", showlegend=False,
            ))

        # ── L2: Ring-adjacent edges — 20% opacity ────────────────────────
        if ring_adj_edges:
            rx, ry = _straight_xy(ring_adj_edges)
            fig.add_trace(go.Scatter(
                x=rx, y=ry, mode="lines",
                line=dict(width=0.7, color="rgba(249,115,22,0.20)"),
                hoverinfo="none", showlegend=False,
            ))

        # ── L3: Critical-path edges — bezier curve + 85% opacity ──────────
        if critical_edges:
            # Sort to separate on-path vs off-path within critical
            on_path  = [(s, d, dat) for s, d, dat in critical_edges
                        if (s, d) in _crit_edge_set]
            off_path = [(s, d, dat) for s, d, dat in critical_edges
                        if (s, d) not in _crit_edge_set]

            # Off-path critical (ring-to-ring, not on shortest path): lower opacity
            if off_path:
                ox, oy = _straight_xy(off_path)
                fig.add_trace(go.Scatter(
                    x=ox, y=oy, mode="lines",
                    line=dict(width=1.2, color="rgba(239,68,68,0.40)"),
                    hoverinfo="none", showlegend=False,
                ))

            # On-path: bezier curves for long hops, straight for 1-hop
            c_bx, c_by = [], []
            for s, d, dat in on_path:
                x0, y0 = pos[s]; x1, y1 = pos[d]
                hop_dist = abs(round(x0 / X_GAP_HIER) - round(x1 / X_GAP_HIER))
                if hop_dist > 1:
                    # Quadratic bezier with perpendicular bow
                    bow    = min(hop_dist * 0.5, 2.5)
                    pts_x, pts_y = _bezier_pts(x0, y0, x1, y1, n_pts=18, bow=bow)
                    c_bx.extend(pts_x); c_by.extend(pts_y)
                else:
                    c_bx.extend([x0, x1, None]); c_by.extend([y0, y1, None])

            if c_bx:
                fig.add_trace(go.Scatter(
                    x=c_bx, y=c_by, mode="lines",
                    line=dict(width=2.4, color="rgba(239,68,68,0.85)"),
                    hoverinfo="none", showlegend=False,
                ))

            # Arrowheads on critical path only
            axs, ays = _arrow_pts(critical_edges)
            if axs:
                fig.add_trace(go.Scatter(
                    x=axs, y=ays, mode="markers",
                    marker=dict(symbol="triangle-right", size=6,
                                color="#ef4444", opacity=0.85),
                    hoverinfo="none", showlegend=False,
                ))

        # ── Top-10 amount labels (offset above edge midpoint) ─────────────
        label_edges = sorted(critical_edges,
                             key=lambda e: float(e[2].get("weight", 0)),
                             reverse=True)[:10]
        for s, d, dat in label_edges:
            x0, y0 = pos[s]; x1, y1 = pos[d]
            mx = (x0 + x1) / 2.0
            my = (y0 + y1) / 2.0 + 0.36
            amt = float(dat.get("weight", 0))
            annotations.append(dict(
                x=mx, y=my,
                text=f"\u20b9{amt * inr_rate:,.0f}",
                showarrow=False,
                font=dict(size=7.5, color="#fb923c", family="JetBrains Mono"),
                bgcolor="rgba(5,8,16,0.82)",
                borderpad=2, bordercolor="rgba(249,115,22,0.15)",
                borderwidth=1, xanchor="center",
            ))

        # ── GNN glow rings (structural anomaly > 0.55) ────────────────────
        gnn_glow_x, gnn_glow_y, gnn_glow_sz, gnn_glow_col = [], [], [], []
        for node in render_nodes:
            gnn_v = float(gnn_sc.get(node, 0))
            if gnn_v > 0.55 and node in pos:
                x, y = pos[node]
                gnn_glow_x.append(x); gnn_glow_y.append(y)
                # Glow size proportional to GNN confidence
                gnn_glow_sz.append(round(24 + gnn_v * 24, 1))
                alpha = round(0.10 + gnn_v * 0.20, 2)
                gnn_glow_col.append(f"rgba(168,85,247,{alpha})")   # purple glow

        if gnn_glow_x:
            fig.add_trace(go.Scatter(
                x=gnn_glow_x, y=gnn_glow_y, mode="markers",
                marker=dict(size=gnn_glow_sz, color=gnn_glow_col,
                            line=dict(width=0)),
                hoverinfo="none", showlegend=False,
            ))

        # ── Nodes ─────────────────────────────────────────────────────────
        dna_vals = [float(dna_lookup.get(n, 0)) for n in render_nodes]
        max_dna  = max(max(dna_vals, default=1.0), 1.0)

        STAGE_COLORS = {
            "Exit Imminent": ("#dc2626", "#ff6b6b"),
            "Pre-Cashout":   ("#ef4444", "#fca5a5"),
            "Layering":      ("#f97316", "#fdba74"),
            "Compromised":   ("#eab308", "#fde047"),
            "Normal":        ("#475569", "#64748b"),
        }

        n_colors, n_sizes, n_borders, n_symbols, n_hover, nx_pts, ny_pts = \
            [], [], [], [], [], [], []

        for node in render_nodes:
            if node not in pos: continue
            x, y = pos[node]
            nx_pts.append(x); ny_pts.append(y)

            dna   = float(dna_lookup.get(node, 0))
            stage = stage_lookup.get(node, "Normal")
            hops  = hop_lookup.get(node, -1)
            out_w = sum(d.get("weight", 0) for _, _, d in G.out_edges(node, data=True))
            in_w  = sum(d.get("weight", 0) for _, _, d in G.in_edges(node, data=True))
            gnn_v = float(gnn_sc.get(node, 0))
            hyb_v = float(hyb_sc.get(node, dna / max_dna))

            norm_dna  = dna / max_dna
            base_size = max(6.0, min(32.0, 6 + norm_dna * 26))

            if node in frozen_set:
                color, border, symbol = "#374151", "#6b7280", "x"
                base_size = max(base_size, 14)
                role = "\U0001f9ca FROZEN"
            elif node in cashout_set:
                color, border, symbol = "#dc2626", "#fca5a5", "diamond"
                base_size = max(base_size, 24)
                role = "\U0001f6a8 CASHOUT SINK"
            elif node == origin_node:
                color, border, symbol = "#3b82f6", "#93c5fd", "star"
                base_size = max(base_size, 22)
                role = "\U0001f3af ORIGIN"
            elif node in ring_set:
                color, border = STAGE_COLORS.get(stage, ("#f97316", "#fdba74"))
                symbol = "circle"
                base_size = max(base_size, 12)
                role = f"\U0001f504 RING \u00b7 {stage}"
            else:
                alpha = max(0.07, norm_dna * 0.35)
                color  = f"rgba(59,130,246,{alpha:.2f})"
                border = "rgba(99,102,241,0.12)"
                base_size = min(base_size, 8)
                symbol = "circle"
                role = "\U0001f464 NORMAL"

            hover = (
                f"<b>{node}</b><br>"
                f"Role: {role}<br>"
                f"Stage: <b>{stage}</b><br>"
                f"DNA Score: <b>{dna:.1f}</b><br>"
                f"GNN Structural: <b>{gnn_v*100:.1f}%</b><br>"
                f"Hybrid Score: {hyb_v:.3f}<br>"
                f"Out-flow: \u20b9{out_w * inr_rate:,.0f}<br>"
                f"In-flow:  \u20b9{in_w  * inr_rate:,.0f}<br>"
                f"Hops to cashout: {hops if hops != -1 else 'N/A'}<br>"
                f"Degree: {G.in_degree(node)} in / {G.out_degree(node)} out"
            )
            n_colors.append(color)
            n_sizes.append(base_size)
            n_borders.append(border)
            n_symbols.append(symbol)
            n_hover.append(hover)

        fig.add_trace(go.Scatter(
            x=nx_pts, y=ny_pts, mode="markers",
            marker=dict(
                size=n_sizes, color=n_colors, symbol=n_symbols,
                line=dict(width=1.4, color=n_borders),
            ),
            text=n_hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

        frozen_note = f"  \u2502  \U0001f9ca {len(frozen_set)} frozen" if frozen_set else ""
        fig.update_layout(
            paper_bgcolor="#050810",
            plot_bgcolor="#050810",
            height=fig_height,
            title=dict(
                text=f"FORENSIC LAUNDERING TOPOLOGY \u2014 BFS Topology-Driven Layout{frozen_note}",
                font=dict(size=9, color="#475569", family="JetBrains Mono"),
                x=0.01, y=0.99,
            ),
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[x_min, x_max], fixedrange=False,
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[y_min - 0.3, y_max + 1.6], fixedrange=False,
            ),
            margin=dict(l=12, r=12, t=44, b=12),
            annotations=annotations,
            dragmode="pan",
            hoverlabel=dict(
                bgcolor="#0d1520", font_size=11,
                font_family="JetBrains Mono", font_color="#e2e8f0",
                align="left",
            ),
        )

    # ═══════════════════════════════════════════════════════════ RAW MODE
    else:
        bx, by = _straight_xy(bg_edges)
        if bx:
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode="lines",
                line=dict(width=0.6, color="rgba(71,85,105,0.22)"),
                hoverinfo="none", showlegend=False,
            ))

        n_colors, n_sizes, n_hover, nx_pts, ny_pts = [], [], [], [], []
        for node in render_nodes:
            if node not in pos: continue
            x, y = pos[node]
            nx_pts.append(x); ny_pts.append(y)
            out_w = sum(d.get("weight", 0) for _, _, d in G.out_edges(node, data=True))
            in_w  = sum(d.get("weight", 0) for _, _, d in G.in_edges(node, data=True))
            deg   = G.degree(node)
            sz    = max(5, min(18, 5 + deg * 0.5))
            if node in set(cashout_nodes):
                clr = "rgba(220,38,38,0.65)"; sz = max(sz, 14)
            elif node in set(ring_accounts):
                clr = "rgba(124,58,237,0.55)"; sz = max(sz, 10)
            else:
                clr = "rgba(30,58,95,0.60)"
            hov = (
                f"<b>{node}</b><br>"
                f"In: {G.in_degree(node)} / Out: {G.out_degree(node)}<br>"
                f"Out-flow: \u20b9{out_w * inr_rate:,.0f}<br>"
                f"In-flow:  \u20b9{in_w  * inr_rate:,.0f}"
            )
            n_colors.append(clr); n_sizes.append(sz); n_hover.append(hov)

        fig.add_trace(go.Scatter(
            x=nx_pts, y=ny_pts, mode="markers",
            marker=dict(size=n_sizes, color=n_colors,
                        line=dict(width=0.8, color="rgba(99,102,241,0.20)")),
            text=n_hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

        fig.update_layout(
            paper_bgcolor="#050810",
            plot_bgcolor="#050810",
            height=580,
            title=dict(
                text="PRE-INTELLIGENCE TRANSACTION NETWORK \u2014 Spring Layout",
                font=dict(size=9, color="#475569", family="JetBrains Mono"),
                x=0.01, y=0.99,
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=12, r=12, t=36, b=12),
            dragmode="pan",
            hoverlabel=dict(
                bgcolor="#0d1520", font_size=11,
                font_family="JetBrains Mono", font_color="#e2e8f0",
            ),
        )

    return fig


# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChronoTrace | FinCrime Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialise SQLite schema on startup
db.init_db()


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #050810;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }

.main .block-container { padding: 0.8rem 1.8rem 2rem; max-width: 100%; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #050810 100%);
    border-right: 1px solid #1a2744;
}
[data-testid="stSidebar"] label { color: #64748b !important; font-size: 0.78rem; }

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #0c1526 0%, #0a1020 100%);
    border: 1px solid #1a2f52;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    position: relative; overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(59,130,246,0.12); }
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--kpi-accent, linear-gradient(90deg,#3b82f6,#06b6d4));
}
.kpi-label { font-size: 0.65rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:#475569; margin-bottom:0.3rem; }
.kpi-value { font-size: 1.8rem; font-weight:700; font-family:'JetBrains Mono',monospace; line-height:1; }
.kpi-sub   { font-size: 0.68rem; color:#374151; margin-top:0.25rem; }

/* Section headers */
.sec-hdr {
    display:flex; align-items:center; gap:0.5rem;
    font-size:0.68rem; font-weight:600; letter-spacing:0.14em;
    text-transform:uppercase; color:#374151;
    border-bottom:1px solid #1a2744; padding-bottom:0.45rem; margin-bottom:0.9rem;
}
.sec-dot { width:5px; height:5px; border-radius:50%; background:#3b82f6; display:inline-block; }

/* Alert feed */
.alert-card {
    background:#08101d; border-left:3px solid var(--ac,#3b82f6);
    border-radius:0 8px 8px 0; padding:0.55rem 0.75rem; margin-bottom:0.45rem;
}
.alert-meta { font-size:0.6rem; color:#374151; font-family:'JetBrains Mono',monospace; margin-bottom:0.18rem; }
.alert-msg  { font-size:0.74rem; color:#94a3b8; line-height:1.4; }
.alert-scroll { max-height:380px; overflow-y:auto; }
.alert-scroll::-webkit-scrollbar { width:3px; }
.alert-scroll::-webkit-scrollbar-thumb { background:#1a2744; border-radius:2px; }

/* Countdown */
.cd-box {
    background:linear-gradient(135deg,#1a0505,#2d0a0a);
    border:1px solid #7f1d1d; border-radius:12px;
    padding:1rem 1.2rem; text-align:center;
}
.cd-timer {
    font-size:2.4rem; font-weight:700;
    font-family:'JetBrains Mono',monospace; color:#ef4444;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.5} }
.cd-blink { animation: blink 1s ease-in-out infinite; }

/* Stage steps */
.stage-step {
    display:flex; align-items:center; gap:0.55rem;
    padding:0.3rem 0.55rem; border-radius:6px;
    margin-bottom:0.28rem; font-size:0.75rem;
    border:1px solid transparent; transition:all 0.2s;
}

/* DNA bars */
.dna-bar-bg { background:#1a2744; border-radius:3px; height:5px; }
.dna-bar-fill { height:5px; border-radius:3px; transition:width .6s ease; }

/* AI Intelligence panel */
.ai-panel {
    background:linear-gradient(135deg,#080f1e,#0c1428);
    border:1px solid #1a3a6a; border-radius:14px; padding:1.2rem 1.4rem;
}
.ai-badge {
    display:inline-flex; align-items:center; gap:0.4rem;
    padding:0.2rem 0.7rem; border-radius:20px;
    font-size:0.68rem; font-weight:700; letter-spacing:0.08em;
}
.ai-field-label { font-size:0.65rem; color:#475569; text-transform:uppercase; letter-spacing:0.1em; }
.ai-field-value { font-size:0.82rem; color:#cbd5e1; line-height:1.5; }

/* Metric rows */
.mrow { display:flex; justify-content:space-between; align-items:center; padding:0.38rem 0; border-bottom:1px solid #1a2744; font-size:0.76rem; }
.mrow:last-child { border:none; }
.mk { color:#475569; }
.mv { font-family:'JetBrains Mono',monospace; font-weight:500; }

/* Intervention outcome */
.int-outcome {
    background:#08101d; border:1px solid #1a2744; border-radius:10px;
    padding:0.9rem; margin-top:0.5rem;
}

/* Table tweaks */
.stDataFrame { border-radius:8px; }

/* Buttons */
.stButton>button {
    border-radius:8px; font-weight:600; font-size:0.8rem;
    letter-spacing:0.03em; transition:all 0.2s; border:none;
}
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 4px 16px rgba(0,0,0,.5); }
hr { border-color:#1a2744; }

/* Phase 1 — Stage Timeline Strip */
.timeline-strip {
    display:flex; align-items:center;
    background:linear-gradient(90deg,#080f1e,#0a1525);
    border:1px solid #1a2744; border-radius:10px;
    padding:.55rem 1.2rem; gap:0; margin-bottom:1.1rem;
    overflow:hidden;
}
.tl-step {
    flex:1; display:flex; flex-direction:column; align-items:center;
    padding:.3rem .5rem; position:relative;
    font-size:.62rem; letter-spacing:.09em; text-transform:uppercase;
    color:#374151; font-weight:500; cursor:default;
    transition:all .3s;
}
.tl-step.active {
    color:#f1f5f9; font-weight:700;
}
.tl-dot {
    width:9px; height:9px; border-radius:50%;
    margin-bottom:.32rem; transition:all .3s;
    border:2px solid #1a2744;
}
.tl-step.active .tl-dot {
    animation: pulse-glow 1.4s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%,100% { box-shadow:0 0 0 0 rgba(255,255,255,0.08); transform:scale(1); }
    50%      { box-shadow:0 0 0 6px rgba(255,255,255,0); transform:scale(1.15); }
}
.tl-connector {
    flex:0 0 1.5rem; height:1px; background:#1a2744; margin-bottom:.56rem;
}
.tl-step.active .tl-connector { background:currentColor; }

/* Phase 3 — Risk Drivers card */
.risk-driver-bar {
    height:4px; border-radius:2px;
    background:linear-gradient(90deg,var(--dclr),transparent);
    transition:width .6s ease;
}

/* Phase 6 — Quantum Scanner */
.pqc-badge {
    display:inline-block; padding:.22rem .75rem; border-radius:20px;
    font-size:.68rem; font-weight:700; letter-spacing:.07em;
    border:1px solid currentColor; white-space:nowrap;
}
.qrow {
    display:flex; justify-content:space-between; align-items:center;
    padding:.38rem 0; border-bottom:1px solid #1a2744; font-size:.74rem;
}
.qrow:last-child { border:none; }
.qk { color:#475569; font-size:.65rem; text-transform:uppercase; letter-spacing:.08em; }
.qv { font-family:'JetBrains Mono',monospace; color:#e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "sim_result":            None,
        "analysis":              None,
        "pred_df":               None,
        "alerts":                [],
        "ring_summary":          {},
        "intervention":          None,
        "intervention_out":      None,        # vel_col alert_engine output
        "intervention_graph_out": None,       # tab3 graph intervention output
        "last_mode":             None,
        "predicted_exit_ts":     None,
        "ai_intelligence":       None,
        "ai_report":             None,
        "ai_report_requested":   False,
        "ai_intel_requested":    False,
        "ai_metrics_json":       None,
        "anchored":              False,
        "anchor_result":         None,
        "current_stage":         "Normal",
        "pdf_report":            None,
        "analysis_complete":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(mode: str, n_rings: int = 1):
    """
    Step 1: Data Generation / Loading.
    Populates sim_result and clears intelligence results to enter 'Raw' mode.
    """
    with st.spinner("🔄 Loading transaction data…"):
        if mode == "dataset":
            result = dataset_loader.build_sim_result_from_dataset()
            db.init_db()
            db.clear_all()
            db.save_transactions(result["transactions"])
        else:
            result = simulator.run_and_persist(mode=mode, n_rings=n_rings)

        st.session_state.sim_result = result
        st.session_state.last_mode = mode
        st.session_state.analysis_complete = False
        st.session_state.analysis = None
        st.session_state.pred_df = None
        st.session_state.alerts = []
        st.session_state.anchored = False


def compute_intelligence():
    """
    Step 2: Intelligence Analysis.
    Computes DNA, stage, and alerts. Sets analysis_complete = True.
    """
    if not st.session_state.sim_result:
        return

    result = st.session_state.sim_result
    mode = st.session_state.last_mode

    with st.spinner("🧠 Activating Graph Intelligence…"):
        analysis = graph_engine.analyse_from_db(result)
        
        # Phase 3: GNN Structural Validation
        gnn_dna_df = gnn_layer.run_gnn_validation(analysis["dna_df"], analysis["graph"])
        analysis["dna_df"] = gnn_dna_df
        
        pred_df  = stage_predictor.predict(analysis["dna_df"])

        simulator.persist_predictions(result, pred_df, analysis["dna_df"])

        ring_summary = stage_predictor.predict_ring_summary(
            pred_df, result["ring_accounts"])

        all_alerts = alert_engine.generate_all_alerts(result, analysis, pred_df)

        ttc = ring_summary.get("min_time_to_cashout", -1)
        predicted_exit_ts = (
            datetime.now(timezone.utc) + timedelta(minutes=ttc) if ttc >= 0 else None
        )

        metrics_json = gemini_layer.build_metrics_json(pred_df, ring_summary)

        # Store in session state
        st.session_state.analysis = analysis
        st.session_state.pred_df = pred_df
        st.session_state.alerts = all_alerts
        st.session_state.ring_summary = ring_summary
        st.session_state.predicted_exit_ts = predicted_exit_ts
        st.session_state.ai_metrics_json = metrics_json
        st.session_state.current_stage = ring_summary.get("dominant_label", "Normal")
        st.session_state.analysis_complete = True
    st.session_state.ai_intelligence     = None
    st.session_state.ai_intel_requested  = False
    st.session_state.ai_report           = None
    st.session_state.ai_report_requested = False
    st.session_state.intervention        = None
    st.session_state.intervention_out    = None
    st.session_state.anchored            = False
    st.session_state.anchor_result       = None
    st.session_state.last_mode           = mode
    # Single source of truth for stage — set once here, never overwritten by Gemini
    st.session_state.current_stage       = ring_summary.get("dominant_label", "Normal")


# ─────────────────────────────────────────────────────────────────────────────
# ── UNPACK SESSION DATA ─────────────────────────────────────────────────────────
result       = st.session_state.sim_result
analysis     = st.session_state.analysis
pred_df      = st.session_state.pred_df if st.session_state.pred_df is not None else pd.DataFrame()
all_alerts   = st.session_state.alerts
ring_summary = st.session_state.ring_summary or {}
is_analyzed = st.session_state.analysis_complete
intelligence_mode = is_analyzed

# Safe defaults if intelligence not yet run
if is_analyzed and analysis:
    summary = analysis["summary"]
    G       = analysis["graph"]
elif result:
    import dna_engine
    G = dna_engine.build_graph(result["transactions"])
    summary = {
        "max_dna_score": 0.0, "n_critical": 0, "n_clusters": 0,
        "total_nodes": G.number_of_nodes(), "total_edges": G.number_of_edges(),
        "avg_dna_score": 0.0
    }
else:
    G = None
    summary = {}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'>🔍</div>
        <div style='font-size:1.05rem;font-weight:800;color:#f1f5f9;letter-spacing:.06em;'>ChronoTrace</div>
        <div style='font-size:.65rem;color:#374151;letter-spacing:.1em;margin-top:.2rem;'>
            FINCRIME INTELLIGENCE PLATFORM
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Dataset-backed run (primary for demo / Streamlit Cloud) ────────
    st.caption("🗂️ Dataset-Backed Analysis")
    if st.button("📊 Load transactions.csv", use_container_width=True, key="btn_dataset"):
        # Invalidate dataset cache so the CSV is re-read and graph rebuilt
        dataset_loader.invalidate_cache()
        run_pipeline(mode="dataset")
        st.rerun()

    st.markdown("<hr style='border-color:#1a2744;margin:.5rem 0'>", unsafe_allow_html=True)

    # ── Synthetic simulation (existing) ────────────────────────────────
    st.caption("🎲 Synthetic Simulation")
    mode_label = st.selectbox(
        "Mode",
        ["Attack Simulation", "Normal (Baseline)"],
        help="Attack mode injects mule ring patterns.",
        label_visibility="collapsed",
    )
    mode = "attack" if "Attack" in mode_label else "normal"

    n_rings = 1
    if mode == "attack":
        n_rings = st.slider("Mule Rings", 1, 10, 3,
                            help="Number of simultaneous mule rings to inject (1–10)")

    if st.button("▶️  Run Synthetic Simulation", use_container_width=True):
        run_pipeline(mode, n_rings)
        st.rerun()

    if st.session_state.sim_result:
        st.markdown("---")
        # Export JSON
        if st.button("📥  Export Report", use_container_width=True):
            report = {
                "generated_at":  datetime.now().isoformat(),
                "mode":          st.session_state.last_mode,
                "summary":       analysis["summary"] if analysis else {},
                "ring_summary":  st.session_state.ring_summary,
                "ai_analysis":   st.session_state.ai_intelligence,
                "top_risks":     analysis["top_risks"].to_dict("records") if analysis and not analysis["top_risks"].empty else [],
                "alerts": [
                    {k: (v.isoformat() if isinstance(v, datetime) else v)
                     for k, v in a.items()}
                    for a in st.session_state.alerts
                ],
            }
            st.download_button(
                "💾 Download JSON",
                data=json.dumps(report, indent=2, default=str),
                file_name=f"chronotrace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

            # Phase 9 — PDF export (HTML print)
            _stage_pdf  = st.session_state.get("current_stage", "N/A")
            _dna_pdf    = report.get("summary", {}).get("max_dna_score", 0)
            _prob_pdf   = report.get("ring_summary", {}).get("max_cashout_probability", 0)
            _hash_pdf   = (st.session_state.anchor_result or {}).get("alert_hash", "Not Anchored")
            _ai_pdf     = (st.session_state.ai_intelligence or {})
            _pdf_html   = f"""
<!DOCTYPE html><html><head>
<meta charset='utf-8'>
<title>ChronoTrace Executive Report</title>
<style>
  body{{font-family:Arial,sans-serif;color:#1e293b;margin:40px;background:#fff;}}
  h1{{color:#1d4ed8;font-size:22px;margin-bottom:4px;}}
  .sub{{color:#64748b;font-size:12px;margin-bottom:24px;}}
  table{{width:100%;border-collapse:collapse;margin-bottom:20px;}}
  th{{background:#f1f5f9;padding:8px;text-align:left;font-size:11px;
      text-transform:uppercase;letter-spacing:.07em;color:#475569;}}
  td{{padding:8px;border-bottom:1px solid #e2e8f0;font-size:12px;}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:20px;
          font-size:11px;font-weight:700;border:1px solid;}}
  .footer{{margin-top:30px;font-size:10px;color:#94a3b8;}}
</style>
</head><body>
<h1>&#55356;&#57076; ChronoTrace — Executive Investigation Report</h1>
<div class='sub'>Generated: {datetime.now().strftime('%d %b %Y %H:%M IST')} &nbsp;|&nbsp; Mode: {st.session_state.get('last_mode','N/A').upper()}</div>
<table><tr><th>Field</th><th>Value</th></tr>
<tr><td>Laundering Stage</td><td><b>{_stage_pdf}</b></td></tr>
<tr><td>Max DNA Score</td><td>{_dna_pdf:.1f}</td></tr>
<tr><td>Cashout Probability</td><td>{_prob_pdf:.0f}%</td></tr>
<tr><td>Blockchain Anchor</td><td style='font-family:monospace;font-size:10px;'>{_hash_pdf[:48]}...</td></tr>
<tr><td>Risk Reasoning</td><td>{_ai_pdf.get('risk_reasoning','N/A')}</td></tr>
<tr><td>Recommended Action</td><td>{_ai_pdf.get('recommended_action','N/A')}</td></tr>
</table>
<div class='footer'>ChronoTrace v2.0 &middot; Powered by Gemini 2.5 Flash &middot; Ethereum Sepolia &middot; NetworkX</div>
</body></html>"""
            import base64 as _b64
            _pdf_b64 = _b64.b64encode(_pdf_html.encode()).decode()
            st.markdown(f"""
            <a href="data:text/html;base64,{_pdf_b64}"
               download="chronotrace_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
               style='display:block;width:100%;text-align:center;
                      background:#1d4ed8;color:#fff;padding:.45rem;
                      border-radius:8px;font-size:.78rem;font-weight:600;
                      text-decoration:none;margin-top:.4rem;'>
                📊 Download Executive Report (HTML/PDF)
            </a>
            """, unsafe_allow_html=True)

    # API key status
    st.markdown("---")
    key_set = bool(gemini_layer._get_api_key())
    st.markdown(f"""
    <div style='font-size:.68rem;color:{"#22c55e" if key_set else "#ef4444"};
                text-align:center;font-weight:600;'>
        {'🟢 Gemini API Active' if key_set else '🔴 Gemini API Not Configured'}
    </div>""", unsafe_allow_html=True)
    if not key_set:
        st.markdown("""
        <div style='font-size:.62rem;color:#374151;text-align:center;margin-top:.3rem;line-height:1.5;'>
            Set GEMINI_API_KEY in<br>environment or secrets.toml
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.62rem;color:#1e293b;text-align:center;line-height:1.7;'>
        ChronoTrace v2.0<br>
        Powered by Gemini · NetworkX<br>
        © 2025 ChronoTrace Labs
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER — Executive Summary Bar  (Phase 7)
# ─────────────────────────────────────────────────────────────────────────────

_now_str = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
_raw_mode = st.session_state.get("last_mode")
_mode_lbl = _raw_mode.upper() if isinstance(_raw_mode, str) and _raw_mode.strip() else "STANDBY"
_mode_color = "#ef4444" if _mode_lbl == "ATTACK" else "#f97316" if _mode_lbl == "DATASET" else "#3b82f6"
st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;
            border-bottom:1px solid #1a2744;padding-bottom:.9rem;margin-bottom:1.1rem;'>
    <div style='display:flex;align-items:center;gap:.9rem;'>
        <div style='font-size:1.5rem;font-weight:800;letter-spacing:.05em;color:#f1f5f9;'>🔍 ChronoTrace</div>
        <div style='font-size:.62rem;color:#475569;letter-spacing:.08em;margin-top:.2rem;
                    border-left:1px solid #1a2744;padding-left:.9rem;line-height:1.8;'>
            Predictive AML Intelligence Platform<br>
            <span style='color:{_mode_color};font-weight:700;letter-spacing:.1em;'>{_mode_lbl}</span>
        </div>
    </div>
    <div style='display:flex;align-items:center;gap:1.2rem;'>
        <div style='text-align:right;'>
            <div style='font-size:.58rem;color:#1e293b;font-family:monospace;letter-spacing:.06em;'>LIVE THREAT INTELLIGENCE</div>
            <div style='font-size:.65rem;color:#1e293b;font-family:monospace;'>{_now_str}</div>
        </div>
        <div style='background:rgba(59,130,246,.08);border:1px solid #1d4ed8;
                    border-radius:6px;padding:.25rem .7rem;
                    font-size:.68rem;color:#3b82f6;font-weight:700;font-family:monospace;
                    animation:blink 2s ease-in-out infinite;'>● SYSTEM ACTIVE</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.sim_result is None:
    st.markdown("""
    <div style='text-align:center;padding:5rem 2rem;'>
        <div style='font-size:4rem;margin-bottom:1rem;'>🛡️</div>
        <div style='font-size:1.3rem;font-weight:700;color:#e2e8f0;margin-bottom:.5rem;'>
            Intelligence Engine Ready
        </div>
        <div style='font-size:.85rem;color:#374151;max-width:500px;margin:0 auto;line-height:1.9;'>
            Select simulation mode, set the number of mule rings, then click
            <strong style='color:#3b82f6;'>▶ Run Simulation</strong> to begin<br>
            real-time mule ring detection + AI-powered threat intelligence.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# UNPACK SESSION DATA
# ─────────────────────────────────────────────────────────────────────────────

# Redundant block removed (Unpacked at top)

transactions  = result["transactions"]
ring_accounts = result["ring_accounts"]
cashout_nodes = result.get("cashout_nodes", [])
is_attack     = result.get("mode") == "attack"
ai_intel      = st.session_state.ai_intelligence or {}




# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5, k6 = st.columns(6)

_KPI_COLORS = {
    "red":    "#ef4444",
    "orange": "#f97316",
    "purple": "#a855f7",
    "blue":   "#3b82f6",
    "cyan":   "#06b6d4",
    "slate":  "#64748b",
}

def _kpi(col, label, value, sub, color_key, gradient):
    c = _KPI_COLORS[color_key]
    col.markdown(f"""
    <div class="kpi-card" style="--kpi-accent:{gradient}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{c}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# Dynamic countdown for KPI
predicted_exit_ts = st.session_state.predicted_exit_ts
if predicted_exit_ts:
    remaining = (predicted_exit_ts - datetime.now(timezone.utc)).total_seconds() / 60
    ttc_display = f"{max(remaining, 0):.0f}m" if remaining > 0 else "DONE"
else:
    ttc_display, remaining = "N/A", -1

cashout_prob = ring_summary.get("max_cashout_probability", 0)

_kpi(k1, "Max DNA Score",  f"{summary['max_dna_score']:.0f}",
     "/ 100 threat level", "red",    "linear-gradient(90deg,#ef4444,#f97316)")
_kpi(k2, "Critical Nodes", str(summary["n_critical"]),
     "critical risk nodes", "orange", "linear-gradient(90deg,#f97316,#eab308)")
_kpi(k3, "Ring Clusters",  str(summary["n_clusters"]),
     "suspicious clusters", "purple", "linear-gradient(90deg,#a855f7,#6366f1)")
_kpi(k4, "Time to Cashout", ttc_display,
     "from now (live)",     "red",    "linear-gradient(90deg,#ef4444,#dc2626)")
_kpi(k5, "Cashout Prob",   f"{cashout_prob:.0f}%",
     ring_summary.get("dominant_label","Normal"), "cyan",
     "linear-gradient(90deg,#3b82f6,#06b6d4)")
_kpi(k6, "Total Nodes",    str(summary["total_nodes"]),
     f"{summary['total_edges']} edges", "slate",
     "linear-gradient(90deg,#475569,#334155)")

st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — INCIDENT TIMELINE STRIP + RISK DRIVERS
# ─────────────────────────────────────────────────────────────────────────────

_tl_col, _rd_col = st.columns([2.6, 1.4], gap="medium")

with _tl_col:
    _cur_stage = st.session_state.get("current_stage", "Normal")
    _stages_tl = [
        ("Compromised",   "#eab308"),
        ("Layering",      "#f97316"),
        ("Pre-Cashout",   "#ef4444"),
        ("Exit Imminent", "#dc2626"),
    ]
    _stage_order = ["Normal", "Compromised", "Layering", "Pre-Cashout", "Exit Imminent"]
    _cur_idx = _stage_order.index(_cur_stage) if _cur_stage in _stage_order else 0

    html_tl = "<div class='timeline-strip'>"
    for i, (sname, scolor) in enumerate(_stages_tl):
        s_idx = _stage_order.index(sname)
        is_active  = _cur_stage == sname
        is_passed  = _cur_idx > s_idx
        dot_bg     = scolor if (is_active or is_passed) else "#1a2744"
        dot_shadow = f"box-shadow:0 0 10px {scolor};" if is_active else ""
        cls        = "tl-step active" if is_active else ("tl-step" if not is_passed else "tl-step")
        tc         = scolor if is_active else ("#64748b" if is_passed else "#374151")
        label_wt   = "700" if is_active else "400"
        if i > 0:
            conn_bg = scolor if is_passed else "#1a2744"
            html_tl += f"<div class='tl-connector' style='background:{conn_bg};'></div>"
        html_tl += f"""
        <div class='{cls}' style='color:{tc};font-weight:{label_wt};'>
            <div class='tl-dot' style='background:{dot_bg};border-color:{scolor};{dot_shadow}'></div>
            {sname}
        </div>"""
    html_tl += "</div>"
    st.markdown(html_tl, unsafe_allow_html=True)

with _rd_col:
    # Phase 3 — Risk Drivers card
    _top = None
    if intelligence_mode and analysis and not analysis["top_risks"].empty:
        _top = analysis["top_risks"].iloc[0]
    
    if _top is not None:
        _factors = [
            ("Fan-out",       _top.get("fan_out_ratio",  0), "#ef4444"),
            ("Hop Proximity", _top.get("hop_proximity",  0), "#f97316"),
            ("Velocity",      _top.get("velocity_score", 0), "#eab308"),
            ("Burst",         _top.get("burst_score",    0), "#8b5cf6"),
            ("Amt Anomaly",   _top.get("amount_anomaly", 0), "#06b6d4"),
        ]
        _factors_sorted = sorted(_factors, key=lambda x: x[1], reverse=True)[:3]

        # Auto-generate narrative
        top_f  = _factors_sorted[0][0] if _factors_sorted else ""
        top_f2 = _factors_sorted[1][0] if len(_factors_sorted) > 1 else ""
        hop_p  = float(_top.get("hop_proximity", 0))
        fan_o  = float(_top.get("fan_out_ratio", 0))
        _narrative = (
            f"{'High proximity to cashout' if hop_p > 0.5 else 'Elevated ' + top_f} "
            f"{'and abnormal fan-out behaviour' if fan_o > 0.6 else 'and ' + top_f2 + ' signals'} "
            f"increase exit probability."
        )

        # Build individual bar divs as plain strings (no outer f-string nesting)
        bar_divs = []
        for fname, fval, fclr in _factors_sorted:
            pct = min(float(fval), 1.0) * 100
            bar_divs.append(
                "<div style='margin-bottom:.5rem;'>"
                "<div style='display:flex;justify-content:space-between;"
                "font-size:.65rem;margin-bottom:.18rem;color:#64748b;'>"
                f"<span>{fname}</span>"
                f"<span style='color:#e2e8f0;font-family:monospace;'>{fval:.3f}</span>"
                "</div>"
                "<div style='background:#1a2744;border-radius:2px;height:4px;'>"
                f"<div style='width:{int(pct)}%;height:4px;border-radius:2px;"
                f"background:linear-gradient(90deg,{fclr},transparent);'>"
                "</div></div></div>"
            )
        bars_joined = "".join(bar_divs)

        card_html = (
            "<div style='background:#080f1e;border:1px solid #1a2744;"
            "border-radius:10px;padding:.7rem .9rem;'>"
            "<div style='font-size:.58rem;color:#475569;text-transform:uppercase;"
            "letter-spacing:.1em;margin-bottom:.65rem;display:flex;align-items:center;"
            "gap:.4rem;border-bottom:1px solid #1a2744;padding-bottom:.45rem;'>"
            "<span style='width:5px;height:5px;border-radius:50%;"
            "background:#3b82f6;display:inline-block;'></span>"
            "RISK DRIVERS</div>"
            + bars_joined +
            "<div style='font-size:.63rem;color:#475569;margin-top:.5rem;"
            "line-height:1.6;font-style:italic;border-top:1px solid #1a2744;"
            f"padding-top:.4rem;'>{_narrative}</div>"
            "</div>"
        )

        with st.container():
            st.markdown(card_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK GRAPH + ALERT FEED
# ─────────────────────────────────────────────────────────────────────────────

graph_col, alert_col = st.columns([3, 1.15], gap="medium")

with graph_col:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>FORENSIC TRANSACTION GRAPH — HIERARCHICAL DIRECTED</div>',
                unsafe_allow_html=True)

    # Pre-compute lookups once
    # analysis = st.session_state.analysis # Already unpacked
    # pred_df = st.session_state.pred_df # Already unpacked
    # intelligence_mode = st.session_state.analysis_complete # Already unpacked

    if intelligence_mode and analysis:
        dna_df_graph = analysis["dna_df"]
        dna_lookup   = dna_df_graph.set_index("node")["dna_score"].to_dict()
        hop_lookup   = dna_df_graph.set_index("node")["hops_to_cashout"].to_dict() \
                       if "hops_to_cashout" in dna_df_graph.columns else {}
        stage_lookup_g = {}
        if pred_df is not None and not pred_df.empty and "node" in pred_df.columns and "stage_label" in pred_df.columns:
            stage_lookup_g = pred_df.set_index("node")["stage_label"].to_dict()
        # GNN + hybrid score lookups for layout sorting and visual glow
        gnn_scores_g    = gnn_layer.get_gnn_scores_dict(dna_df_graph)
        hybrid_scores_g = gnn_layer.get_hybrid_scores_dict(dna_df_graph)
    else:
        dna_lookup      = {}
        hop_lookup      = {}
        stage_lookup_g  = {}
        gnn_scores_g    = {}
        hybrid_scores_g = {}

    # Phase 6 + 3: cap and warn
    MAX_NODES = 200
    GRAPH_TOO_LARGE = G.number_of_nodes() > MAX_NODES if G else False
    if GRAPH_TOO_LARGE:
        st.info(
            f"⚠️ Graph limited to top {MAX_NODES} accounts for performance clarity. "
            f"Full dataset has {G.number_of_nodes()} nodes.",
            icon="📊",
        )
    elif st.session_state.get("_ctl_dataset_pruned"):
        st.info(
            f"⚠️ Dataset pruned to top {MAX_NODES} accounts by transaction volume "
            "(fraud accounts always preserved).",
            icon="📊",
        )

    # Compute hierarchical layout (cached by graph fingerprint + intelligence state)
    if G:
        # Include intelligence_mode in key so layout rebuilds when GNN activates
        layout_key = (
            f"_hier_layout_bfs_{G.number_of_nodes()}_"
            f"{len(ring_accounts)}_{len(cashout_nodes)}_{int(intelligence_mode)}"
        )
        if layout_key not in st.session_state:
            st.session_state[layout_key] = _hierarchical_layout(
                G, ring_accounts, cashout_nodes,
                stage_lookup  = stage_lookup_g,
                hybrid_scores = hybrid_scores_g,
            )
        hier_pos = st.session_state[layout_key]

        # Spring layout for raw view (cached separately, no stage info needed)
        raw_layout_key = f"_spring_layout_{G.number_of_nodes()}"
        if raw_layout_key not in st.session_state:
            st.session_state[raw_layout_key] = _spring_layout_raw(G)
        spring_pos = st.session_state[raw_layout_key]
    else:
        hier_pos   = {}
        spring_pos = {}

    # Phase 5: focus toggle
    focus_opts = ["\U0001f50d Focus: Suspicious Ring", "\U0001f310 Show Full Network"]
    focus_default = 0 if (ring_accounts or GRAPH_TOO_LARGE) else 1
    focus_sel = st.radio(
        "Graph view",
        focus_opts,
        index=focus_default,
        horizontal=True,
        label_visibility="collapsed",
        key="graph_focus_toggle",
    )
    focus_ring_mode = "Suspicious" in focus_sel

    # Phase 2 & 4: Run button ABOVE tabs — always visible regardless of active tab
    if not intelligence_mode:
        _run_c1, _run_c2 = st.columns([3, 1])
        with _run_c1:
            st.warning(
                "⚠️ Graph intelligence analysis required to unlock forensic mapping.",
                icon="🔒",
            )
        with _run_c2:
            if st.button(
                "🔥 RUN GRAPH INTELLIGENCE ANALYSIS",
                use_container_width=True,
                type="primary",
                key="btn_run_intel_main",
            ):
                compute_intelligence()
                st.rerun()
    else:
        st.success(
            "🧠 Graph Intelligence Activated — DNA scores, stage labels, and risk overlays are live.",
            icon="✅",
        )

    tab1, tab2, tab3 = st.tabs(["🌐 Raw Network", "🧠 Graph Intelligence", "🛡️ Tiered Intervention"])

    with tab1:
        st.markdown("""<div style='font-size:.7rem;color:#475569;margin-bottom:.5rem;'>
        📊 <b>Pre-Intelligence Raw Network</b> — Spring layout, no DNA overlay.
        Nodes sized by degree. Run Graph Intelligence to unlock forensic analysis.
        </div>""", unsafe_allow_html=True)
        if G and spring_pos:
            fig_raw = _build_forensic_graph(
                G            = G,
                pos          = spring_pos,
                ring_accounts= ring_accounts,
                cashout_nodes= cashout_nodes,
                dna_lookup   = {},
                stage_lookup = {},
                hop_lookup   = {},
                inr_rate     = INR_RATE,
                intelligence_mode = False,
            )
            st.plotly_chart(fig_raw, use_container_width=True,
                            config={"displayModeBar": False}, key="fig_raw")
        else:
            st.info("Load a dataset or run a simulation to view the raw transaction network.")

    with tab2:
        if not intelligence_mode:
            st.info(
                "🔥 Click **RUN GRAPH INTELLIGENCE ANALYSIS** above to unlock the forensic graph overlay.",
                icon="🧠",
            )
        else:
            if G:
                # Pass frozen nodes so the graph renders them in grey
                _frozen_in_graph = (
                    st.session_state.get("intervention_graph_out", {}) or {}
                ).get("stats", {}).get("frozen_nodes", [])
                fig_intel = _build_forensic_graph(
                    G            = G,
                    pos          = hier_pos,
                    ring_accounts= ring_accounts,
                    cashout_nodes= cashout_nodes,
                    dna_lookup   = dna_lookup,
                    stage_lookup = stage_lookup_g,
                    hop_lookup   = hop_lookup,
                    inr_rate     = INR_RATE,
                    focus_ring   = focus_ring_mode,
                    intelligence_mode = True,
                    frozen_nodes = _frozen_in_graph,
                    gnn_scores   = gnn_scores_g,
                    hybrid_scores= hybrid_scores_g,
                )
                st.plotly_chart(fig_intel, use_container_width=True,
                                config={"displayModeBar": False}, key="fig_intel")
            else:
                st.info("No graph data available for intelligence display.")

    with tab3:
        if not intelligence_mode:
            st.warning("⚠️ Intelligence analysis must be run before applying interventions.", icon="🔒")
        elif not ring_accounts:
            st.info("🛡️ No suspicious ring detected in this dataset.\n\nFreeze controls are disabled.", icon="✅")
        else:
            # -- Buttons
            _int_col1, _int_col2 = st.columns([2, 1])
            _already_frozen = bool(st.session_state.get("intervention_graph_out"))
            with _int_col1:
                st.caption("🛡️ ACTIVE MITIGATION CONTROLS")
                _btn_ring = st.button(
                    "🔴 FREEZE SUSPICIOUS RING",
                    use_container_width=True,
                    help="Tiered freeze: HIGH risk nodes frozen, MEDIUM flagged for KYC.",
                    disabled=_already_frozen,
                )
                _btn_origin = st.button(
                    "🟠 ISOLATE ORIGIN ACCOUNT",
                    use_container_width=True,
                    help="Apply freeze only to origin node.",
                    disabled=_already_frozen,
                )
                if _already_frozen:
                    if st.button("🔄 Reset Intervention", use_container_width=True):
                        st.session_state.intervention_graph_out = None
                        st.rerun()
            with _int_col2:
                st.markdown("""<div style='background:#080f1e;border:1px solid #1a2744;
                border-radius:8px;padding:.7rem;font-size:.65rem;color:#64748b;'>
                <b style='color:#e2e8f0;'>Tier Logic</b><br><br>
                🧊 <b style='color:#ef4444;'>HIGH</b> — Exit Imminent / Pre-Cashout<br>&nbsp;&nbsp;Edges removed<br><br>
                🟡 <b style='color:#eab308;'>MEDIUM</b> — Layering / Compromised<br>&nbsp;&nbsp;KYC Flagged<br><br>
                ✅ <b style='color:#22c55e;'>LOW</b> — Normal<br>&nbsp;&nbsp;Approved
                </div>""", unsafe_allow_html=True)

            # -- Apply intervention logic
            if _btn_ring or _btn_origin:
                targets = ring_accounts if _btn_ring else ring_accounts[:1]

                # Fresh copy of graph so we don't mutate the cached one
                work_G = dna_engine.build_graph(transactions)

                # Use actual edge-weight-based loss calculation
                out = intervention_engine.apply_intervention(
                    work_G,
                    targets,
                    dna_lookup   = dna_lookup,
                    stage_lookup = stage_lookup_g,
                    cashout_nodes= cashout_nodes,
                )

                # Recompute post-intervention analysis
                try:
                    int_analysis = dna_engine.analyse(
                        {"transactions": transactions, "cashout_nodes": cashout_nodes}
                    )
                except Exception:
                    int_analysis = analysis
                int_analysis["graph"] = work_G

                # Prefer edge-weight loss; fallback to heuristic if 0
                _edge_loss = out.get("loss_avoided_usd", 0.0)
                _heuri_loss = intervention_engine.calculate_loss_avoided(
                    pred_df, out.get("frozen_nodes", [])
                )
                _final_loss = _edge_loss if _edge_loss > 0 else _heuri_loss

                st.session_state.intervention_graph_out = {
                    "analysis":    int_analysis,
                    "stats":       out,
                    "loss_avoided": _final_loss,
                }
                _n_frozen = len(out.get("frozen_nodes", []))
                _n_kyc    = len(out.get("kyc_nodes",    []))
                _n_appr   = len(out.get("approved_nodes", []))
                if _n_frozen > 0 or _n_kyc > 0:
                    st.toast(
                        f"✅ {_n_frozen} frozen · {_n_kyc} KYC · {_n_appr} approved",
                        icon="🛡️",
                    )
                else:
                    st.toast("No high/medium risk nodes found in target set.", icon="ℹ️")
                st.rerun()

            # -- Render post-intervention state
            i_out = st.session_state.get("intervention_graph_out")
            if i_out:
                i_analysis  = i_out["analysis"]
                i_stats     = i_out["stats"]
                frozen_list = i_stats.get("frozen_nodes", [])
                kyc_list    = i_stats.get("kyc_nodes",    [])
                appr_list   = i_stats.get("approved_nodes", [])
                edges_rm    = i_stats.get("edges_removed", 0)

                # Show tier explanation if no HIGH-risk nodes
                if not frozen_list and kyc_list:
                    st.warning(
                        "⚠️ No HIGH-risk nodes (Exit Imminent / Pre-Cashout) found in target.\n\n"
                        f"{len(kyc_list)} node(s) flagged for KYC (MEDIUM risk). "
                        "No edges removed.",
                        icon="🟡",
                    )
                elif not frozen_list and not kyc_list:
                    st.info(
                        "✅ All target nodes are LOW risk (Normal stage). No action required.",
                        icon="🛡️",
                    )

                # Post-intervention graph with frozen nodes greyed
                i_dna_lookup = dna_lookup.copy()
                for n in frozen_list:
                    i_dna_lookup[n] = 0

                if i_analysis and i_analysis.get("graph"):
                    fig_post = _build_forensic_graph(
                        G             = i_analysis["graph"],
                        pos           = hier_pos,
                        ring_accounts = ring_accounts,
                        cashout_nodes = cashout_nodes,
                        dna_lookup    = i_dna_lookup,
                        stage_lookup  = stage_lookup_g,
                        hop_lookup    = hop_lookup,
                        inr_rate      = INR_RATE,
                        intelligence_mode = True,
                        frozen_nodes  = frozen_list,
                        gnn_scores    = gnn_scores_g,
                        hybrid_scores = hybrid_scores_g,
                    )
                    st.plotly_chart(fig_post, use_container_width=True,
                                    config={"displayModeBar": False}, key="fig_post")

                # Metrics
                _loss_usd = i_out.get("loss_avoided", 0.0)
                _loss_inr = _loss_usd * INR_RATE
                st.success(
                    f"🛡️ **Projected Loss Avoided: ₹{_loss_inr:,.0f}** "
                    f"({edges_rm} transaction edges severed)",
                    icon="📈",
                )
                st.info(
                    f"Nodes: **{len(frozen_list)}** Frozen (High Risk) │ "
                    f"**{len(kyc_list)}** KYC Required (Medium Risk) │ "
                    f"**{len(appr_list)}** Approved (Low Risk)"
                )
            else:
                st.info("Select an intervention strategy above to visualize the post-mitigation state.")

    # Graph legend (outside tabs — always visible)
    st.markdown("""
    <div style='display:flex;gap:1.4rem;font-size:.66rem;color:#475569;margin-top:.3rem;flex-wrap:wrap;'>
        <span>&#9670; <span style='color:#ef4444;'>&#9670;</span> Cashout (diamond)</span>
        <span>&#9733; <span style='color:#3b82f6;'>&#9733;</span> Origin (star)</span>
        <span>&#9679; <span style='color:#dc2626;'>&#9679;</span> Exit Imminent</span>
        <span>&#9679; <span style='color:#f97316;'>&#9679;</span> Layering</span>
        <span>&#9679; <span style='color:#eab308;'>&#9679;</span> Compromised</span>
        <span>&#215; <span style='color:#6b7280;'>&#215;</span> Frozen</span>
        <span><span style='color:#a855f7;'>&#9675;</span> GNN Anom. Glow</span>
        <span>&#9658; Bold red = critical path &nbsp;&middot;&nbsp; Curved = long-hop &nbsp;&middot;&nbsp; Size &#8733; DNA</span>
    </div>""", unsafe_allow_html=True)



with alert_col:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>LIVE ALERT FEED</div>',
                unsafe_allow_html=True)
    n_crit = sum(1 for a in all_alerts if a["severity"] == "CRITICAL")
    n_high = sum(1 for a in all_alerts if a["severity"] == "HIGH")
    st.markdown(f"""
    <div style='display:flex;gap:.4rem;margin-bottom:.7rem;'>
        <div style='background:rgba(239,68,68,.1);border:1px solid #7f1d1d;
                    border-radius:6px;padding:.2rem .55rem;font-size:.65rem;
                    color:#ef4444;font-weight:700;'>🔴 {n_crit} CRITICAL</div>
        <div style='background:rgba(249,115,22,.1);border:1px solid #7c2d12;
                    border-radius:6px;padding:.2rem .55rem;font-size:.65rem;
                    color:#f97316;font-weight:700;'>🟠 {n_high} HIGH</div>
    </div>""", unsafe_allow_html=True)

    ah = '<div class="alert-scroll">'
    for a in all_alerts[:30]:
        ts_ = a["timestamp"].strftime("%H:%M:%S") if isinstance(a["timestamp"], datetime) else "—"
        ah += f"""<div class="alert-card" style="--ac:{a['color']}">
            <div class="alert-meta">{a['badge']} {a['severity']} · {a['category']} · {ts_}</div>
            <div class="alert-msg">{a['message']}</div>
        </div>"""
    ah += "</div>"
    st.markdown(ah, unsafe_allow_html=True)


st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 2: Live Countdown | Stage | DNA | Risk Gauge
# ─────────────────────────────────────────────────────────────────────────────

cd_col, stage_col, dna_col, gauge_col = st.columns([1, 1.1, 1.5, 1.4], gap="medium")

# ── Live Countdown (components.html — scripts execute inside iframe) ──────────
with cd_col:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>CASHOUT COUNTDOWN</div>',
                unsafe_allow_html=True)

    if predicted_exit_ts and remaining > 0:
        exit_ts_ms = int(predicted_exit_ts.timestamp() * 1000)
        countdown_html = f"""<!DOCTYPE html><html><head>
<style>
  body{{margin:0;padding:0;background:transparent;}}
  .wrap{{background:linear-gradient(135deg,#1a0505,#2d0a0a);
         border:1px solid #7f1d1d;border-radius:12px;
         padding:12px 16px;text-align:center;}}
  .lbl{{font-size:.58rem;color:#7f1d1d;letter-spacing:.1em;
        text-transform:uppercase;margin-bottom:6px;
        font-family:sans-serif;}}
  .dgt{{font-size:2.2rem;font-weight:700;
        font-family:'Courier New',monospace;color:#ef4444;
        animation:blink 1s ease-in-out infinite;}}
  .sub{{font-size:.58rem;color:#7f1d1d;margin-top:4px;font-family:sans-serif;}}
  @keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.5}}}}
</style></head><body>
<div class="wrap">
  <div class="lbl">⏱ Time To Cashout</div>
  <div class="dgt" id="cd">--:--</div>
  <div class="sub" id="cs">minutes · seconds</div>
</div>
<script>
  var T={exit_ts_ms};
  function tick(){{
    var d=T-Date.now(),el=document.getElementById("cd"),sl=document.getElementById("cs");
    if(!el){{setTimeout(tick,200);return;}}
    if(d<=0){{el.textContent="00:00";el.style.animation="none";
              if(sl)sl.textContent="⚠ Cashout Window Expired";return;}}
    var m=Math.floor(d/60000),s=Math.floor((d%60000)/1000);
    el.textContent=String(m).padStart(2,"0")+":"+String(s).padStart(2,"0");
    setTimeout(tick,1000);
  }}
  tick();
</script></body></html>"""
        components.html(countdown_html, height=115, scrolling=False)

    elif predicted_exit_ts and remaining <= 0:
        st.markdown("""
        <div class="cd-box" style='border-color:#dc2626;
             background:linear-gradient(135deg,#2d0505,#1a0000);text-align:center;'>
            <div class="cd-timer" style='color:#dc2626;font-size:0.95rem;'>00:00</div>
            <div style='font-size:.68rem;color:#dc2626;margin-top:.4rem;font-weight:600;'>
                ⚠️ Cashout Occurred — Funds Exited Network
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="cd-box" style='border-color:#14532d;
             background:linear-gradient(135deg,#0a1a0a,#0f2d1a);'>
            <div style='font-size:.6rem;color:#166534;letter-spacing:.1em;
                        text-transform:uppercase;'>✅ No Active Threat</div>
            <div class="cd-timer" style='color:#22c55e;'>—:—</div>
        </div>""", unsafe_allow_html=True)

# ── Laundering Stage ─────────────────────────────────────────────────────────
with stage_col:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>LAUNDERING STAGE</div>',
                unsafe_allow_html=True)
    stage_num  = ring_summary.get("max_stage", 0)
    stage_info = stage_predictor.STAGE_DEFINITIONS[stage_num]
    stages_cfg = [
        (0,"Normal","#22c55e"),
        (1,"Compromised","#eab308"),
        (2,"Layering","#f97316"),
        (3,"Pre-Cashout","#ef4444"),
        (4,"Exit Imminent","#dc2626"),
    ]
    for sn, sname, sc in stages_cfg:
        active = sn == stage_num
        bdr = f"1px solid {sc}" if active else "1px solid #1a2744"
        r, g, b_ = int(sc[1:3],16), int(sc[3:5],16), int(sc[5:7],16)
        bg  = f"rgba({r},{g},{b_},0.12)" if active else "transparent"
        wt  = "700" if active else "400"
        ico = stage_info["icon"] if active else "○"
        tc  = "#e2e8f0" if active else "#374151"
        st.markdown(f"""
        <div class="stage-step" style="border:{bdr};background:{bg};
                    font-weight:{wt};color:{tc};">
            <span>{ico}</span><span>Stage {sn} — {sname}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:.68rem;color:#374151;margin-top:.5rem;line-height:1.5;'>
        {stage_info['description']}</div>""", unsafe_allow_html=True)

# ── DNA Breakdown ─────────────────────────────────────────────────────────────
with dna_col:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>DNA BREAKDOWN — TOP RISK</div>',
                unsafe_allow_html=True)
    top = None
    if intelligence_mode and analysis and not analysis["top_risks"].empty:
        top = analysis["top_risks"].iloc[0]
    
    if top is not None:
        _gnn_raw = top.get("gnn_risk_score", None)
        gnn_score = float(_gnn_raw) if _gnn_raw is not None else (float(top.get("dna_score", 0)) / 100.0)
        gnn_pct = min(gnn_score * 100, 100.0)

        st.markdown(f"""
        <div style='font-size:.68rem;color:#475569;margin-bottom:.5rem;font-family:monospace;'>
            Node: <span style='color:#e2e8f0;'>{top['node']}</span>
            &nbsp;|&nbsp; DNA: <span style='color:#ef4444;font-weight:700;'>{top['dna_score']:.1f}</span>
            &nbsp;|&nbsp; GNN Conf: <span style='color:#a855f7;font-weight:700;'>{gnn_pct:.1f}%</span>
        </div>""", unsafe_allow_html=True)

        bars = [
            ("Fan-out Ratio",  top.get("fan_out_ratio",  0), "#ef4444"),
            ("Velocity Score", top.get("velocity_score", 0), "#f97316"),
            ("Burst Score",    top.get("burst_score",    0), "#eab308"),
            ("Circularity",    top.get("circularity",    0), "#3b82f6"),
            ("Amt Anomaly",    top.get("amount_anomaly", 0), "#8b5cf6"),
            ("Hop Proximity",  top.get("hop_proximity",  0), "#06b6d4"),
        ]
        for lbl, val, clr in bars:
            pct = min(float(val), 1.0) * 100
            st.markdown(f"""
            <div style='margin-bottom:.42rem;'>
                <div style='display:flex;justify-content:space-between;
                            font-size:.67rem;margin-bottom:.18rem;'>
                    <span style='color:#475569;'>{lbl}</span>
                    <span style='color:#e2e8f0;font-family:monospace;'>{val:.3f}</span>
                </div>
                <div class="dna-bar-bg">
                    <div class="dna-bar-fill" style="background:{clr};width:{pct:.1f}%;"></div>
                </div>
            </div>""", unsafe_allow_html=True)
            
        # GNN confidence bar
        st.markdown(f"""
        <div style='margin-top:.8rem;border-top:1px solid #1a2744;padding-top:.8rem;'>
            <div style='display:flex;justify-content:space-between;font-size:.62rem;margin-bottom:.3rem;'>
                <span style='color:#818cf8;font-weight:600;'>GNN STRUCTURAL CONFIDENCE</span>
                <span style='color:#818cf8;font-family:monospace;'>{gnn_pct:.1f}%</span>
            </div>
            <div style='height:3px;background:rgba(129,140,248,0.1);border-radius:2px;'>
                <div style='height:100%;width:{min(gnn_pct, 100):.1f}%;background:#818cf8;border-radius:2px;'></div>
            </div>
            <div style='font-size:.55rem;color:#334155;margin-top:.4rem;'>
                {gnn_layer.get_gnn_status()}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Risk Gauge ───────────────────────────────────────────────────────────────
with gauge_col:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>RISK GAUGE</div>',
                unsafe_allow_html=True)
    rs = float(summary.get("max_dna_score", 0.0))
    bar_color = "#ef4444" if rs >= 70 else "#f97316" if rs >= 50 else "#22c55e"
    fig_g2 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rs,
        number={"font": {"size": 32, "family": "JetBrains Mono", "color": "#e2e8f0"}},
        delta={"reference": 50, "increasing": {"color": "#ef4444"},
               "decreasing": {"color": "#22c55e"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#1a2744",
                     "tickfont": {"color": "#374151", "size": 9}},
            "bar": {"color": bar_color, "thickness": 0.22},
            "bgcolor": "#080f1e", "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(34,197,94,.07)"},
                {"range": [30, 50], "color": "rgba(234,179,8,.07)"},
                {"range": [50, 70], "color": "rgba(249,115,22,.07)"},
                {"range": [70,100], "color": "rgba(239,68,68,.1)"},
            ],
            "threshold": {"line": {"color": bar_color, "width": 3},
                          "thickness": 0.75, "value": rs},
        },
        title={"text": "Threat Index", "font": {"color": "#374151", "size": 11}},
    ))
    fig_g2.update_layout(paper_bgcolor="#050810", font_color="#e2e8f0",
                         height=210, margin=dict(l=15, r=15, t=15, b=5))
    st.plotly_chart(fig_g2, use_container_width=True, config={"displayModeBar": False})
    rl = ("🔴 CRITICAL THREAT" if rs >= 70 else "🟠 HIGH RISK" if rs >= 50
          else "🟡 MODERATE" if rs >= 30 else "🟢 LOW RISK")
    st.markdown(f"""<div style='text-align:center;font-size:.76rem;font-weight:700;
                    letter-spacing:.06em;color:{bar_color};'>{rl}</div>""",
                unsafe_allow_html=True)

    # Phase 4 — Baseline Comparison
    if intelligence_mode and analysis:
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        _dna_all   = analysis["dna_df"]["dna_score"].dropna()
        _susp_col  = "is_suspicious" if "is_suspicious" in analysis["dna_df"].columns else None
        if _susp_col and _susp_col in analysis["dna_df"].columns:
            _normal_df = analysis["dna_df"][~analysis["dna_df"][_susp_col].fillna(False)]
        else:
            _normal_df = analysis["dna_df"][analysis["dna_df"]["dna_score"] < 20]
        _mean_dna     = float(_dna_all.mean()) if not _dna_all.empty else 0
        _top_normal   = float(_normal_df["dna_score"].max()) if not _normal_df.empty else _mean_dna
        _deviation    = ((rs - _mean_dna) / max(_mean_dna, 1)) * 100
        _dev_color    = "#ef4444" if _deviation > 60 else "#f97316" if _deviation > 30 else "#22c55e"
        st.markdown(f"""
        <div style='background:#080f1e;border:1px solid #1a2744;border-radius:8px;
                    padding:.6rem .8rem;margin-top:.4rem;'>
            <div style='font-size:.58rem;color:#475569;text-transform:uppercase;
                        letter-spacing:.09em;margin-bottom:.5rem;'>Baseline Comparison</div>
            <div class='mrow'>
                <span class='mk'>Network Mean DNA</span>
                <span class='mv'>{_mean_dna:.1f}</span>
            </div>
            <div class='mrow'>
                <span class='mk'>Highest Non-Suspicious</span>
                <span class='mv'>{_top_normal:.1f}</span>
            </div>
            <div class='mrow'>
                <span class='mk'>Deviation from Mean</span>
                <span class='mv' style='color:{_dev_color};'>+{_deviation:.0f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# VELOCITY CHART + INTERVENTION
# ─────────────────────────────────────────────────────────────────────────────

vel_col, int_col = st.columns([2.4, 1.6], gap="medium")

with vel_col:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>TRANSACTION VELOCITY TIME-SERIES</div>',
                unsafe_allow_html=True)
    try:
        tx = transactions.copy()
        tx["bucket"] = tx["timestamp"].dt.floor("5min")
        # Guard: ensure is_suspicious column exists before groupby
        if "is_suspicious" not in tx.columns:
            tx["is_suspicious"] = False
        tx["is_suspicious"] = tx["is_suspicious"].fillna(False).astype(bool)
        vdf = tx.groupby(["bucket", "is_suspicious"]).size().reset_index(name="n")
        norm_v  = vdf[~vdf["is_suspicious"]].set_index("bucket")["n"]
        ring_v  = vdf[ vdf["is_suspicious"]].set_index("bucket")["n"]

        fv = go.Figure()
        fv.add_trace(go.Scatter(x=norm_v.index, y=norm_v.values, name="Normal",
                                mode="lines", line=dict(color="#3b82f6", width=1.5),
                                fill="tozeroy", fillcolor="rgba(59,130,246,.05)"))
        if is_attack and not ring_v.empty:
            fv.add_trace(go.Scatter(x=ring_v.index, y=ring_v.values, name="Ring/Suspicious",
                                    mode="lines+markers", line=dict(color="#ef4444", width=2),
                                    marker=dict(size=5, color="#ef4444"),
                                    fill="tozeroy", fillcolor="rgba(239,68,68,.08)"))
        fv.update_layout(
            paper_bgcolor="#050810", plot_bgcolor="#050810", height=240,
            font=dict(family="Inter", color="#64748b", size=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            xaxis=dict(showgrid=True, gridcolor="#1a2744", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#1a2744", zeroline=False,
                       title="Tx / 5 min"),
            margin=dict(l=45, r=15, t=8, b=35),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#0d1520", font_size=10),
        )
        st.plotly_chart(fv, use_container_width=True, config={"displayModeBar": False})
    except Exception as _vel_err:
        st.warning(f"⚠️ Velocity chart could not be rendered: {str(_vel_err)[:80]}")

with int_col:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>INTERVENTION SIMULATION</div>',
                unsafe_allow_html=True)
    if is_attack:
        ca, cb, cc = st.columns(3)
        with ca:
            if st.button("🧊 Freeze\nOrigin", use_container_width=True):
                st.session_state.intervention = "freeze_origin"
                st.session_state.intervention_out = alert_engine.compute_intervention_outcome(
                    "freeze_origin", ring_summary, transactions, ring_accounts)
        with cb:
            if st.button("❄️ Freeze\nRing", use_container_width=True):
                st.session_state.intervention = "freeze_ring"
                st.session_state.intervention_out = alert_engine.compute_intervention_outcome(
                    "freeze_ring", ring_summary, transactions, ring_accounts)
        with cc:
            if st.button("👁 Monitor\nOnly", use_container_width=True):
                st.session_state.intervention = "monitor_only"
                st.session_state.intervention_out = alert_engine.compute_intervention_outcome(
                    "monitor_only", ring_summary, transactions, ring_accounts)

        out = st.session_state.intervention_out
        # Guard: intervention_out (vel_col) holds alert_engine output {total_at_risk, ...}
        # intervention_graph_out (tab3) holds graph engine output {analysis, stats, ...}
        # Only render if it's the right shape.
        if out and isinstance(out, dict) and "total_at_risk" in out:
            labels = {"freeze_ring":   ("❄️ Full Ring Freeze", "#3b82f6"),
                      "freeze_origin": ("🧊 Origin Freeze",   "#f97316"),
                      "monitor_only":  ("👁 Monitor Only",     "#eab308")}
            al, ac = labels.get(st.session_state.intervention, ("Action", "#64748b"))
            st.markdown(f"""
            <div class="int-outcome">
                <div style='font-size:.67rem;font-weight:700;color:{ac};
                            letter-spacing:.08em;margin-bottom:.6rem;'>{al}</div>
                <div class="mrow"><span class="mk">Total at Risk</span>
                    <span class="mv">{_inr(out['total_at_risk'])}</span></div>
                <div class="mrow"><span class="mk">Est. Loss</span>
                    <span class="mv" style="color:#ef4444;">{_inr(out['estimated_loss'])}</span></div>
                <div class="mrow"><span class="mk">Loss Prevented</span>
                    <span class="mv" style="color:#22c55e;">{_inr(out['loss_prevented'])}</span></div>
                <div class="mrow"><span class="mk">Recovery %</span>
                    <span class="mv" style="color:#3b82f6;">{out['recovery_pct']:.0f}%</span></div>
            </div>""", unsafe_allow_html=True)
            st.progress(out["recovery_pct"] / 100)
            st.caption(f"💡 {out['recommendation']}")
    else:
        st.markdown("""
        <div style='background:#08101d;border:1px solid #1a2744;border-radius:10px;
                    padding:1.5rem;text-align:center;'>
            <div style='font-size:1.4rem;'>✅</div>
            <div style='font-size:.78rem;color:#374151;margin-top:.4rem;'>
                Normal baseline — no intervention needed.
            </div>
        </div>""", unsafe_allow_html=True)


st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# AI INTELLIGENCE PANEL (Gemini)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("🤖 AI Intelligence (Powered by Gemini)")

ai_left, ai_right = st.columns([2, 1.5], gap="medium")

with ai_left:
    has_ai = (
        st.session_state.ai_intelligence
        and st.session_state.ai_intelligence.get("laundering_stage")
            not in ("API key not configured", None, "")
    )
    ai_intel = st.session_state.ai_intelligence or {}

    # ── On-demand button — ONLY way to call Gemini for intelligence ──
    key_active = bool(gemini_layer._get_api_key())
    if not st.session_state.ai_intel_requested:
        btn_label = (
            "🤖 Generate AI Analysis"
            if key_active
            else "🔒 Gemini API Key Required"
        )
        if st.button(btn_label, use_container_width=True, disabled=not key_active,
                     key="btn_ai_analysis"):
            with st.spinner("� Calling Gemini 2.5 Flash…"):
                metrics = st.session_state.get("ai_metrics_json") or \
                          gemini_layer.build_metrics_json(pred_df, ring_summary)
                result_intel = gemini_layer.generate_intelligence_cached(
                    json.dumps(metrics, sort_keys=True)
                )
                st.session_state.ai_intelligence    = result_intel
                st.session_state.ai_intel_requested = True
            st.rerun()
    else:
        if st.button("🔄 Refresh AI Analysis", use_container_width=True,
                     key="btn_ai_refresh"):
            with st.spinner("🔄 Refreshing Gemini analysis…"):
                metrics = st.session_state.get("ai_metrics_json") or \
                          gemini_layer.build_metrics_json(pred_df, ring_summary)
                result_intel = gemini_layer.generate_intelligence_cached(
                    json.dumps(metrics, sort_keys=True)
                )
                st.session_state.ai_intelligence    = result_intel
                st.session_state.ai_intel_requested = True
            st.rerun()

    st.markdown("")

    if has_ai:
        st.caption("🔬 Gemini 2.5 Flash Analysis — Real-time AML risk assessment")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Laundering Stage**")
            # Always read from session_state — stage_predictor is the single source of truth.
            # ai_intel['laundering_stage'] is never used here.
            stage_engine = st.session_state.get("current_stage", "Normal")
            stage_icon = {
                "Normal": "🟢", "Compromised": "🟡",
                "Layering": "🟠", "Pre-Cashout": "🔴",
                "Exit Imminent": "🚨",
            }.get(stage_engine, "⚪")
            st.markdown(f"### {stage_icon} {stage_engine}")
            st.caption("Source: deterministic rule engine")
        with col2:
            st.markdown("**Confidence Level**")
            conf = ai_intel.get("confidence_level", "N/A")
            conf_icon = {
                "Critical": "🔴", "High": "🟠",
                "Moderate": "🟡", "Low": "🟢",
            }.get(conf, "⚪")
            st.markdown(f"### {conf_icon} {conf}")

        st.divider()
        st.markdown("**Recommended Action**")
        st.write(ai_intel.get("recommended_action", "—"))
        st.markdown("**Risk Reasoning**")
        st.write(ai_intel.get("risk_reasoning", "—"))

    elif st.session_state.ai_intel_requested:
        # Was requested but returned an error / quota message
        err = ai_intel.get("risk_reasoning", "")
        if "429" in err or "quota" in err.lower() or "exhausted" in err.lower():
            st.warning(
                "⚠️ **AI quota temporarily exceeded.**\n\n"
                "Using last available analysis. Try again in 60 seconds.",
                icon="⏳",
            )
        else:
            st.info(
                "🤖 **Gemini AI not connected.**\n\n"
                "Add `GEMINI_API_KEY` to your environment variables or "
                "`.streamlit/secrets.toml` to enable AI-powered threat intelligence.",
                icon="🔑",
            )
    else:
        st.info(
            "🤖 Click **Generate AI Analysis** to get Gemini-powered AML intelligence.",
            icon="✨",
        )

with ai_right:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>AI INVESTIGATION REPORT</div>',
                unsafe_allow_html=True)

    if st.button("📋 Generate AI Investigation Report", use_container_width=True):
        with st.spinner("🔄 Generating AML investigation report…"):
            summary_json = gemini_layer.build_summary_json(
                ring_summary, transactions, ring_accounts
            )
            report_text = gemini_layer.generate_investigation_report(summary_json)
            st.session_state.ai_report = report_text
            st.session_state.ai_report_requested = True

    if st.session_state.ai_report:
        with st.expander("📄 View Full Investigation Report", expanded=False):
            sections = _parse_report(st.session_state.ai_report)
            for title, body in sections:
                if title:
                    st.subheader(title)
                if body:
                    # Replace residual $NNN.NN amounts in report body with ₹
                    # Pattern: $ followed by digits/commas, optional decimal part.
                    # Does NOT capture trailing sentence periods.
                    body = re.sub(
                        r"\$(\d[\d,]*(?:\.\d+)?)",
                        lambda m: f"₹{_safe_float(m.group(1).replace(',', '')) * INR_RATE:,.0f}",
                        body,
                    )
                    st.write(body)
                    st.divider()


st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA TABLES (SQLite-backed)
# ─────────────────────────────────────────────────────────────────────────────

tbl_left, tbl_right = st.columns([1.4, 1], gap="medium")

with tbl_left:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>LIVE TRANSACTION TABLE</div>',
                unsafe_allow_html=True)

    tx_filter = st.selectbox(
        "Filter",
        ["All Transactions", "Suspicious Only"],
        label_visibility="collapsed",
        key="tx_filter",
    )
    tx_db = db.get_transactions(
        limit=200,
        suspicious_only=(tx_filter == "Suspicious Only"),
    )

    if not tx_db.empty:
        tx_display = tx_db[["id","sender","receiver","amount","timestamp","tx_type"]].copy()
        tx_display.columns = ["ID","From","To","Amount (₹)","Timestamp","Type"]
        tx_display["Amount (₹)"] = tx_display["Amount (₹)"].apply(
            lambda v: _inr(_safe_float(v))
        )
        st.dataframe(
            tx_display,
            use_container_width=True,
            height=300,
            column_config={"ID": st.column_config.NumberColumn(width="small")},
        )
    else:
        st.info("No transactions in database yet. Run a simulation first.")

with tbl_right:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>ACCOUNT STATUS TABLE</div>',
                unsafe_allow_html=True)

    risk_stages = ["All", "Normal", "Compromised", "Layering", "Pre-Cashout", "Exit Imminent"]
    risk_filter = st.selectbox(
        "Risk Level Filter",
        risk_stages,
        label_visibility="collapsed",
        key="risk_filter",
    )

    acc_db = db.get_accounts(
        risk_filter=risk_filter if risk_filter != "All" else None,
        compromised_only=False,
    )
    if not acc_db.empty:
        acc_display = acc_db.copy()
        acc_display["compromised"] = acc_display["compromised"].map(
            {1: "⚠️ Yes", 0: "✅ No"})
        acc_display["risk_score"]  = acc_display["risk_score"].map("{:.1f}".format)
        acc_display.columns = ["Account ID","Ring Member","DNA Score","Stage"]
        st.dataframe(
            acc_display,
            use_container_width=True,
            height=300,
            column_config={
                "DNA Score": st.column_config.NumberColumn(format="%.1f"),
            },
        )
    else:
        st.info("No account data in database yet.")


st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOP RISK TABLE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>TOP RISK NODES — DNA INTELLIGENCE TABLE</div>',
            unsafe_allow_html=True)

if intelligence_mode and not pred_df.empty:
    display_cols = [
        "node", "dna_score", "risk_level", "stage_label",
        "gnn_score", "hybrid_score",
        "fan_out_ratio", "velocity_score", "burst_score",
        "hops_to_cashout", "cashout_probability", "time_to_cashout_min",
    ]
    # Filter only available columns to prevent KeyError
    valid_cols = [c for c in display_cols if c in pred_df.columns]
    top_tbl = pred_df[valid_cols].head(15).copy()

    col_cfg = {
        "dna_score":          st.column_config.NumberColumn(label="DNA Score",   format="%.1f"),
        "cashout_probability": st.column_config.ProgressColumn(
                               label="Cashout %", format="%.1f%%", min_value=0, max_value=100),
        "time_to_cashout_min": st.column_config.NumberColumn(label="ETA (min)",  format="%.0f"),
    }
    if "gnn_score" in valid_cols:
        col_cfg["gnn_score"]    = st.column_config.ProgressColumn(
            label="\U0001f9e0 GNN Score", format="%.3f", min_value=0.0, max_value=1.0)
    if "hybrid_score" in valid_cols:
        col_cfg["hybrid_score"] = st.column_config.ProgressColumn(
            label="Hybrid", format="%.3f", min_value=0.0, max_value=1.0)

    st.dataframe(
        top_tbl,
        use_container_width=True,
        height=380,
        column_config=col_cfg,
    )

else:
    st.info("Top risk intelligence table will populate after analysis.")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — CRYPTOGRAPHIC EXPOSURE & PQC READINESS
# ─────────────────────────────────────────────────────────────────────────────

_qs_dna   = float(pred_df["dna_score"].max()) if (intelligence_mode and not pred_df.empty and "dna_score" in pred_df.columns) else 0.0
_qs_stage = st.session_state.get("current_stage", "Normal")

if _qs_dna >= 35 or _qs_stage == "Exit Imminent":
    _qs_class, _qs_color, _qs_risk = "Quantum Vulnerable",  "#ef4444", "CRITICAL"
    _qs_tls, _qs_cipher = "TLS 1.2 (RSA-2048)", "AES-256-CBC + HMAC-SHA256"
    _qs_kex,  _qs_fs    = "ECDH (secp256r1)",   "Partial (session-only)"
elif _qs_dna >= 25 or _qs_stage in ("Pre-Cashout", "Layering"):
    _qs_class, _qs_color, _qs_risk = "Transition Required", "#f97316", "HIGH"
    _qs_tls, _qs_cipher = "TLS 1.2 / 1.3 Mixed", "AES-256-GCM"
    _qs_kex,  _qs_fs    = "ECDH (X25519)",         "Yes (DHE)"
elif _qs_stage == "Compromised":
    _qs_class, _qs_color, _qs_risk = "PQC Ready",           "#3b82f6", "MODERATE"
    _qs_tls, _qs_cipher = "TLS 1.3",  "ChaCha20-Poly1305"
    _qs_kex,  _qs_fs    = "X25519 + Kyber-768", "Yes (PQC-Hybrid)"
else:
    _qs_class, _qs_color, _qs_risk = "Fully Quantum Safe",  "#22c55e", "LOW"
    _qs_tls, _qs_cipher = "TLS 1.3",  "AES-256-GCM-SHA384"
    _qs_kex,  _qs_fs    = "CRYSTALS-Kyber (NIST L3)", "Yes (PQC-Native)"

st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>CRYPTOGRAPHIC EXPOSURE &amp; PQC READINESS</div>',
            unsafe_allow_html=True)
_qs_l, _qs_r = st.columns([1.6, 1], gap="medium")

with _qs_l:
    _rgb = ",".join(str(int(bytes.fromhex(_qs_color[1:])[i])) for i in range(3))
    st.markdown(f"""
    <div style='background:#080f1e;border:1px solid #1a2744;border-radius:12px;padding:1rem 1.2rem;'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:.8rem;'>
            <span style='font-size:.65rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;'>Cryptographic Profile</span>
            <span class='pqc-badge' style='color:{_qs_color};border-color:{_qs_color};background:rgba({_rgb},0.1);'>{_qs_class}</span>
        </div>
        <div class='qrow'><span class='qk'>TLS Version</span>    <span class='qv'>{_qs_tls}</span></div>
        <div class='qrow'><span class='qk'>Cipher Suite</span>   <span class='qv'>{_qs_cipher}</span></div>
        <div class='qrow'><span class='qk'>Key Exchange</span>   <span class='qv'>{_qs_kex}</span></div>
        <div class='qrow'><span class='qk'>Forward Secrecy</span><span class='qv'>{_qs_fs}</span></div>
        <div class='qrow'><span class='qk'>Risk Category</span>
            <span class='qv' style='color:{_qs_color};font-weight:700;'>{_qs_risk}</span></div>
    </div>""", unsafe_allow_html=True)

with _qs_r:
    if _qs_class == "Quantum Vulnerable":
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1a0505,#2d0808);border:1px solid {_qs_color};
                    border-radius:12px;padding:1rem 1.1rem;text-align:center;'>
            <div style='font-size:.58rem;color:{_qs_color};text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:.5rem;'>⚠ Quantum Threat Detected</div>
            <div style='font-size:.76rem;color:#e2e8f0;line-height:1.7;'>
                RSA / ECDH key material is vulnerable to Shor's algorithm.
                Blockchain anchoring creates a <b>tamper-proof pre-quantum audit record</b>.
            </div>
            <div style='margin-top:.6rem;font-size:.63rem;color:{_qs_color};font-weight:700;'>
                ↓ Anchor Evidence Now
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#050c18;border:1px solid #1a2744;border-radius:12px;
                    padding:1rem 1.1rem;text-align:center;'>
            <div style='font-size:.58rem;color:{_qs_color};text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:.5rem;'>✔ Cryptographic Status</div>
            <div style='font-size:.76rem;color:#64748b;line-height:1.7;'>
                Rated <b style='color:{_qs_color};'>{_qs_class}</b>.<br>No immediate quantum exposure.
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# IMMUTABLE AUDIT ANCHOR — Blockchain Proof of Detection
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="sec-hdr"><span class="sec-dot"></span>IMMUTABLE AUDIT ANCHOR — BLOCKCHAIN PROOF OF DETECTION</div>
""", unsafe_allow_html=True)


_top_dna_score = float(pred_df["dna_score"].max()) if (intelligence_mode and not pred_df.empty and "dna_score" in pred_df.columns) else 0.0

# Always read stage from session_state — single source of truth
_top_stage = str(st.session_state.get("current_stage") or
                 ring_summary.get("dominant_label", "Normal"))

# Phase 4: full PQC tier vocabulary — 4 tiers, no truncation
if _top_dna_score >= 35 or _top_stage == "Exit Imminent":
    _pqc_status = "Quantum Vulnerable"
elif _top_dna_score >= 25 or _top_stage in ("Pre-Cashout", "Layering"):
    _pqc_status = "Transition Required"
elif _top_stage == "Compromised":
    _pqc_status = "PQC Ready"
else:
    _pqc_status = "Fully Quantum Safe"

_anch_left, _anch_right = st.columns([1.6, 1], gap="medium")

with _anch_left:
    eligible = blockchain_layer.should_anchor(
        _top_dna_score, _top_stage, _pqc_status
    )
    # Phase 4: fixed-width metric row so PQC label never wraps
    st.markdown(f"""
    <div style='display:flex;gap:.6rem;flex-wrap:nowrap;margin-bottom:.8rem;'>
        <div style='flex:1;background:#080f1e;border:1px solid #1a2744;border-radius:8px;
                    padding:.5rem .7rem;min-width:0;'>
            <div style='font-size:.58rem;color:#475569;text-transform:uppercase;
                        letter-spacing:.09em;'>Top DNA</div>
            <div style='font-size:1.05rem;font-weight:700;color:{"#ef4444" if _top_dna_score>=25 else "#22c55e"};
                        font-family:JetBrains Mono,monospace;'>{_top_dna_score:.1f}</div>
        </div>
        <div style='flex:1.4;background:#080f1e;border:1px solid #1a2744;border-radius:8px;
                    padding:.5rem .7rem;min-width:0;'>
            <div style='font-size:.58rem;color:#475569;text-transform:uppercase;
                        letter-spacing:.09em;'>Stage</div>
            <div style='font-size:.85rem;font-weight:700;color:#e2e8f0;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{_top_stage}</div>
        </div>
        <div style='flex:1.8;background:#080f1e;border:1px solid #1a2744;border-radius:8px;
                    padding:.5rem .7rem;min-width:0;'>
            <div style='font-size:.58rem;color:#475569;text-transform:uppercase;
                        letter-spacing:.09em;'>PQC Status</div>
            <div style='font-size:.78rem;font-weight:700;white-space:nowrap;
                        color:{"#ef4444" if "Vulnerable" in _pqc_status else "#f97316" if "Transition" in _pqc_status else "#3b82f6" if "Ready" in _pqc_status else "#22c55e"};'>{_pqc_status}</div>
        </div>
        <div style='flex:1;background:#080f1e;border:1px solid {"#22c55e" if eligible else "#1a2744"};
                    border-radius:8px;padding:.5rem .7rem;min-width:0;'>
            <div style='font-size:.58rem;color:#475569;text-transform:uppercase;
                        letter-spacing:.09em;'>Eligible</div>
            <div style='font-size:.95rem;font-weight:700;color:{"#22c55e" if eligible else "#374151"};'>
                {"✔ Yes" if eligible else "✘ No"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    if not st.session_state.anchored:
        if eligible:
            if st.button(
                "⛓️  Anchor to Ethereum Sepolia",
                use_container_width=True,
                key="btn_anchor",
            ):
                top_alert = all_alerts[0] if all_alerts else {}
                anchor_payload = {
                    "alert":         top_alert,
                    "dna_score":     _top_dna_score,
                    "stage":         _top_stage,
                    "pqc_status":    _pqc_status,
                    "timestamp":     datetime.now().isoformat(),
                }
                alert_hash = blockchain_layer.hash_alert(anchor_payload)
                with st.spinner("🔗 Anchoring to Sepolia…"):
                    anchor = blockchain_layer.anchor_to_blockchain(alert_hash)
                st.session_state.anchor_result = {
                    "alert_hash":    alert_hash,
                    "tx_hash":       anchor["tx_hash"],
                    "mode":          anchor["mode"],
                    "etherscan_url": anchor["etherscan_url"],
                }
                st.session_state.anchored = True
                st.rerun()
        else:
            st.info(
                "⛓️ Blockchain anchoring activates when **Exit Imminent** stage "
                "is detected with DNA score ≥ 25 and Quantum Vulnerable status.",
                icon="🔐",
            )
    else:
        ar_mode = (st.session_state.anchor_result or {}).get("mode", "Simulation")

        mode_icon = "🟢" if ar_mode == "Live Sepolia" else "🟡"

        st.success(f"✔️ Alert anchored! Mode: **{ar_mode}** {mode_icon}", icon="⛓️")

with _anch_right:
    ar = st.session_state.anchor_result
    if ar:
        mode_color = "#22c55e" if ar["mode"] == "Live Sepolia" else "#f97316"
        st.markdown(f"""
        <div style='background:#080f1e;border:1px solid #1a2744;border-radius:10px;
                    padding:1rem 1.1rem;font-size:.72rem;line-height:2;'>
            <div style='color:#64748b;text-transform:uppercase;letter-spacing:.08em;
                        font-size:.58rem;margin-bottom:.3rem;'>Audit Record</div>
            <div style='color:#94a3b8;'>SHA-256&#8195;
                <span style='color:#e2e8f0;font-family:monospace;font-size:.65rem;'>
                    {ar['alert_hash'][:24]}…{ar['alert_hash'][-8:]}
                </span>
            </div>
            <div style='color:#94a3b8;'>TX Hash&#8195;
                <span style='color:#3b82f6;font-family:monospace;font-size:.65rem;'>
                    {ar['tx_hash'][:18]}…{ar['tx_hash'][-6:]}
                </span>
            </div>
            <div style='color:#94a3b8;'>Mode&#8195;
                <span style='color:{mode_color};font-weight:600;'>{ar['mode']}</span>
            </div>
            <div style='margin-top:.5rem;'>
                {"🔗 <a href='" + ar['etherscan_url'] + "' target='_blank' style='color:#3b82f6;text-decoration:none;font-size:.7rem;'>View on Sepolia Etherscan →</a>" if ar['mode'] == 'Live Sepolia' else "<span style='color:#22c55e;font-size:.68rem;font-weight:600;'>✔ Anchored successfully (Testnet Simulation)</span>"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#080f1e;border:1px dashed #1a2744;border-radius:10px;
                    padding:1.5rem;text-align:center;color:#374151;font-size:.75rem;'>
            ⛓️ No anchor yet<br>
            <span style='font-size:.65rem;'>Trigger anchoring from the left panel</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='border-top:1px solid #1a2744;margin-top:2rem;padding-top:.9rem;
            display:flex;justify-content:space-between;align-items:center;'>
    <div style='font-size:.65rem;color:#475569;'>
        🔍 ChronoTrace v2.0 · AI-Powered FinCrime Intelligence
    </div>
    <div style='font-size:.65rem;color:#475569;'>
        Gemini 2.5 Flash · NetworkX · Ethereum Sepolia · Streamlit
    </div>
    <div style='font-size:.65rem;color:#475569;'>
        © 2025 ChronoTrace Labs
    </div>
</div>
""", unsafe_allow_html=True)
