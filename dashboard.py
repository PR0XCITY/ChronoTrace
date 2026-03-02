"""
dashboard.py — ChronoTrace | Enterprise FinCrime Intelligence Dashboard
Main Streamlit entry point for Streamlit Cloud deployment.

Run with:  streamlit run dashboard.py

Requires GEMINI_API_KEY in environment or .streamlit/secrets.toml
"""

import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Internal modules ─────────────────────────────────────────────────────────
import database as db
import simulator
import graph_engine
import stage_predictor
import alerts as alert_engine
import gemini_layer
import re
import streamlit.components.v1 as components
import dataset_loader
import blockchain_layer

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
# FORENSIC GRAPH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

import networkx as nx

def _hierarchical_layout(
    G: "nx.DiGraph",
    ring_accounts: list,
    cashout_nodes: list,
) -> dict:
    """
    Assign left-to-right x positions by node role:
        Column 0  — Origin (high in-degree from normal, high out-degree to ring)
        Column 1  — Layer-1 mules (direct successors of origin)
        Column 2  — Layer-2 mules (successors of L1)
        Column 3  — Pre-cashout aggregators
        Column 4  — Cashout sinks
        Column 5  — All other (normal) nodes

    Y positions are spread evenly within each column with a small jitter.
    Returns dict {node: (x, y)}.
    """
    import random as _rnd
    _rnd.seed(7)  # deterministic jitter

    ring_set    = set(ring_accounts)
    cashout_set = set(cashout_nodes)

    # Identify origin: ring node with no in-edges from other ring nodes
    origin_set = set()
    for n in ring_set:
        if n in G and not any(p in ring_set for p in G.predecessors(n)):
            origin_set.add(n)
    if not origin_set and ring_set:
        # Fallback: highest out-degree ring node
        origin_set = {max(ring_set, key=lambda n: G.out_degree(n) if n in G else 0)}

    # BFS columns for ring nodes
    col_map: dict = {}
    for o in origin_set:
        col_map[o] = 0
    queue = list(origin_set)
    visited = set(origin_set)
    while queue:
        node = queue.pop(0)
        cur_col = col_map.get(node, 0)
        for succ in (G.successors(node) if node in G else []):
            if succ in ring_set and succ not in visited:
                next_col = min(cur_col + 1, 3 if succ not in cashout_set else 4)
                col_map[succ] = next_col
                visited.add(succ)
                queue.append(succ)

    # Cashout column
    for n in cashout_set:
        col_map[n] = 4

    # Group by column
    col_nodes: dict = {}
    for n, c in col_map.items():
        col_nodes.setdefault(c, []).append(n)

    # Normal nodes — last column
    normal_nodes = [n for n in G.nodes() if n not in ring_set and n not in cashout_set]
    col_nodes[5] = normal_nodes

    # Build positions
    pos = {}
    x_spacing = 2.2
    for col, nodes in sorted(col_nodes.items()):
        x = col * x_spacing
        n_nodes = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            y = (i - n_nodes / 2) * 1.1 + _rnd.uniform(-0.18, 0.18)
            pos[node] = (x, y)

    # Nodes not in any column — scatter to the right
    for node in G.nodes():
        if node not in pos:
            pos[node] = (5 * x_spacing + _rnd.uniform(-0.5, 0.5),
                         _rnd.uniform(-5, 5))
    return pos


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
) -> go.Figure:
    """
    Construct a forensic-quality directed Plotly graph.

    Layers (bottom to top):
      1. Faint background edges (non-ring)
      2. Bold highlighted ring-path edges
      3. Arrow markers (direction)
      4. Amount annotations on main path
      5. Nodes (size ∝ DNA)
    """
    ring_set    = set(ring_accounts)
    cashout_set = set(cashout_nodes)

    # Determine origin node
    origin_node = None
    if ring_set:
        ring_dns = {n: dna_lookup.get(n, 0) for n in ring_set if n in pos}
        if ring_dns:
            origin_node = max(ring_dns, key=ring_dns.get)

    # Which nodes to render
    if focus_ring:
        render_nodes = [n for n in pos if n in ring_set or n in cashout_set]
    else:
        render_nodes = list(pos.keys())[:300]

    render_set = set(render_nodes)

    # ── Classify edges ─────────────────────────────────────────────────────────
    all_edges     = [(s, d, dat) for s, d, dat in G.edges(data=True)
                     if s in pos and d in pos and s in render_set and d in render_set]
    ring_edges    = [(s, d, dat) for s, d, dat in all_edges
                     if s in ring_set or d in ring_set]
    bg_edges      = [(s, d, dat) for s, d, dat in all_edges
                     if s not in ring_set and d not in ring_set]

    # Sample background to avoid overload
    if len(bg_edges) > 200:
        import random as _r
        _r.seed(42)
        bg_edges = _r.sample(bg_edges, 200)

    def _edge_xy(edges):
        ex, ey = [], []
        for s, d, _ in edges:
            x0, y0 = pos[s]; x1, y1 = pos[d]
            ex.extend([x0, x1, None]); ey.extend([y0, y1, None])
        return ex, ey

    fig = go.Figure()

    # Layer 1 — faint background edges
    if not focus_ring:
        bx, by = _edge_xy(bg_edges)
        if bx:
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode="lines",
                line=dict(width=0.3, color="rgba(99,102,241,0.07)"),
                hoverinfo="none", showlegend=False,
            ))

    # Layer 2 — bold ring-path edges
    rx, ry = _edge_xy(ring_edges)
    if rx:
        fig.add_trace(go.Scatter(
            x=rx, y=ry, mode="lines",
            line=dict(width=2.0, color="rgba(239,68,68,0.55)"),
            hoverinfo="none", showlegend=False,
        ))

    # Layer 3 — directed arrow markers at 80% along ring edges
    ax_pts, ay_pts, a_colors = [], [], []
    for s, d, _ in ring_edges:
        x0, y0 = pos[s]; x1, y1 = pos[d]
        ax_pts.append(x0 + 0.80 * (x1 - x0))
        ay_pts.append(y0 + 0.80 * (y1 - y0))
        a_colors.append("#ef4444" if d in cashout_set else "#f97316")

    if ax_pts:
        fig.add_trace(go.Scatter(
            x=ax_pts, y=ay_pts, mode="markers",
            marker=dict(symbol="triangle-right", size=9,
                        color=a_colors,
                        line=dict(width=0)),
            hoverinfo="none", showlegend=False,
        ))

    # Layer 4 — amount labels on main ring path (top 30 edges by weight)
    ann_edges = sorted(ring_edges, key=lambda e: e[2].get("weight", 0), reverse=True)[:30]
    annotations = []
    for s, d, dat in ann_edges:
        x0, y0 = pos[s]; x1, y1 = pos[d]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        amt = dat.get("weight", 0)
        annotations.append(dict(
            x=mx, y=my,
            text=f"₹{amt * inr_rate:,.0f}",
            showarrow=False,
            font=dict(size=7, color="#f97316", family="JetBrains Mono"),
            bgcolor="rgba(5,8,16,0.85)",
            borderpad=2,
        ))

    # Layer 5 — nodes
    n_colors, n_sizes, n_borders, n_hover, nx_pos, ny_pos = [], [], [], [], [], []
    for node in render_nodes:
        dna   = dna_lookup.get(node, 0)
        stage = stage_lookup.get(node, "Normal")
        hops  = hop_lookup.get(node, -1)

        out_w = sum(d.get("weight", 0) for _, _, d in G.out_edges(node, data=True))
        in_w  = sum(d.get("weight", 0) for _, _, d in G.in_edges(node, data=True))

        # Size scales with DNA score
        base_size = max(7, min(28, 7 + dna * 0.45))

        if node in cashout_set:
            color, border = "#ef4444", "#ff6b6b"
            base_size = max(base_size, 20)
        elif node == origin_node:
            color, border = "#3b82f6", "#93c5fd"
            base_size = max(base_size, 18)
        elif node in ring_set:
            color  = "#f97316" if dna >= 28 else "#eab308"
            border = "#fbbf24"
        else:
            alpha = max(0.08, dna / 100)
            color  = f"rgba(59,130,246,{alpha:.2f})"
            border = "rgba(255,255,255,0.04)"
            base_size = min(base_size, 8)

        tag = ("\U0001f534 CASHOUT" if node in cashout_set else
               "\U0001f535 ORIGIN"  if node == origin_node else
               "\U0001f7e0 RING"    if node in ring_set else "\u26aa NORMAL")
        hops_str = str(hops) if hops >= 0 else "unreachable"

        hover = (
            f"<b>{node}</b><br>"
            f"Role: {tag}<br>"
            f"DNA Score: <b>{dna:.1f}</b><br>"
            f"Stage: {stage}<br>"
            f"In-flow:  ₹{in_w * inr_rate:,.0f}<br>"
            f"Out-flow: ₹{out_w * inr_rate:,.0f}<br>"
            f"Hops to cashout: {hops_str}<br>"
            f"Degree: {G.in_degree(node)}→{G.out_degree(node)}"
        )

        x, y = pos[node]
        nx_pos.append(x); ny_pos.append(y)
        n_colors.append(color)
        n_sizes.append(base_size)
        n_borders.append(border)
        n_hover.append(hover)

    fig.add_trace(go.Scatter(
        x=nx_pos, y=ny_pos, mode="markers",
        marker=dict(size=n_sizes, color=n_colors,
                    line=dict(width=1.4, color=n_borders)),
        text=n_hover,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    graph_h = 480 if not focus_ring else 420
    fig.update_layout(
        paper_bgcolor="#050810", plot_bgcolor="#050810",
        height=graph_h,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title=dict(text="← Origin        Layering        Pre-Cashout        Cashout →",
                               font=dict(size=9, color="#374151"))),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=30),
        annotations=annotations,
        hoverlabel=dict(bgcolor="#0d1520", font_size=11,
                        font_family="JetBrains Mono", font_color="#e2e8f0"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
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
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "sim_result":           None,
        "analysis":             None,
        "pred_df":              None,
        "alerts":               [],
        "ring_summary":         {},
        "intervention":          None,
        "intervention_out":      None,
        "last_mode":             None,
        "predicted_exit_ts":     None,
        "ai_intelligence":       None,
        "ai_report":             None,
        "ai_report_requested":   False,
        "ai_intel_requested":    False,
        "ai_metrics_json":       None,
        "anchored":              False,
        "anchor_result":         None,
        "current_stage":         "Normal",   # set by run_pipeline; read by AI panel
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(mode: str, n_rings: int = 1):
    """Run full intelligence pipeline and cache everything to session state."""
    with st.spinner("🔄 Running intelligence pipeline…"):

        if mode == "dataset":
            # Dataset-backed path — no random generation
            result = dataset_loader.build_sim_result_from_dataset()
        else:
            # Synthetic simulation path
            result = simulator.run_and_persist(mode=mode, n_rings=n_rings)

        analysis = graph_engine.analyse_from_db(result)
        pred_df  = stage_predictor.predict(analysis["dna_df"])

        if mode != "dataset":
            simulator.persist_predictions(result, pred_df, analysis["dna_df"])

        ring_summary = stage_predictor.predict_ring_summary(
            pred_df, result["ring_accounts"])

        all_alerts = alert_engine.generate_all_alerts(result, analysis, pred_df)

        ttc = ring_summary.get("min_time_to_cashout", -1)
        predicted_exit_ts = (
            datetime.now() + timedelta(minutes=ttc) if ttc >= 0 else None
        )

        metrics_json = gemini_layer.build_metrics_json(pred_df, ring_summary)

    st.session_state.sim_result          = result
    st.session_state.analysis            = analysis
    st.session_state.pred_df             = pred_df
    st.session_state.alerts              = all_alerts
    st.session_state.ring_summary        = ring_summary
    st.session_state.predicted_exit_ts   = predicted_exit_ts
    st.session_state.ai_metrics_json     = metrics_json
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
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

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
                "summary":       st.session_state.analysis["summary"],
                "ring_summary":  st.session_state.ring_summary,
                "ai_analysis":   st.session_state.ai_intelligence,
                "top_risks":     st.session_state.analysis["top_risks"].to_dict("records"),
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
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='display:flex;align-items:center;justify-content:space-between;
            border-bottom:1px solid #1a2744;padding-bottom:.9rem;margin-bottom:1.3rem;'>
    <div>
        <div style='font-size:1.5rem;font-weight:800;letter-spacing:.05em;color:#f1f5f9;'>
            🔍 ChronoTrace
        </div>
        <div style='font-size:.7rem;color:#374151;letter-spacing:.08em;margin-top:.1rem;'>
            PREDICTING FINANCIAL CRIME BEFORE MONEY DISAPPEARS
        </div>
    </div>
    <div style='text-align:right;'>
        <div style='font-size:.65rem;color:#1e293b;font-family:monospace;'>LIVE THREAT INTELLIGENCE</div>
        <div style='font-size:.8rem;color:#3b82f6;font-weight:700;font-family:monospace;'>● SYSTEM ACTIVE</div>
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

result       = st.session_state.sim_result
analysis     = st.session_state.analysis
pred_df      = st.session_state.pred_df
all_alerts   = st.session_state.alerts
ring_summary = st.session_state.ring_summary
summary      = analysis["summary"]
G            = analysis["graph"]
transactions = result["transactions"]
ring_accounts= result["ring_accounts"]
cashout_nodes= result.get("cashout_nodes", [])
is_attack    = result.get("mode") == "attack"
ai_intel     = st.session_state.ai_intelligence or {}


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
    remaining = (predicted_exit_ts - datetime.now()).total_seconds() / 60
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

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)


# NETWORK GRAPH + ALERT FEED
# ─────────────────────────────────────────────────────────────────────────────

graph_col, alert_col = st.columns([3, 1.15], gap="medium")

with graph_col:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>FORENSIC TRANSACTION GRAPH — HIERARCHICAL DIRECTED</div>',
                unsafe_allow_html=True)

    # Pre-compute lookups once
    dna_df_graph = analysis["dna_df"]
    dna_lookup   = dna_df_graph.set_index("node")["dna_score"].to_dict()
    hop_lookup   = dna_df_graph.set_index("node")["hops_to_cashout"].to_dict() \
                   if "hops_to_cashout" in dna_df_graph.columns else {}
    stage_lookup_g = {}
    if not pred_df.empty and "node" in pred_df.columns and "stage_label" in pred_df.columns:
        stage_lookup_g = pred_df.set_index("node")["stage_label"].to_dict()

    # Phase 6: use suspicious subgraph when full graph > 300 nodes
    GRAPH_TOO_LARGE = G.number_of_nodes() > 300

    # Phase 1: compute hierarchical layout (cached in session_state by graph size)
    layout_key = f"_hier_layout_{G.number_of_nodes()}_{len(ring_accounts)}_{len(cashout_nodes)}"
    if layout_key not in st.session_state:
        st.session_state[layout_key] = _hierarchical_layout(G, ring_accounts, cashout_nodes)
    hier_pos = st.session_state[layout_key]

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

    with st.expander("📁 Transaction Network", expanded=True):
        fig_forensic = _build_forensic_graph(
            G            = G,
            pos          = hier_pos,
            ring_accounts= ring_accounts,
            cashout_nodes= cashout_nodes,
            dna_lookup   = dna_lookup,
            stage_lookup = stage_lookup_g,
            hop_lookup   = hop_lookup,
            inr_rate     = INR_RATE,
            focus_ring   = focus_ring_mode,
        )
        st.plotly_chart(fig_forensic, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown("""
        <div style='display:flex;gap:1.6rem;font-size:.67rem;color:#475569;margin-top:-.5rem;
                    flex-wrap:wrap;'>
            <span>● <span style='color:#ef4444;'>●</span> Cashout sink</span>
            <span>● <span style='color:#3b82f6;'>●</span> Origin node</span>
            <span>● <span style='color:#f97316;'>●</span> High-risk ring (DNA≥28)</span>
            <span>● <span style='color:#eab308;'>●</span> Mid-ring</span>
            <span>● <span style='color:#475569;'>●</span> Normal node</span>
            <span>▶ Arrow = tx direction &nbsp;·&nbsp; Size ∝ DNA score &nbsp;·&nbsp; Labels = top amounts</span>
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
        exit_iso = predicted_exit_ts.strftime("%Y-%m-%dT%H:%M:%S")
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
  var T=new Date("{exit_iso}").getTime();
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
            <div class="cd-timer" style='color:#dc2626;font-size:1.1rem;'>00:00</div>
            <div style='font-size:.68rem;color:#dc2626;margin-top:.4rem;font-weight:600;'>
                ⚠️ Cashout Window Expired
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
    top = analysis["top_risks"].iloc[0] if not analysis["top_risks"].empty else None
    if top is not None:
        st.markdown(f"""
        <div style='font-size:.68rem;color:#475569;margin-bottom:.5rem;font-family:monospace;'>
            Node: <span style='color:#e2e8f0;'>{top['node']}</span>
            &nbsp;|&nbsp; DNA: <span style='color:#ef4444;font-weight:700;'>{top['dna_score']:.1f}</span>
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

# ── Risk Gauge ───────────────────────────────────────────────────────────────
with gauge_col:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>RISK GAUGE</div>',
                unsafe_allow_html=True)
    rs = summary["max_dna_score"]
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


st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# VELOCITY CHART + INTERVENTION
# ─────────────────────────────────────────────────────────────────────────────

vel_col, int_col = st.columns([2.4, 1.6], gap="medium")

with vel_col:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>TRANSACTION VELOCITY TIME-SERIES</div>',
                unsafe_allow_html=True)
    tx = transactions.copy()
    tx["bucket"] = tx["timestamp"].dt.floor("5min")
    vdf = tx.groupby(["bucket","is_suspicious"]).size().reset_index(name="n")
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
        if out:
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

display_cols = [
    "node","dna_score","risk_level","stage_label",
    "fan_out_ratio","velocity_score","burst_score",
    "hops_to_cashout","cashout_probability","time_to_cashout_min",
]
top_tbl = pred_df[display_cols].head(15).copy()
top_tbl.columns = [
    "Account","DNA Score","Risk Level","Stage",
    "Fan-out","Velocity","Burst",
    "Hops","Cashout %","ETA (min)",
]
st.dataframe(
    top_tbl,
    use_container_width=True,
    height=380,
    column_config={
        "DNA Score":  st.column_config.NumberColumn(format="%.1f"),
        "Cashout %":  st.column_config.ProgressColumn(
            format="%.1f%%", min_value=0, max_value=100),
        "ETA (min)":  st.column_config.NumberColumn(format="%.0f"),
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# IMMUTABLE AUDIT ANCHOR — Blockchain Proof of Detection
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="sec-hdr"><span class="sec-dot"></span>IMMUTABLE AUDIT ANCHOR — BLOCKCHAIN PROOF OF DETECTION</div>
""", unsafe_allow_html=True)

_top_dna_score = float(pred_df["dna_score"].max()) if not pred_df.empty else 0.0

# Phase 3: always read stage from session_state (set by run_pipeline) — single source of truth
_top_stage = st.session_state.get("current_stage",
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
        st.success("✔️ Alert hash anchored to Ethereum Sepolia.", icon="⛓️")

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
                <a href='{ar["etherscan_url"]}' target='_blank'
                   style='color:#3b82f6;text-decoration:none;font-size:.7rem;'>
                    🔗 View on Sepolia Etherscan →
                </a>
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
    <div style='font-size:.65rem;color:#1e293b;'>
        🔍 ChronoTrace v2.0 · AI-Powered FinCrime Intelligence
    </div>
    <div style='font-size:.65rem;color:#1e293b;'>
        Gemini 2.5 Flash · NetworkX · Ethereum Sepolia · Streamlit
    </div>
    <div style='font-size:.65rem;color:#1e293b;'>
        © 2025 ChronoTrace Labs
    </div>
</div>
""", unsafe_allow_html=True)
