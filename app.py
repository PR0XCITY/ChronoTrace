"""
app.py — ChronoTrace | Fintech Cybersecurity Intelligence Platform
Main Streamlit application entry point.

Run with:  streamlit run app.py
"""

import json
import time
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Internal modules ─────────────────────────────────────────────────────────
import simulate as sim_engine
import dna_engine
import predictor as pred_engine
import alerts as alert_engine

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChronoTrace | FinCrime Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject enterprise-grade CSS
st.markdown("""
<style>
/* ── Base & fonts ────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #050810;
    color: #e2e8f0;
}

/* ── Remove Streamlit branding ───────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Main layout ─────────────────────────────────── */
.main .block-container {
    padding: 1rem 2rem 2rem 2rem;
    max-width: 100%;
}

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0f1a 100%);
    border-right: 1px solid #1e2a3a;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.8rem; }

/* ── KPI Cards ───────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #0f1927 0%, #0d1520 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(59,130,246,0.15);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, linear-gradient(90deg, #3b82f6, #06b6d4));
}
.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 0.3rem;
}

/* ── Section headers ─────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    border-bottom: 1px solid #1e2a3a;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.section-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #3b82f6;
    display: inline-block;
}

/* ── Alert cards ─────────────────────────────────── */
.alert-card {
    background: #0a0f1a;
    border-left: 3px solid var(--alert-color, #3b82f6);
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    transition: background 0.2s;
}
.alert-card:hover { background: #0d1520; }
.alert-meta {
    font-size: 0.65rem;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.2rem;
}
.alert-msg {
    font-size: 0.78rem;
    color: #cbd5e1;
    line-height: 1.4;
}

/* ── Stage badge ─────────────────────────────────── */
.stage-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.4);
}

/* ── Risk gauge container ────────────────────────── */
.gauge-container {
    background: #0d1117;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 1rem;
}

/* ── Intervention buttons ────────────────────────── */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    border: none;
    width: 100%;
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}

/* ── Progress bars ───────────────────────────────── */
.stProgress > div > div { background-color: #1e3a5f; }
.stProgress > div > div > div { background: linear-gradient(90deg, #3b82f6, #06b6d4); }

/* ── Dividers ────────────────────────────────────── */
hr { border-color: #1e2a3a; }

/* ── Scrollable alert panel ──────────────────────── */
.alert-scroll {
    max-height: 420px;
    overflow-y: auto;
    padding-right: 4px;
}
.alert-scroll::-webkit-scrollbar { width: 4px; }
.alert-scroll::-webkit-scrollbar-track { background: #0a0f1a; }
.alert-scroll::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }

/* ── Countdown ───────────────────────────────────── */
.countdown-box {
    background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
    border: 1px solid #7f1d1d;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}
.countdown-value {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #ef4444;
    animation: pulse 1s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.6; }
}

/* ── Metric table ────────────────────────────────── */
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1e2a3a;
    font-size: 0.78rem;
}
.metric-row:last-child { border-bottom: none; }
.metric-key { color: #64748b; }
.metric-val { color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "sim_result":      None,
        "analysis":        None,
        "pred_df":         None,
        "alerts":          [],
        "ring_summary":    {},
        "intervention":    None,
        "intervention_out":None,
        "live_step":       0,
        "streaming":       False,
        "last_run_mode":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(mode: str, n_rings: int = 1):
    """Run simulation → analysis → prediction → alerts and cache to session."""
    with st.spinner("🔄 Running intelligence pipeline…"):
        result      = sim_engine.run_simulation(mode=mode, n_rings=n_rings)
        analysis    = dna_engine.analyse(result)
        pred_df     = pred_engine.predict(analysis["dna_df"])
        all_alerts  = alert_engine.generate_all_alerts(result, analysis, pred_df)
        ring_summary= pred_engine.predict_ring_summary(pred_df, result["ring_accounts"])

    st.session_state.sim_result      = result
    st.session_state.analysis        = analysis
    st.session_state.pred_df         = pred_df
    st.session_state.alerts          = all_alerts
    st.session_state.ring_summary    = ring_summary
    st.session_state.intervention    = None
    st.session_state.intervention_out= None
    st.session_state.live_step       = 0
    st.session_state.last_run_mode   = mode


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2rem;'>🔍</div>
        <div style='font-size:1.1rem; font-weight:700; color:#e2e8f0; letter-spacing:0.05em;'>ChronoTrace</div>
        <div style='font-size:0.7rem; color:#475569; letter-spacing:0.08em; margin-top:0.2rem;'>
            FINCRIME INTELLIGENCE PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Mode selector
    mode_label = st.selectbox(
        "Simulation Mode",
        options=["Attack Simulation", "Normal (Baseline)"],
        index=0,
        help="Attack mode injects mule ring patterns into the transaction graph.",
    )
    mode = "attack" if mode_label == "Attack Simulation" else "normal"

    n_rings = 1
    if mode == "attack":
        n_rings = st.slider("Number of Mule Rings", min_value=1, max_value=3, value=1)

    st.markdown("---")

    if st.button("▶  Run Simulation", use_container_width=True):
        run_full_pipeline(mode, n_rings)

    if st.session_state.sim_result:
        st.markdown("---")

        # Export JSON
        if st.button("📥  Export Report (JSON)", use_container_width=True):
            report = {
                "generated_at": datetime.now().isoformat(),
                "mode": st.session_state.last_run_mode,
                "summary": st.session_state.analysis["summary"],
                "ring_summary": st.session_state.ring_summary,
                "top_risks": st.session_state.analysis["top_risks"].to_dict(orient="records"),
                "alerts": [
                    {k: (v.isoformat() if isinstance(v, datetime) else v)
                     for k, v in a.items()}
                    for a in st.session_state.alerts
                ],
            }
            st.download_button(
                "💾 Download JSON",
                data=json.dumps(report, indent=2),
                file_name=f"chronotrace_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.65rem; color:#334155; text-align:center; line-height:1.6;'>
        ChronoTrace v1.0.0<br>
        FinCrime Graph Intelligence<br>
        © 2025 ChronoTrace Labs
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='display:flex; align-items:center; justify-content:space-between;
            border-bottom: 1px solid #1e2a3a; padding-bottom: 1rem; margin-bottom: 1.5rem;'>
    <div>
        <div style='font-size:1.6rem; font-weight:800; letter-spacing:0.04em; color:#f1f5f9;'>
            🔍 ChronoTrace
        </div>
        <div style='font-size:0.78rem; color:#475569; letter-spacing:0.06em; margin-top:0.1rem;'>
            PREDICTING FINANCIAL CRIME BEFORE MONEY DISAPPEARS
        </div>
    </div>
    <div style='text-align:right;'>
        <div style='font-size:0.7rem; color:#334155; font-family: monospace;'>
            LIVE THREAT INTELLIGENCE
        </div>
        <div style='font-size:0.85rem; color:#3b82f6; font-weight:600; font-family:monospace;'>
            ● SYSTEM ACTIVE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# NO DATA STATE
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.sim_result is None:
    st.markdown("""
    <div style='text-align:center; padding: 5rem 2rem;'>
        <div style='font-size:4rem; margin-bottom:1rem;'>🛡️</div>
        <div style='font-size:1.4rem; font-weight:700; color:#e2e8f0; margin-bottom:0.5rem;'>
            Intelligence Engine Ready
        </div>
        <div style='font-size:0.88rem; color:#475569; max-width:480px; margin:0 auto; line-height:1.8;'>
            Select a simulation mode from the sidebar and click <strong>Run Simulation</strong> 
            to begin real-time mule ring detection and time-to-cashout prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT SESSION DATA
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
cashout_nodes= result["cashout_nodes"]
is_attack    = result["mode"] == "attack"


# ─────────────────────────────────────────────────────────────────────────────
# KPI METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5, k6 = st.columns(6)

def _kpi(col, label, value, sub, accent_css):
    col.markdown(f"""
    <div class="kpi-card" style="--accent:{accent_css}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{accent_css.split(',')[-1].strip().rstrip(')')}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

ttc = ring_summary.get("min_time_to_cashout", -1)
ttc_str = f"{ttc:.0f}m" if ttc >= 0 else "N/A"
ttc_accent = "linear-gradient(90deg,#ef4444,#dc2626)" if ttc >= 0 else "linear-gradient(90deg,#22c55e,#16a34a)"

_kpi(k1, "Max DNA Score",   f"{summary['max_dna_score']:.0f}",
     "/ 100 threat level", "linear-gradient(90deg,#ef4444,#f97316)")
_kpi(k2, "Critical Nodes",  str(summary["n_critical"]),
     "CRITICAL risk nodes", "linear-gradient(90deg,#f97316,#eab308)")
_kpi(k3, "Ring Clusters",   str(summary["n_clusters"]),
     "suspicious clusters", "linear-gradient(90deg,#a855f7,#6366f1)")
_kpi(k4, "Time to Cashout", ttc_str,
     "estimated minutes",   ttc_accent)
cashout_prob = ring_summary.get("max_cashout_probability", 0)
_kpi(k5, "Cashout Prob",    f"{cashout_prob:.0f}%",
     ring_summary.get("dominant_label", "Normal"),
     "linear-gradient(90deg,#3b82f6,#06b6d4)")
_kpi(k6, "Total Nodes",    str(summary["total_nodes"]),
     f"{summary['total_edges']} edges",
     "linear-gradient(90deg,#64748b,#475569)")

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT AREA: Graph | Alert Feed
# ─────────────────────────────────────────────────────────────────────────────

graph_col, alert_col = st.columns([3, 1.1], gap="medium")


# ── NETWORK GRAPH ─────────────────────────────────────────────────────────────

with graph_col:
    st.markdown('<div class="section-header"><span class="section-dot"></span>TRANSACTION NETWORK GRAPH</div>',
                unsafe_allow_html=True)

    layout  = analysis["layout"]
    dna_df  = analysis["dna_df"]

    # Limit displayed nodes for performance
    display_nodes = list(layout.keys())[:300]

    # Build node colour map
    node_color_map = {}
    node_size_map  = {}
    node_dna_map   = {}
    for node in display_nodes:
        row = dna_df[dna_df["node"] == node]
        if row.empty:
            dna_score  = 0
            risk_level = "LOW"
        else:
            dna_score  = float(row["dna_score"].iloc[0])
            risk_level = row["risk_level"].iloc[0]

        node_dna_map[node] = dna_score

        if node in cashout_nodes:
            node_color_map[node] = "#ef4444"    # red — cashout
            node_size_map[node]  = 18
        elif node in ring_accounts:
            if risk_level == "CRITICAL":
                node_color_map[node] = "#f97316"  # orange — critical ring
                node_size_map[node]  = 14
            else:
                node_color_map[node] = "#eab308"   # yellow — ring
                node_size_map[node]  = 11
        else:
            # Normal nodes — colour by DNA score
            alpha_val = round(max(0.15, dna_score / 100), 2)
            node_color_map[node] = f"rgba(59,130,246,{alpha_val})"
            node_size_map[node]  = 6

    # Build edge traces (sample for performance)
    edge_x, edge_y = [], []
    display_edge_set = set()
    for src, dst, data in G.edges(data=True):
        if src in layout and dst in layout:
            display_edge_set.add((src, dst))

    # Use sample of edges
    sampled_edges = random.sample(list(display_edge_set), min(500, len(display_edge_set)))

    for src, dst in sampled_edges:
        x0, y0 = layout[src]
        x1, y1 = layout[dst]
        is_ring_edge = src in ring_accounts or dst in ring_accounts
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.4, color="rgba(99,102,241,0.2)"),
        hoverinfo="none",
        showlegend=False,
    )

    # Node trace
    nx_vals, ny_vals       = [], []
    n_colors, n_sizes      = [], []
    n_text, n_hover        = [], []

    for node in display_nodes:
        x, y = layout[node]
        nx_vals.append(x)
        ny_vals.append(y)
        n_colors.append(node_color_map.get(node, "#3b82f6"))
        n_sizes.append( node_size_map.get(node, 6))
        n_text.append("")

        dna = node_dna_map.get(node, 0)
        tag = "🔴 CASHOUT" if node in cashout_nodes else \
              ("🟠 RING"   if node in ring_accounts  else "🔵 NODE")
        n_hover.append(
            f"<b>{node}</b><br>"
            f"Tag: {tag}<br>"
            f"DNA Score: {dna:.1f}<br>"
            f"Out-degree: {G.out_degree(node)}<br>"
            f"In-degree: {G.in_degree(node)}"
        )

    node_trace = go.Scatter(
        x=nx_vals, y=ny_vals,
        mode="markers",
        marker=dict(
            size=n_sizes,
            color=n_colors,
            line=dict(width=0.5, color="rgba(255,255,255,0.1)"),
        ),
        text=n_hover,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )

    # Highlight ring edges
    ring_ex, ring_ey = [], []
    for src, dst in sampled_edges:
        if src in ring_accounts and dst in ring_accounts:
            x0, y0 = layout[src]
            x1, y1 = layout[dst]
            ring_ex.extend([x0, x1, None])
            ring_ey.extend([y0, y1, None])

    ring_edge_trace = go.Scatter(
        x=ring_ex, y=ring_ey,
        mode="lines",
        line=dict(width=1.5, color="rgba(239,68,68,0.55)"),
        hoverinfo="none",
        showlegend=False,
    )

    fig_graph = go.Figure(
        data=[edge_trace, ring_edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#050810",
            plot_bgcolor="#050810",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0),
            height=460,
            hoverlabel=dict(
                bgcolor="#0d1520",
                font_size=12,
                font_family="JetBrains Mono",
                font_color="#e2e8f0",
            ),
        )
    )

    st.plotly_chart(fig_graph, use_container_width=True, config={"displayModeBar": False})

    # Legend
    st.markdown("""
    <div style='display:flex; gap:1.5rem; font-size:0.72rem; color:#475569; margin-top:-0.5rem; padding-left:0.5rem;'>
        <span>🔴 Cashout Node</span>
        <span>🟠 Critical Ring Member</span>
        <span>🟡 Ring Member</span>
        <span>🔵 Normal Node</span>
    </div>
    """, unsafe_allow_html=True)


# ── ALERT FEED ────────────────────────────────────────────────────────────────

with alert_col:
    st.markdown('<div class="section-header"><span class="section-dot"></span>LIVE ALERT FEED</div>',
                unsafe_allow_html=True)

    critical_count = sum(1 for a in all_alerts if a["severity"] == "CRITICAL")
    high_count     = sum(1 for a in all_alerts if a["severity"] == "HIGH")

    st.markdown(f"""
    <div style='display:flex; gap:0.5rem; margin-bottom:0.8rem;'>
        <div style='background:rgba(239,68,68,0.12); border:1px solid #7f1d1d;
                    border-radius:6px; padding:0.25rem 0.6rem;
                    font-size:0.7rem; color:#ef4444; font-weight:600;'>
            🔴 {critical_count} CRITICAL
        </div>
        <div style='background:rgba(249,115,22,0.12); border:1px solid #7c2d12;
                    border-radius:6px; padding:0.25rem 0.6rem;
                    font-size:0.7rem; color:#f97316; font-weight:600;'>
            🟠 {high_count} HIGH
        </div>
    </div>
    """, unsafe_allow_html=True)

    alert_html = '<div class="alert-scroll">'
    for alert in all_alerts[:30]:
        ts_str = alert["timestamp"].strftime("%H:%M:%S") \
            if isinstance(alert["timestamp"], datetime) else "—"
        alert_html += f"""
        <div class="alert-card" style="--alert-color:{alert['color']}">
            <div class="alert-meta">
                {alert['badge']} {alert['severity']} · {alert['category']} · {ts_str}
            </div>
            <div class="alert-msg">{alert['message']}</div>
        </div>"""
    alert_html += "</div>"
    st.markdown(alert_html, unsafe_allow_html=True)


st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 2: Countdown | Stage | DNA Breakdown | Risk Gauge
# ─────────────────────────────────────────────────────────────────────────────

count_col, stage_col, dna_col, gauge_col = st.columns([1, 1, 1.5, 1.5], gap="medium")

# ── Countdown ────────────────────────────────────────────────────────────────
with count_col:
    st.markdown('<div class="section-header"><span class="section-dot"></span>CASHOUT COUNTDOWN</div>',
                unsafe_allow_html=True)
    ttc = ring_summary.get("min_time_to_cashout", -1)
    if ttc >= 0:
        mins = int(ttc)
        secs = int((ttc % 1) * 60)
        st.markdown(f"""
        <div class="countdown-box">
            <div style='font-size:0.65rem; color:#7f1d1d; letter-spacing:0.1em;
                        text-transform:uppercase; margin-bottom:0.5rem;'>
                ⏱ Estimated Time to Cashout
            </div>
            <div class="countdown-value">{mins:02d}:{secs:02d}</div>
            <div style='font-size:0.65rem; color:#7f1d1d; margin-top:0.4rem;'>minutes · seconds</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="countdown-box" style='border-color:#14532d; background: linear-gradient(135deg,#0a1a0a,#0f2d1a);'>
            <div style='font-size:0.65rem; color:#166534; letter-spacing:0.1em;
                        text-transform:uppercase; margin-bottom:0.5rem;'>
                ✅ No Threat Detected
            </div>
            <div style='font-size:2.5rem; font-weight:700; font-family:monospace; color:#22c55e;'>—:—</div>
        </div>
        """, unsafe_allow_html=True)


# ── Laundering Stage ─────────────────────────────────────────────────────────
with stage_col:
    st.markdown('<div class="section-header"><span class="section-dot"></span>LAUNDERING STAGE</div>',
                unsafe_allow_html=True)
    stage_num  = ring_summary.get("max_stage", 0)
    stage_info = pred_engine.STAGE_DEFINITIONS[stage_num]

    stages_list = [
        ("0", "Normal",      "#22c55e"),
        ("1", "Compromised", "#eab308"),
        ("2", "Layering",    "#f97316"),
        ("3", "Pre-Cashout", "#ef4444"),
        ("4", "Exit Imminent","#dc2626"),
    ]
    for s_num, s_name, s_color in stages_list:
        active = int(s_num) == stage_num
        border = f"1px solid {s_color}" if active else "1px solid #1e2a3a"
        bg     = f"rgba({','.join(str(int(s_color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.12)" if active else "transparent"
        weight = "700" if active else "400"
        icon   = stage_info["icon"] if active else "○"
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:0.6rem; padding:0.35rem 0.6rem;
                    border-radius:6px; border:{border}; background:{bg}; margin-bottom:0.3rem;
                    font-size:0.78rem; font-weight:{weight}; color:{"#e2e8f0" if active else "#475569"};'>
            <span>{icon}</span>
            <span>Stage {s_num} — {s_name}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='font-size:0.7rem; color:#475569; margin-top:0.6rem; line-height:1.5;'>
        {stage_info['description']}
    </div>""", unsafe_allow_html=True)


# ── DNA Breakdown ─────────────────────────────────────────────────────────────
with dna_col:
    st.markdown('<div class="section-header"><span class="section-dot"></span>DNA BREAKDOWN — TOP RISK NODE</div>',
                unsafe_allow_html=True)

    top_node = analysis["top_risks"].iloc[0] if not analysis["top_risks"].empty else None
    if top_node is not None:
        metrics = [
            ("Fan-out Ratio",    top_node.get("fan_out_ratio", 0),   1.0),
            ("Velocity Score",   top_node.get("velocity_score", 0),  1.0),
            ("Burst Score",      top_node.get("burst_score", 0),     1.0),
            ("Circularity",      top_node.get("circularity", 0),     1.0),
            ("Amount Anomaly",   top_node.get("amount_anomaly", 0),  1.0),
            ("Hop Proximity",    top_node.get("hop_proximity", 0),   1.0),
        ]
        st.markdown(f"""
        <div style='font-size:0.72rem; color:#64748b; margin-bottom:0.6rem;
                    font-family:monospace;'>
            Node: <span style='color:#e2e8f0;'>{top_node['node']}</span>
            &nbsp;|&nbsp; DNA: <span style='color:#ef4444; font-weight:700;'>
            {top_node['dna_score']:.1f}</span>
        </div>""", unsafe_allow_html=True)

        colors = ["#ef4444","#f97316","#eab308","#3b82f6","#8b5cf6","#06b6d4"]
        for (label, val, max_val), color in zip(metrics, colors):
            pct = min(val / max_val, 1.0) * 100
            st.markdown(f"""
            <div style='margin-bottom:0.45rem;'>
                <div style='display:flex; justify-content:space-between;
                            font-size:0.7rem; margin-bottom:0.2rem;'>
                    <span style='color:#64748b;'>{label}</span>
                    <span style='color:#e2e8f0; font-family:monospace;'>{val:.3f}</span>
                </div>
                <div style='background:#1e2a3a; border-radius:3px; height:5px;'>
                    <div style='background:{color}; width:{pct:.1f}%;
                                height:5px; border-radius:3px;
                                transition:width 0.5s ease;'></div>
                </div>
            </div>""", unsafe_allow_html=True)


# ── Risk Gauge ───────────────────────────────────────────────────────────────
with gauge_col:
    st.markdown('<div class="section-header"><span class="section-dot"></span>RISK GAUGE</div>',
                unsafe_allow_html=True)

    risk_score = summary["max_dna_score"]

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        number={
            "font": {"size": 36, "family": "JetBrains Mono", "color": "#e2e8f0"},
            "suffix": "",
        },
        delta={"reference": 50, "valueformat": ".1f",
               "increasing": {"color": "#ef4444"},
               "decreasing": {"color": "#22c55e"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#334155", "tickfont": {"color": "#475569", "size": 10}},
            "bar":  {"color": "#ef4444" if risk_score >= 70 else
                              "#f97316" if risk_score >= 50 else "#22c55e",
                     "thickness": 0.25},
            "bgcolor": "#0d1117",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(34,197,94,0.08)"},
                {"range": [30, 50], "color": "rgba(234,179,8,0.08)"},
                {"range": [50, 70], "color": "rgba(249,115,22,0.08)"},
                {"range": [70,100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 3},
                "thickness": 0.75,
                "value": risk_score,
            },
        },
        title={"text": "Threat Index", "font": {"color": "#475569", "size": 12}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#050810",
        font_color="#e2e8f0",
        height=230,
        margin=dict(l=20, r=20, t=20, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # Risk label
    risk_label = ("🔴 CRITICAL THREAT" if risk_score >= 70 else
                  "🟠 HIGH RISK"       if risk_score >= 50 else
                  "🟡 MODERATE"        if risk_score >= 30 else
                  "🟢 LOW RISK")
    st.markdown(f"""
    <div style='text-align:center; font-size:0.8rem; font-weight:700;
                color:{"#ef4444" if risk_score>=70 else "#f97316" if risk_score>=50 else "#22c55e"};
                letter-spacing:0.08em;'>
        {risk_label}
    </div>""", unsafe_allow_html=True)


st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 3: Velocity Chart | Intervention Panel
# ─────────────────────────────────────────────────────────────────────────────

vel_col, int_col = st.columns([2.5, 1.5], gap="medium")


# ── Transaction Velocity Chart ────────────────────────────────────────────────
with vel_col:
    st.markdown('<div class="section-header"><span class="section-dot"></span>TRANSACTION VELOCITY TIME-SERIES</div>',
                unsafe_allow_html=True)

    tx = transactions.copy()
    tx["minute_bucket"] = tx["timestamp"].dt.floor("5min")
    velocity_df = tx.groupby(["minute_bucket", "is_suspicious"]).size().reset_index(name="count")

    normal_v = velocity_df[~velocity_df["is_suspicious"]].set_index("minute_bucket")["count"]
    ring_v   = velocity_df[ velocity_df["is_suspicious"]].set_index("minute_bucket")["count"]

    fig_vel = go.Figure()
    fig_vel.add_trace(go.Scatter(
        x=normal_v.index, y=normal_v.values,
        name="Normal Transactions",
        mode="lines",
        line=dict(color="#3b82f6", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.06)",
    ))
    if is_attack and not ring_v.empty:
        fig_vel.add_trace(go.Scatter(
            x=ring_v.index, y=ring_v.values,
            name="Ring / Suspicious",
            mode="lines+markers",
            line=dict(color="#ef4444", width=2),
            marker=dict(size=6, color="#ef4444",
                        line=dict(color="#fff", width=1)),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.10)",
        ))

    fig_vel.update_layout(
        paper_bgcolor="#050810",
        plot_bgcolor="#050810",
        font=dict(family="Inter", color="#94a3b8", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(
            showgrid=True, gridcolor="#1e2a3a",
            zeroline=False, tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1e2a3a",
            zeroline=False, tickfont=dict(size=10),
            title="Tx Count / 5 min",
        ),
        margin=dict(l=50, r=20, t=10, b=40),
        height=260,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0d1520", font_size=11),
    )
    st.plotly_chart(fig_vel, use_container_width=True, config={"displayModeBar": False})


# ── Intervention Panel ────────────────────────────────────────────────────────
with int_col:
    st.markdown('<div class="section-header"><span class="section-dot"></span>INTERVENTION SIMULATION</div>',
                unsafe_allow_html=True)

    if is_attack:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🧊 Freeze\nOrigin", use_container_width=True):
                st.session_state.intervention = "freeze_origin"
                st.session_state.intervention_out = alert_engine.compute_intervention_outcome(
                    "freeze_origin", ring_summary, transactions, ring_accounts)
        with col_b:
            if st.button("❄️ Freeze\nRing", use_container_width=True):
                st.session_state.intervention = "freeze_ring"
                st.session_state.intervention_out = alert_engine.compute_intervention_outcome(
                    "freeze_ring", ring_summary, transactions, ring_accounts)
        with col_c:
            if st.button("👁 Monitor\nOnly", use_container_width=True):
                st.session_state.intervention = "monitor_only"
                st.session_state.intervention_out = alert_engine.compute_intervention_outcome(
                    "monitor_only", ring_summary, transactions, ring_accounts)

        out = st.session_state.intervention_out
        if out:
            action_labels = {
                "freeze_ring":   ("❄️ Full Ring Freeze", "#3b82f6"),
                "freeze_origin": ("🧊 Origin Freeze",    "#f97316"),
                "monitor_only":  ("👁 Monitor Only",     "#eab308"),
            }
            a_label, a_color = action_labels.get(
                st.session_state.intervention, ("Action", "#64748b"))

            st.markdown(f"""
            <div style='background:#0a0f1a; border:1px solid #1e2a3a; border-radius:10px;
                        padding:0.9rem; margin-top:0.6rem;'>
                <div style='font-size:0.7rem; font-weight:700; color:{a_color};
                            letter-spacing:0.08em; margin-bottom:0.7rem;'>
                    {a_label}
                </div>
            """, unsafe_allow_html=True)

            metrics_out = [
                ("Total at Risk",   f"${out['total_at_risk']:,.0f}",  "#475569"),
                ("Est. Loss",       f"${out['estimated_loss']:,.0f}", "#ef4444"),
                ("Loss Prevented",  f"${out['loss_prevented']:,.0f}", "#22c55e"),
                ("Recovery %",      f"{out['recovery_pct']:.0f}%",    "#3b82f6"),
            ]
            rows_html = ""
            for k, v, c in metrics_out:
                rows_html += f"""
                <div class='metric-row'>
                    <span class='metric-key'>{k}</span>
                    <span class='metric-val' style='color:{c};'>{v}</span>
                </div>"""
            st.markdown(rows_html + "</div>", unsafe_allow_html=True)

            # Recovery progress bar
            st.progress(out["recovery_pct"] / 100)
            st.markdown(f"""
            <div style='font-size:0.7rem; color:#475569; margin-top:0.3rem;
                        font-style:italic; line-height:1.5;'>
                💡 {out['recommendation']}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#0a0f1a; border:1px solid #1e2a3a; border-radius:10px;
                    padding:1.5rem; text-align:center;'>
            <div style='font-size:1.5rem; margin-bottom:0.5rem;'>✅</div>
            <div style='font-size:0.8rem; color:#475569;'>
                No intervention required.<br>Normal transaction baseline detected.
            </div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 4: Top Risk Nodes Table
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><span class="section-dot"></span>TOP RISK NODES — DNA INTELLIGENCE TABLE</div>',
            unsafe_allow_html=True)

display_cols = [
    "node", "dna_score", "risk_level", "stage_label",
    "fan_out_ratio", "velocity_score", "burst_score",
    "hops_to_cashout", "cashout_probability", "time_to_cashout_min",
]

top_table = pred_df[display_cols].head(15).copy()
top_table.columns = [
    "Account", "DNA Score", "Risk Level", "Stage",
    "Fan-out", "Velocity", "Burst",
    "Hops", "Cashout %", "ETA (min)",
]

def _style_table(df):
    """Apply a dark-themed style to the risk table."""
    def highlight_risk(val):
        colors = {"CRITICAL": "#7f1d1d", "HIGH": "#7c2d12",
                  "MEDIUM": "#713f12", "LOW": "#14532d"}
        return f"background-color: {colors.get(val, 'transparent')}; color: #fef2f2;" \
               if val in colors else ""

    def highlight_dna(val):
        try:
            v = float(val)
            if v >= 70: return "color: #ef4444; font-weight: 700;"
            if v >= 50: return "color: #f97316;"
            if v >= 30: return "color: #eab308;"
            return "color: #22c55e;"
        except:
            return ""

    styled = df.style \
        .applymap(highlight_risk, subset=["Risk Level"]) \
        .applymap(highlight_dna, subset=["DNA Score"]) \
        .format({
            "DNA Score": "{:.1f}",
            "Fan-out":   "{:.2f}",
            "Velocity":  "{:.2f}",
            "Burst":     "{:.2f}",
            "Cashout %": "{:.1f}",
            "ETA (min)": lambda x: f"{x:.0f}" if x >= 0 else "—",
        }) \
        .set_properties(**{
            "background-color": "#0a0f1a",
            "color": "#cbd5e1",
            "border": "1px solid #1e2a3a",
            "font-size": "12px",
            "font-family": "JetBrains Mono, monospace",
        }) \
        .set_table_styles([{
            "selector": "th",
            "props": [
                ("background-color", "#0d1520"),
                ("color", "#64748b"),
                ("font-size", "11px"),
                ("letter-spacing", "0.06em"),
                ("text-transform", "uppercase"),
                ("border", "1px solid #1e2a3a"),
            ]
        }])
    return styled

st.dataframe(
    top_table,
    use_container_width=True,
    height=390,
    column_config={
        "DNA Score":   st.column_config.NumberColumn(format="%.1f"),
        "Cashout %":   st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        "ETA (min)":   st.column_config.NumberColumn(format="%.0f"),
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='border-top: 1px solid #1e2a3a; margin-top: 2rem; padding-top: 1rem;
            display: flex; justify-content: space-between; align-items: center;'>
    <div style='font-size:0.7rem; color:#334155;'>
        🔍 ChronoTrace v1.0 · FinCrime Graph Intelligence Platform
    </div>
    <div style='font-size:0.7rem; color:#334155;'>
        Powered by NetworkX · Pandas · Streamlit · Plotly
    </div>
    <div style='font-size:0.7rem; color:#334155;'>
        © 2025 ChronoTrace Labs — Predicting Financial Crime Before Money Disappears
    </div>
</div>
""", unsafe_allow_html=True)