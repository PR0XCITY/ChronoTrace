"""
predictor.py — ChronoTrace Time-to-Cashout Predictor
Uses rule-based intelligence to classify laundering stage
and estimate remaining minutes before funds exit the system.
"""

import math
import pandas as pd
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# LAUNDERING STAGE DEFINITIONS
# ────────────────────────────────────────────────────────────────────────────

STAGE_DEFINITIONS = {
    0: {
        "label": "Normal",
        "description": "No suspicious activity detected.",
        "color": "#22c55e",          # green
        "icon": "✅",
    },
    1: {
        "label": "Compromised",
        "description": "Origin account shows signs of breach. Unusual outflows detected.",
        "color": "#eab308",          # yellow
        "icon": "⚠️",
    },
    2: {
        "label": "Layering",
        "description": "Funds are being rapidly redistributed through multiple intermediaries.",
        "color": "#f97316",          # orange
        "icon": "🔄",
    },
    3: {
        "label": "Pre-Cashout",
        "description": "Funds converging. Aggregation detected near exit point.",
        "color": "#ef4444",          # red
        "icon": "🚨",
    },
    4: {
        "label": "Exit Imminent",
        "description": "Cashout transaction detected. Immediate intervention required.",
        "color": "#dc2626",          # dark red
        "icon": "💀",
    },
}

# Average minutes per hop in each laundering phase (approximate)
MINUTES_PER_HOP = {
    1: 45,   # Compromised phase — initiation takes time
    2: 5,    # Layering — rapid churn
    3: 8,    # Pre-cashout aggregation
    4: 0,    # Already at exit
}

# Typical cascade: 1 origin  → 4 mules (L1) → 5 mules (L2) → 2 aggregators → 1 cashout
STAGE_HOP_RANGES = {
    1: (6, 10),   # Far from cashout
    2: (3, 6),    # Mid-chain
    3: (1, 3),    # Near cashout
    4: (0, 1),    # At or passing through cashout
}


# ────────────────────────────────────────────────────────────────────────────
# STAGE CLASSIFIER
# ────────────────────────────────────────────────────────────────────────────

def classify_stage(row: pd.Series) -> int:
    """
    Classify a node's laundering stage using its DNA metrics.

    Thresholds tuned for real-dataset DNA ranges (typical max 35–50):
      Stage 4 (Exit Imminent) → hop_proximity ≈ 1  AND dna_score ≥ 35
      Stage 3 (Pre-Cashout)   → dna_score ≥ 28  AND burst ≥ 0.3  AND hops ≤ 4
      Stage 2 (Layering)      → dna_score ≥ 20  AND velocity ≥ 0.2
      Stage 1 (Compromised)   → dna_score ≥ 12  AND fan_out > 1
      Stage 0 (Normal)        → default
    """
    dna       = row.get("dna_score", 0)
    burst     = row.get("burst_score", 0)
    velocity  = row.get("velocity_score", 0)
    hop_prox  = row.get("hop_proximity", 0)
    hops      = row.get("hops_to_cashout", -1)
    fan_out   = row.get("fan_out_ratio", 0)

    if hop_prox >= 0.9 and dna >= 35:
        return 4
    if dna >= 28 and burst >= 0.3 and (hops != -1 and hops <= 4):
        return 3
    if dna >= 20 and velocity >= 0.2:
        return 2
    if dna >= 12 and fan_out > 1.0:
        return 1
    return 0


# ────────────────────────────────────────────────────────────────────────────
# TIME-TO-CASHOUT ESTIMATION
# ────────────────────────────────────────────────────────────────────────────

def estimate_time_to_cashout(row: pd.Series, stage: int) -> float:
    """
    Estimate remaining minutes before cashout for a node in a given stage.

    Uses hops_to_cashout and stage-specific average hop durations.
    Returns 0.0 for Stage 4 (already exiting).
    Returns -1.0 for Stage 0 (not suspicious).
    """
    if stage == 0:
        return -1.0
    if stage == 4:
        return 0.0

    hops = row.get("hops_to_cashout", -1)
    velocity = row.get("velocity_score", 0.3)

    if hops == -1:
        # Estimate based on stage position
        avg_hops = sum(STAGE_HOP_RANGES[stage]) / 2
    else:
        avg_hops = max(hops, 1)

    base_time = avg_hops * MINUTES_PER_HOP.get(stage, 10)

    # Velocity modifier: higher velocity → faster movement
    velocity_factor = max(0.3, 1.0 - velocity * 0.6)
    estimated = base_time * velocity_factor

    # Add jitter for realism
    jitter = np.random.uniform(-estimated * 0.1, estimated * 0.1)
    estimated = max(1.0, estimated + jitter)

    return round(estimated, 1)


# ────────────────────────────────────────────────────────────────────────────
# CASHOUT PROBABILITY
# ────────────────────────────────────────────────────────────────────────────

def compute_cashout_probability(row: pd.Series, stage: int) -> float:
    """
    Compute cashout probability as a percentage (0–100).

    Based on DNA score, stage, hop proximity, and burst.
    """
    base = {0: 2.0, 1: 20.0, 2: 50.0, 3: 78.0, 4: 95.0}.get(stage, 2.0)

    dna_bonus   = row.get("dna_score", 0) * 0.15
    burst_bonus = row.get("burst_score", 0) * 10
    hop_bonus   = row.get("hop_proximity", 0) * 10

    prob = base + dna_bonus + burst_bonus + hop_bonus
    return round(min(prob, 99.0), 1)


# ────────────────────────────────────────────────────────────────────────────
# FULL PREDICTION PIPELINE
# ────────────────────────────────────────────────────────────────────────────

def predict(dna_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full prediction pipeline on the DNA score DataFrame.

    Adds columns:
        stage, stage_label, stage_color, stage_icon,
        time_to_cashout_min, cashout_probability
    """
    pred_df = dna_df.copy()

    stages    = []
    labels    = []
    colors    = []
    icons     = []
    times     = []
    probs     = []

    for _, row in pred_df.iterrows():
        stage = classify_stage(row)
        info  = STAGE_DEFINITIONS[stage]
        ttc   = estimate_time_to_cashout(row, stage)
        prob  = compute_cashout_probability(row, stage)

        stages.append(stage)
        labels.append(info["label"])
        colors.append(info["color"])
        icons.append(info["icon"])
        times.append(ttc)
        probs.append(prob)

    pred_df["stage"]                = stages
    pred_df["stage_label"]          = labels
    pred_df["stage_color"]          = colors
    pred_df["stage_icon"]           = icons
    pred_df["time_to_cashout_min"]  = times
    pred_df["cashout_probability"]  = probs

    pred_df.sort_values("cashout_probability", ascending=False, inplace=True)
    pred_df.reset_index(drop=True, inplace=True)
    return pred_df


# ────────────────────────────────────────────────────────────────────────────
# RING-LEVEL AGGREGATE PREDICTION
# ────────────────────────────────────────────────────────────────────────────

def predict_ring_summary(pred_df: pd.DataFrame, ring_accounts: list) -> dict:
    """
    Summarise ring-level prediction metrics.

    Args:
        pred_df       : output of predict()
        ring_accounts : list of account IDs in the ring

    Returns a dict:
        max_stage, dominant_label, min_time_to_cashout,
        max_cashout_probability, n_critical_nodes, estimated_loss_usd
    """
    if not ring_accounts:
        return {
            "max_stage": 0,
            "dominant_label": "Normal",
            "min_time_to_cashout": -1.0,
            "max_cashout_probability": 0.0,
            "n_critical_nodes": 0,
            "estimated_loss_usd": 0.0,
        }

    ring_df = pred_df[pred_df["node"].isin(ring_accounts)]
    if ring_df.empty:
        return {
            "max_stage": 0,
            "dominant_label": "Normal",
            "min_time_to_cashout": -1.0,
            "max_cashout_probability": 0.0,
            "n_critical_nodes": 0,
            "estimated_loss_usd": 0.0,
        }

    max_stage   = int(ring_df["stage"].max())
    dominant    = STAGE_DEFINITIONS[max_stage]["label"]
    n_critical  = int((ring_df["risk_level"] == "CRITICAL").sum())

    # Min positive time-to-cashout
    positive_times = ring_df[ring_df["time_to_cashout_min"] >= 0]["time_to_cashout_min"]
    min_ttc = float(positive_times.min()) if not positive_times.empty else -1.0

    max_prob = float(ring_df["cashout_probability"].max())

    # Rough estimated loss: sum of DNA-weighted output flow
    estimated_loss = round(float(ring_df["dna_score"].sum()) * 1500, 2)  # heuristic

    return {
        "max_stage":               max_stage,
        "dominant_label":          dominant,
        "min_time_to_cashout":     round(min_ttc, 1),
        "max_cashout_probability": max_prob,
        "n_critical_nodes":        n_critical,
        "estimated_loss_usd":      estimated_loss,
    }