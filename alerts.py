"""
alerts.py — ChronoTrace Alert Intelligence Feed
Generates prioritised security alerts from transaction data
and DNA/prediction analysis results.
"""

import random
from datetime import datetime, timedelta

import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# ALERT SEVERITY LEVELS
# ────────────────────────────────────────────────────────────────────────────

SEVERITY = {
    "CRITICAL": {"color": "#ef4444", "badge": "🔴", "priority": 0},
    "HIGH":     {"color": "#f97316", "badge": "🟠", "priority": 1},
    "MEDIUM":   {"color": "#eab308", "badge": "🟡", "priority": 2},
    "INFO":     {"color": "#3b82f6", "badge": "🔵", "priority": 3},
}


# ────────────────────────────────────────────────────────────────────────────
# ALERT TEMPLATES
# ────────────────────────────────────────────────────────────────────────────

def _alert(severity: str, category: str, message: str, node: str = None,
           ts: datetime = None, extra: dict = None) -> dict:
    """Factory for alert dictionaries."""
    return {
        "severity":  severity,
        "category":  category,
        "message":   message,
        "node":      node,
        "timestamp": ts or datetime.now(),
        "color":     SEVERITY[severity]["color"],
        "badge":     SEVERITY[severity]["badge"],
        **(extra or {}),
    }


# ────────────────────────────────────────────────────────────────────────────
# ALERT GENERATORS
# ────────────────────────────────────────────────────────────────────────────

def generate_velocity_alerts(transactions: pd.DataFrame, threshold: int = 8) -> list[dict]:
    """
    Detect accounts sending an abnormal number of transactions within 10 minutes.
    """
    alerts = []
    window = timedelta(minutes=10)
    tx_sorted = transactions.sort_values("timestamp")

    node_counts = {}
    for _, row in tx_sorted.iterrows():
        src = row["source"]
        ts = row["timestamp"]
        bucket = node_counts.setdefault(src, [])
        # Trim stale entries
        bucket[:] = [t for t in bucket if t >= ts - window]
        bucket.append(ts)

        if len(bucket) == threshold:  # exactly at threshold → fire once
            alerts.append(_alert(
                severity="HIGH",
                category="VELOCITY ANOMALY",
                message=(
                    f"⚡ Rapid velocity anomaly on {src} — "
                    f"{len(bucket)} transactions in <10 min window."
                ),
                node=src,
                ts=ts,
            ))

    return alerts


def generate_burst_alerts(pred_df: pd.DataFrame, burst_threshold: float = 0.60) -> list[dict]:
    """
    Alert on nodes with very high burst scores (rapid-fire outgoing transactions).
    """
    alerts = []
    burst_nodes = pred_df[pred_df["burst_score"] >= burst_threshold]

    for _, row in burst_nodes.iterrows():
        alerts.append(_alert(
            severity="CRITICAL" if row["burst_score"] >= 0.80 else "HIGH",
            category="BURST DETECTION",
            message=(
                f"🚀 Burst pattern detected on {row['node']} — "
                f"burst index {row['burst_score']:.2f}. "
                "Possible automated layering tool."
            ),
            node=row["node"],
        ))

    return alerts


def generate_ring_alerts(
    suspicious_clusters: list[list[str]],
    pred_df: pd.DataFrame,
) -> list[dict]:
    """
    Generate alerts for identified suspicious clusters (ring signatures).
    """
    alerts = []
    for i, cluster in enumerate(suspicious_clusters):
        cluster_pred = pred_df[pred_df["node"].isin(cluster)]
        max_prob = cluster_pred["cashout_probability"].max() if not cluster_pred.empty else 0
        max_stage = cluster_pred["stage"].max() if not cluster_pred.empty else 0

        stage_labels = {0: "Normal", 1: "Compromised", 2: "Layering",
                        3: "Pre-Cashout", 4: "Exit Imminent"}
        alerts.append(_alert(
            severity="CRITICAL",
            category="RING SIGNATURE",
            message=(
                f"🕸️ Ring signature emerging — Cluster #{i+1} "
                f"({len(cluster)} nodes). Stage: {stage_labels.get(max_stage, 'Unknown')}. "
                f"Cashout probability: {max_prob:.0f}%."
            ),
            extra={"cluster_id": i + 1, "cluster_size": len(cluster),
                   "cashout_prob": max_prob},
        ))

    return alerts


def generate_cashout_alerts(pred_df: pd.DataFrame, prob_threshold: float = 80.0) -> list[dict]:
    """
    Alert when any node's cashout probability exceeds the threshold.
    """
    alerts = []
    high_prob = pred_df[pred_df["cashout_probability"] >= prob_threshold]

    for _, row in high_prob.iterrows():
        ttc = row.get("time_to_cashout_min", -1)
        ttc_str = f"{ttc:.0f} min" if ttc >= 0 else "imminent"

        alerts.append(_alert(
            severity="CRITICAL" if row["cashout_probability"] >= 90 else "HIGH",
            category="CASHOUT RISK",
            message=(
                f"💸 Cashout probability {row['cashout_probability']:.0f}% on {row['node']}. "
                f"Estimated exit: {ttc_str}. Immediate review required."
            ),
            node=row["node"],
            extra={"cashout_prob": row["cashout_probability"], "ttc": ttc},
        ))

    return alerts


def generate_compromise_alerts(pred_df: pd.DataFrame) -> list[dict]:
    """
    Alert on accounts in Stage 1 (Compromised).
    """
    alerts = []
    compromised = pred_df[pred_df["stage"] == 1]

    for _, row in compromised.iterrows():
        alerts.append(_alert(
            severity="HIGH",
            category="COMPROMISE DETECTED",
            message=(
                f"🔓 Compromise detected on {row['node']} — "
                f"DNA score {row['dna_score']:.1f}. "
                "Unusual outflow pattern. Credentials may be stolen."
            ),
            node=row["node"],
        ))

    return alerts


def generate_system_alerts(sim_result: dict, analysis_result: dict) -> list[dict]:
    """
    Generate summary-level system alerts from overall simulation state.
    """
    alerts = []
    summary = analysis_result.get("summary", {})
    mode = sim_result.get("mode", "normal")

    if mode == "attack":
        alerts.append(_alert(
            severity="CRITICAL",
            category="SYSTEM ALERT",
            message=(
                f"⚡ ATTACK MODE ACTIVE — "
                f"{summary.get('n_clusters', 0)} ring cluster(s) detected. "
                f"{summary.get('n_critical', 0)} CRITICAL nodes identified."
            ),
        ))

    if summary.get("max_dna_score", 0) >= 70:
        alerts.append(_alert(
            severity="CRITICAL",
            category="DNA ALERT",
            message=(
                f"🧬 Maximum DNA threat score: {summary.get('max_dna_score', 0):.1f}/100. "
                "Network integrity compromised."
            ),
        ))

    return alerts


# ────────────────────────────────────────────────────────────────────────────
# MAIN ALERT AGGREGATOR
# ────────────────────────────────────────────────────────────────────────────

def generate_all_alerts(
    sim_result: dict,
    analysis_result: dict,
    pred_df: pd.DataFrame,
) -> list[dict]:
    """
    Run all alert generators and return a prioritised, timestamped list.

    Args:
        sim_result      : dict from simulate.run_simulation()
        analysis_result : dict from dna_engine.analyse()
        pred_df         : DataFrame from predictor.predict()

    Returns:
        List of alert dicts sorted by severity (CRITICAL first).
    """
    transactions         = sim_result["transactions"]
    suspicious_clusters  = analysis_result["suspicious_clusters"]

    all_alerts = []

    # System-level alerts first
    all_alerts.extend(generate_system_alerts(sim_result, analysis_result))

    # Ring/cluster alerts
    all_alerts.extend(generate_ring_alerts(suspicious_clusters, pred_df))

    # Cashout probability alerts
    all_alerts.extend(generate_cashout_alerts(pred_df))

    # Burst detection
    all_alerts.extend(generate_burst_alerts(pred_df))

    # Compromise detection
    all_alerts.extend(generate_compromise_alerts(pred_df))

    # Velocity anomalies (applied only to suspicious tx to avoid flood)
    if "is_suspicious" in transactions.columns:
        suspicious_tx = transactions.loc[transactions["is_suspicious"] == True]
    else:
        suspicious_tx = transactions
    all_alerts.extend(generate_velocity_alerts(suspicious_tx))

    # Sort by priority (CRITICAL = 0 sorts first)
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "INFO": 3}
    all_alerts.sort(key=lambda a: severity_order.get(a["severity"], 99))

    # Add sequential event IDs and staggered timestamps for the live feed
    base_ts = datetime.now()
    for idx, alert in enumerate(all_alerts):
        alert["event_id"] = f"EVT-{idx+1:04d}"
        if "timestamp" not in alert or alert["timestamp"] is None:
            alert["timestamp"] = base_ts - timedelta(seconds=idx * 12)

    return all_alerts


# ────────────────────────────────────────────────────────────────────────────
# INTERVENTION OUTCOMES
# ────────────────────────────────────────────────────────────────────────────

def compute_intervention_outcome(
    action: str,
    ring_summary: dict,
    transactions: pd.DataFrame,
    ring_accounts: list,
) -> dict:
    """
    Simulate the financial outcome of a chosen intervention action.

    Actions:
        "freeze_origin"  — Only the origin account is frozen
        "freeze_ring"    — Entire ring is frozen
        "monitor_only"   — No freeze, just monitoring

    Returns:
        estimated_loss    : USD lost if action taken
        loss_prevented    : USD saved vs no action
        recovery_pct      : % of funds recovered
        outcome_label     : human-readable outcome
        recommendation    : next suggested action
    """
    ring_tx = transactions[
        transactions["source"].isin(ring_accounts) |
        transactions["target"].isin(ring_accounts)
    ] if ring_accounts else transactions

    total_at_risk = float(ring_tx["amount"].sum()) if not ring_tx.empty else 500_000.0

    if action == "freeze_ring":
        loss_pct = 0.05
        label = "Full Ring Frozen — Maximum containment achieved."
        rec = "Escalate to financial crime unit. Preserve evidence for prosecution."

    elif action == "freeze_origin":
        loss_pct = 0.40
        label = "Origin Frozen — Partial containment. Layered funds still in transit."
        rec = "Monitor mule accounts. Issue secondary freeze warrants."

    else:  # monitor_only
        loss_pct = 0.85
        label = "Monitor Only — Minimal disruption. High probability of loss."
        rec = "Elevate to freeze_ring immediately before cashout window closes."

    estimated_loss   = round(total_at_risk * loss_pct, 2)
    loss_prevented   = round(total_at_risk * (1 - loss_pct), 2)
    recovery_pct     = round((1 - loss_pct) * 100, 1)

    return {
        "action":           action,
        "total_at_risk":    total_at_risk,
        "estimated_loss":   estimated_loss,
        "loss_prevented":   loss_prevented,
        "recovery_pct":     recovery_pct,
        "outcome_label":    label,
        "recommendation":   rec,
    }
