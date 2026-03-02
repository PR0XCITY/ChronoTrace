"""
dataset_loader.py — ChronoTrace
================================
Loads transactions.csv, builds a NetworkX directed graph, and
serves as the single source-of-truth for the dataset-backed pipeline.

All callers should go through get_dataset() / get_graph() which
cache results in st.session_state on the first call.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import streamlit as st

# Path is relative to this file's directory so it works on any machine / Streamlit Cloud
_CSV_PATH = Path(__file__).parent / "transactions.csv"

# Required columns and their dtypes
_REQUIRED_COLS = {"sender", "receiver", "amount", "timestamp", "label"}

# ── Session-state keys ───────────────────────────────────────────────────────
_KEY_DF    = "_ctl_dataset_df"
_KEY_GRAPH = "_ctl_dataset_graph"
_KEY_FRAUDS = "_ctl_fraud_accounts"


# ── Internal loaders ────────────────────────────────────────────────────────

_MAX_ACCOUNTS = 200   # hard ceiling for rendering performance


def _prune_to_top_accounts(df: pd.DataFrame, max_accounts: int) -> tuple[pd.DataFrame, bool]:
    """
    If unique accounts exceed max_accounts, keep only the top-N by
    combined transaction volume (sum of amounts sent + received).

    Fraud accounts are always preserved regardless of volume rank.
    Only rows where BOTH sender AND receiver are in the selected set
    are kept — preserving graph edge consistency.

    Returns (filtered_df, was_pruned: bool).
    """
    all_accounts = pd.concat([df["sender"], df["receiver"]]).unique()
    if len(all_accounts) <= max_accounts:
        return df, False

    # Volume per account (sender + receiver sides)
    vol_sent = df.groupby("sender")["amount"].sum().rename("vol")
    vol_recv = df.groupby("receiver")["amount"].sum().rename("vol")
    combined = (
        pd.concat([vol_sent, vol_recv])
        .groupby(level=0).sum()
        .sort_values(ascending=False)
    )

    # Always keep fraud accounts
    fraud_set = set(df.loc[df["label"] == "fraud", "sender"]) | \
                set(df.loc[df["label"] == "fraud", "receiver"])

    top_by_vol = set(combined.head(max_accounts).index)
    selected  = fraud_set | top_by_vol        # union, fraud always included
    # Trim to exactly max_accounts (fraud accounts already guaranteed in)
    if len(selected) > max_accounts:
        non_fraud_top = [a for a in combined.index if a not in fraud_set]
        selected = fraud_set | set(non_fraud_top[:max(0, max_accounts - len(fraud_set))])

    # Filter: keep only edges where both endpoints are in selected
    mask = df["sender"].isin(selected) & df["receiver"].isin(selected)
    filtered = df[mask].copy().reset_index(drop=True)
    return filtered, True


def _load_csv() -> pd.DataFrame:
    """
    Read transactions.csv, validate columns, parse types.
    Prunes to _MAX_ACCOUNTS top accounts if dataset is larger.
    Raises FileNotFoundError / ValueError if the CSV is missing or malformed.
    """
    if not _CSV_PATH.exists():
        raise FileNotFoundError(
            f"transactions.csv not found at {_CSV_PATH}. "
            "Please include it in the project root."
        )

    df = pd.read_csv(_CSV_PATH)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"transactions.csv is missing required columns: {missing}. "
            f"Expected: {_REQUIRED_COLS}"
        )

    # ── Type coercion (strict, no silent NaN propagation) ────────────────────
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop rows where coercion failed
    bad = df["amount"].isna() | df["timestamp"].isna()
    if bad.any():
        df = df[~bad].copy()

    df["label"] = df["label"].str.strip().str.lower()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Phase 2: prune to MAX_ACCOUNTS if needed
    df, pruned = _prune_to_top_accounts(df, _MAX_ACCOUNTS)
    # Store pruning flag for UI banner
    try:
        import streamlit as _st
        _st.session_state["_ctl_dataset_pruned"] = pruned
    except Exception:
        pass

    return df



def _build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed weighted graph from transaction rows.
    Edge attribute `weight` = transaction amount (raw float).
    Node attribute `fraud_seed` = True if the node appears in any fraud row.
    """
    G = nx.DiGraph()

    # Pre-compute fraud accounts from label column
    fraud_senders   = set(df.loc[df["label"] == "fraud", "sender"])
    fraud_receivers = set(df.loc[df["label"] == "fraud", "receiver"])
    fraud_accounts  = fraud_senders | fraud_receivers

    for _, row in df.iterrows():
        src, dst, amt = row["sender"], row["receiver"], float(row["amount"])
        G.add_edge(
            src, dst,
            weight=amt,
            timestamp=row["timestamp"],
            label=row["label"],
        )

    # Tag fraud-seed nodes
    for node in G.nodes:
        G.nodes[node]["fraud_seed"] = node in fraud_accounts

    return G


# ── Public API ───────────────────────────────────────────────────────────────

def get_dataset() -> pd.DataFrame:
    """
    Return the transaction DataFrame, loading and caching on first call.
    Cached in st.session_state[_KEY_DF] to avoid re-read on every Streamlit rerun.
    """
    if _KEY_DF not in st.session_state or st.session_state[_KEY_DF] is None:
        df = _load_csv()
        st.session_state[_KEY_DF] = df
    return st.session_state[_KEY_DF]


def get_graph() -> nx.DiGraph:
    """
    Return the NetworkX DiGraph, building and caching on first call.
    Cached in st.session_state[_KEY_GRAPH].
    """
    if _KEY_GRAPH not in st.session_state or st.session_state[_KEY_GRAPH] is None:
        df = get_dataset()
        G  = _build_graph(df)
        st.session_state[_KEY_GRAPH] = G
    return st.session_state[_KEY_GRAPH]


def get_fraud_accounts() -> set[str]:
    """Return the set of account IDs that appear in at least one fraud row."""
    if _KEY_FRAUDS not in st.session_state or st.session_state[_KEY_FRAUDS] is None:
        df = get_dataset()
        fraud_df = df[df["label"] == "fraud"]
        st.session_state[_KEY_FRAUDS] = (
            set(fraud_df["sender"]) | set(fraud_df["receiver"])
        )
    return st.session_state[_KEY_FRAUDS]


def invalidate_cache() -> None:
    """Force a full reload on next call (e.g. after CSV hot-swap in tests)."""
    for key in (_KEY_DF, _KEY_GRAPH, _KEY_FRAUDS):
        st.session_state.pop(key, None)


def dataset_summary() -> dict[str, Any]:
    """Return lightweight KPI dict without full graph traversal."""
    df = get_dataset()
    fraud_df = df[df["label"] == "fraud"]
    return {
        "total_transactions": len(df),
        "fraud_transactions": len(fraud_df),
        "normal_transactions": len(df) - len(fraud_df),
        "unique_accounts": int(pd.concat([df["sender"], df["receiver"]]).nunique()),
        "fraud_accounts": len(get_fraud_accounts()),
        "total_volume_usd": float(df["amount"].sum()),
        "fraud_volume_usd": float(fraud_df["amount"].sum()),
        "date_range_start": df["timestamp"].min().isoformat(),
        "date_range_end":   df["timestamp"].max().isoformat(),
    }


def _derive_cashout_nodes(df: pd.DataFrame, fraud_accounts: set) -> list:
    """
    Identify cashout nodes from the dataset using graph topology.

    A cashout node is a fraud-labelled *receiver* that has no
    outgoing fraud transactions — i.e. it is a terminal sink of
    the suspicious money flow.  This mirrors what the simulator
    explicitly tags as 'cashout'.

    Fallback: if topology yields nothing, return all fraud receivers
    so dna_engine always gets a non-empty list.
    """
    fraud_df = df[df["label"] == "fraud"]

    # Receivers in fraud transactions
    fraud_receivers = set(fraud_df["receiver"])

    # Senders in fraud transactions
    fraud_senders = set(fraud_df["sender"])

    # Terminal sinks = received money but never sent in a fraud tx
    sinks = fraud_receivers - fraud_senders

    if sinks:
        return sorted(sinks)

    # Fallback: return all fraud receivers (degenerate/circular graph)
    return sorted(fraud_receivers) if fraud_receivers else []


def build_sim_result_from_dataset() -> dict:
    """
    Produce a sim_result dict fully compatible with dna_engine.analyse().

    Required keys (same as simulate.run_simulation()):
        transactions  : DataFrame with columns source, target, amount,
                        timestamp, tx_type, is_suspicious, tx_id
        ring_accounts : list[str]
        cashout_nodes : list[str]   ← derived from graph topology
        accounts      : list[str]   ← all unique account IDs
        mode          : str
    """
    df = get_dataset()
    fraud_accounts = get_fraud_accounts()

    # ── Build transaction DataFrame in dna_engine's expected format ──────────
    tx_df = df.rename(columns={"sender": "source", "receiver": "target"}).copy()
    tx_df["is_suspicious"] = tx_df["label"] == "fraud"
    tx_df["tx_type"] = tx_df["label"].map({"fraud": "layering", "normal": "normal"})
    # Add a synthetic tx_id so dna_engine.build_graph() never sees a missing key
    tx_df["tx_id"] = ["TX-DS-" + str(i).zfill(6) for i in range(len(tx_df))]

    # ── Derived cashout nodes (key that dna_engine.analyse() reads directly) ─
    cashout_nodes = _derive_cashout_nodes(df, fraud_accounts)

    return {
        "mode":          "dataset",
        "transactions":  tx_df,
        "ring_accounts": sorted(fraud_accounts),
        "cashout_nodes": cashout_nodes,          # ← fixes the KeyError
        "accounts":      sorted(
            pd.concat([df["sender"], df["receiver"]]).unique()
        ),
    }
