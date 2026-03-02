"""
graph_engine.py — ChronoTrace Graph Engine Facade
Reads transaction data from SQLite and delegates to dna_engine for analysis.
Falls back to in-memory data if database is empty.
"""

import pandas as pd
import database as db
import dna_engine


def analyse_from_db(sim_result: dict) -> dict:
    """
    Primary entry point for graph analysis using SQLite-backed data.

    Strategy:
      1. Try to load transactions from SQLite.
      2. If empty, fall back to sim_result["transactions"] (in-memory).
      3. Run dna_engine.analyse() on the resolved transaction set.

    Returns the same dict format as dna_engine.analyse().
    """
    # Attempt to load from SQLite
    tx_df = db.get_transactions(limit=10_000)

    if tx_df.empty:
        # Fallback to in-memory (e.g., first run before persist completes)
        tx_df = sim_result["transactions"]
        source_col, target_col = "source", "target"
    else:
        # SQLite uses 'sender'/'receiver' column names
        tx_df = tx_df.rename(columns={"sender": "source", "receiver": "target"})
        # Restore is_suspicious flag for ring rows
        tx_df["is_suspicious"] = tx_df["tx_type"] != "normal"
        # Parse timestamps back to datetime
        tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"], errors="coerce")

    # Delegate to dna_engine with the resolved transaction set
    merged_result = dict(sim_result)
    merged_result["transactions"] = tx_df

    return dna_engine.analyse(merged_result)


def get_graph_summary() -> dict:
    """
    Return summary statistics directly from the database.
    Useful for lightweight KPI refreshes without re-running analysis.
    """
    return db.get_summary_stats()
