"""
database.py — ChronoTrace SQLite Persistence Layer
Handles all read/write operations for accounts, transactions, and DNA metrics.
Uses sqlite3 (stdlib) — no external DB dependencies.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional

import pandas as pd

# ── Database file path — works locally and on Streamlit Cloud ────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "chronotrace.db")


# ────────────────────────────────────────────────────────────────────────────
# CONNECTION HELPER
# ────────────────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """Return a sqlite3 connection with row_factory set to dict-like rows."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ────────────────────────────────────────────────────────────────────────────
# SCHEMA CREATION
# ────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Initialise the database schema.
    Creates all tables if they do not already exist.
    Safe to call multiple times — uses IF NOT EXISTS guards.
    """
    conn = _connect()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            id          TEXT PRIMARY KEY,
            compromised INTEGER DEFAULT 0,
            risk_score  REAL    DEFAULT 0.0,
            stage       TEXT    DEFAULT 'Normal'
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            sender    TEXT,
            receiver  TEXT,
            amount    REAL,
            timestamp TEXT,
            tx_type   TEXT    DEFAULT 'normal',
            ring_id   TEXT
        );

        CREATE TABLE IF NOT EXISTS dna_metrics (
            account_id   TEXT,
            hop_count    INTEGER,
            fan_out      REAL,
            burst_score  REAL,
            avg_hop_time REAL,
            dna_score    REAL,
            PRIMARY KEY (account_id)
        );
    """)

    conn.commit()
    conn.close()


# ────────────────────────────────────────────────────────────────────────────
# WRITE OPERATIONS
# ────────────────────────────────────────────────────────────────────────────

def clear_all() -> None:
    """Truncate all tables — called before each new simulation run."""
    conn = _connect()
    conn.execute("DELETE FROM accounts")
    conn.execute("DELETE FROM transactions")
    conn.execute("DELETE FROM dna_metrics")
    conn.commit()
    conn.close()


def save_accounts(accounts_df: pd.DataFrame, ring_accounts: list, pred_df: pd.DataFrame) -> None:
    """
    Persist account records to SQLite.

    Args:
        accounts_df  : raw accounts DataFrame from simulate.generate_accounts()
        ring_accounts: list of account IDs involved in a mule ring
        pred_df      : predictor output (for risk_score + stage)
    """
    conn = _connect()

    # Build a lookup from predictor for fast access
    pred_lookup = {}
    if pred_df is not None and not pred_df.empty:
        for _, row in pred_df.iterrows():
            pred_lookup[row["node"]] = {
                "risk_score": float(row.get("dna_score", 0.0)),
                "stage":      str(row.get("stage_label", "Normal")),
            }

    rows = []
    for _, row in accounts_df.iterrows():
        acc_id = row["account_id"]
        info   = pred_lookup.get(acc_id, {"risk_score": 0.0, "stage": "Normal"})
        rows.append((
            acc_id,
            1 if acc_id in ring_accounts else 0,
            info["risk_score"],
            info["stage"],
        ))

    conn.executemany(
        "INSERT OR REPLACE INTO accounts (id, compromised, risk_score, stage) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def save_transactions(transactions_df: pd.DataFrame) -> None:
    """
    Persist transaction records to SQLite.
    Timestamps are stored as ISO strings.
    """
    conn = _connect()

    rows = []
    for _, row in transactions_df.iterrows():
        ts = row["timestamp"]
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        rows.append((
            str(row.get("source", "")),
            str(row.get("target", "")),
            float(row.get("amount", 0.0)),
            ts_str,
            str(row.get("tx_type", "normal")),
            str(row.get("ring_id", "")) if row.get("ring_id") else None,
        ))

    conn.executemany(
        """INSERT INTO transactions (sender, receiver, amount, timestamp, tx_type, ring_id)
           VALUES (?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    conn.close()


def save_dna_metrics(dna_df: pd.DataFrame) -> None:
    """
    Persist DNA metric scores for each account.
    Uses INSERT OR REPLACE so re-runs are idempotent.
    """
    conn = _connect()

    rows = []
    for _, row in dna_df.iterrows():
        hops = row.get("hops_to_cashout", -1)
        # avg_hop_time is approximated from burst_score inverse
        burst = float(row.get("burst_score", 0))
        avg_hop_time = round((1.0 - burst) * 30, 2)  # heuristic minutes

        rows.append((
            str(row["node"]),
            int(hops) if hops != -1 else 0,
            float(row.get("fan_out_ratio", 0)),
            burst,
            avg_hop_time,
            float(row.get("dna_score", 0)),
        ))

    conn.executemany(
        """INSERT OR REPLACE INTO dna_metrics
           (account_id, hop_count, fan_out, burst_score, avg_hop_time, dna_score)
           VALUES (?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    conn.close()


# ────────────────────────────────────────────────────────────────────────────
# READ OPERATIONS
# ────────────────────────────────────────────────────────────────────────────

def get_accounts(
    risk_filter: Optional[str] = None,
    compromised_only: bool = False,
) -> pd.DataFrame:
    """
    Retrieve accounts from SQLite.

    Args:
        risk_filter      : if set, filter by stage value (e.g. "Layering")
        compromised_only : if True, return only ring-flagged accounts

    Returns a DataFrame with columns: id, compromised, risk_score, stage
    """
    conn = _connect()
    query = "SELECT * FROM accounts"
    conditions = []
    params = []

    if compromised_only:
        conditions.append("compromised = 1")
    if risk_filter and risk_filter != "All":
        conditions.append("stage = ?")
        params.append(risk_filter)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY risk_score DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_transactions(limit: int = 500, suspicious_only: bool = False) -> pd.DataFrame:
    """
    Retrieve transactions from SQLite.

    Args:
        limit           : max rows to return
        suspicious_only : if True, only return ring/cashout transactions
    """
    conn = _connect()
    if suspicious_only:
        query = """SELECT * FROM transactions
                   WHERE tx_type != 'normal'
                   ORDER BY timestamp DESC LIMIT ?"""
    else:
        query = "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT ?"

    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def get_dna_metrics(top_n: int = 20) -> pd.DataFrame:
    """Retrieve top-N DNA metrics rows sorted by dna_score descending."""
    conn = _connect()
    df = pd.read_sql_query(
        "SELECT * FROM dna_metrics ORDER BY dna_score DESC LIMIT ?",
        conn,
        params=(top_n,),
    )
    conn.close()
    return df


def get_summary_stats() -> dict:
    """Return aggregate statistics for the KPI bar."""
    conn = _connect()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM accounts")
    n_accounts = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM accounts WHERE compromised = 1")
    n_compromised = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM transactions")
    n_transactions = cur.fetchone()[0]

    cur.execute("SELECT MAX(dna_score) FROM dna_metrics")
    max_dna = cur.fetchone()[0] or 0.0

    cur.execute("SELECT SUM(amount) FROM transactions WHERE tx_type != 'normal'")
    suspicious_volume = cur.fetchone()[0] or 0.0

    conn.close()
    return {
        "n_accounts":        n_accounts,
        "n_compromised":     n_compromised,
        "n_transactions":    n_transactions,
        "max_dna_score":     round(float(max_dna), 2),
        "suspicious_volume": round(float(suspicious_volume), 2),
    }
