"""
simulator.py — ChronoTrace Simulation Facade
Wraps simulate.py and persists results to SQLite via database.py.
"""

from datetime import datetime

import simulate as _sim
import database as db


def run_and_persist(
    mode: str = "attack",
    n_accounts: int = 1000,
    n_normal_tx: int = 3000,
    n_rings: int = 1,
    ring_size: int = 12,
) -> dict:
    """
    Run the simulation pipeline and persist all results to SQLite.

    Returns the same dict as simulate.run_simulation() so all existing
    downstream code continues to work unchanged.
    """
    # 1. Run simulation (in-memory)
    result = _sim.run_simulation(
        mode=mode,
        n_accounts=n_accounts,
        n_normal_tx=n_normal_tx,
        n_rings=n_rings,
        ring_size=ring_size,
    )

    # 2. Reset database for fresh run
    db.init_db()
    db.clear_all()

    # 3. Persist transactions immediately (accounts need predictor output first)
    db.save_transactions(result["transactions"])

    # Store run metadata on result dict for downstream use
    result["run_timestamp"] = datetime.now()

    return result


def persist_predictions(result: dict, pred_df, dna_df) -> None:
    """
    Called after predictions are computed — persists account & DNA data.
    This is a separate step because pred_df depends on dna_df which
    depends on the simulation result.

    Args:
        result  : dict from run_and_persist()
        pred_df : output of predictor.predict()
        dna_df  : output of dna_engine compute_dna_scores()
    """
    db.save_accounts(
        accounts_df=result["accounts"],
        ring_accounts=result["ring_accounts"],
        pred_df=pred_df,
    )
    db.save_dna_metrics(dna_df)
