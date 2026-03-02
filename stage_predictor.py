"""
stage_predictor.py — ChronoTrace Stage Predictor Facade
Thin wrapper around predictor.py for clean module-level imports.
"""

import predictor as _pred

# Re-export stage definitions so dashboard.py can import from here 
STAGE_DEFINITIONS = _pred.STAGE_DEFINITIONS


def predict(dna_df):
    """Wrapper around predictor.predict() — adds stage/probability columns."""
    return _pred.predict(dna_df)


def predict_ring_summary(pred_df, ring_accounts: list) -> dict:
    """Wrapper around predictor.predict_ring_summary()."""
    return _pred.predict_ring_summary(pred_df, ring_accounts)
