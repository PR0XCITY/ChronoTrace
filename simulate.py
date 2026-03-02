"""
simulate.py — ChronoTrace Simulation Engine
Generates a network of bank accounts and transactions.
Supports normal mode and attack (mule ring) simulation mode.
"""

import random
import string
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Reproducibility seed ────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ────────────────────────────────────────────────────────────────────────────
# ACCOUNT GENERATION
# ────────────────────────────────────────────────────────────────────────────

def _make_account_id(prefix: str, idx: int) -> str:
    """Create a formatted account ID string."""
    return f"{prefix}-{idx:04d}"


def generate_accounts(n: int = 1000) -> pd.DataFrame:
    """
    Generate n synthetic bank accounts with metadata.

    Returns a DataFrame with columns:
        account_id, account_type, initial_balance, bank_branch, country
    """
    account_types = ["personal", "business", "savings"]
    branches = ["NYC", "LON", "SIN", "DXB", "HKG", "FRA", "TOK", "SYD"]
    countries = ["US", "UK", "SG", "AE", "HK", "DE", "JP", "AU"]

    records = []
    for i in range(n):
        acc_type = random.choices(account_types, weights=[0.6, 0.3, 0.1])[0]
        branch_idx = random.randint(0, len(branches) - 1)
        balance = round(random.lognormvariate(10, 1.5), 2)  # realistic distribution
        records.append({
            "account_id": _make_account_id("ACC", i),
            "account_type": acc_type,
            "initial_balance": balance,
            "bank_branch": branches[branch_idx],
            "country": countries[branch_idx],
        })

    return pd.DataFrame(records)


# ────────────────────────────────────────────────────────────────────────────
# NORMAL TRANSACTION GENERATION
# ────────────────────────────────────────────────────────────────────────────

def generate_normal_transactions(
    accounts: pd.DataFrame,
    n_transactions: int = 3000,
    base_time: datetime = None,
) -> pd.DataFrame:
    """
    Generate realistic normal (benign) transactions between accounts.

    Normal transactions have:
    - Random sender/receiver pairs
    - Low-to-moderate amounts (log-normally distributed)
    - Spread over 24 hours with natural time gaps
    """
    if base_time is None:
        base_time = datetime(2025, 1, 1, 0, 0, 0)

    account_ids = accounts["account_id"].tolist()
    records = []

    for i in range(n_transactions):
        src, dst = random.sample(account_ids, 2)
        amount = round(random.lognormvariate(6, 1.2), 2)  # ~$400 median
        ts = base_time + timedelta(minutes=random.randint(0, 1440))
        records.append({
            "tx_id": f"TX-NORM-{i:05d}",
            "source": src,
            "target": dst,
            "amount": amount,
            "timestamp": ts,
            "tx_type": "normal",
            "is_suspicious": False,
            "ring_id": None,
        })

    df = pd.DataFrame(records)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ────────────────────────────────────────────────────────────────────────────
# MULE RING INJECTION
# ────────────────────────────────────────────────────────────────────────────

def inject_mule_ring(
    normal_transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    ring_size: int = 12,
    n_rings: int = 1,
    base_time: datetime = None,
    ring_prefix: str = "RING",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Inject one or more mule ring patterns into the transaction dataset.

    A mule ring follows this attack pattern:
      1. Compromised Origin — a single "breach" account initiates the chain
      2. Layer 1 Fan-out — rapid scatter to 3–5 mule accounts (velocity: 1–3 min)
      3. Layer 2 Deeper mixing — each mule re-distributes (velocity: 2–5 min)
      4. Pre-cashout aggregation — funds converge to 1–2 exit-adjacent accounts
      5. Cashout node — the final withdrawal destination

    Returns:
        combined_df  : all transactions (normal + ring)
        ring_accounts: list of account IDs involved in ring(s)
        cashout_nodes: list of cashout account IDs
    """
    if base_time is None:
        base_time = datetime(2025, 1, 1, 8, 0, 0)  # ring starts at 8 AM

    ring_accounts_all = []
    cashout_nodes_all = []
    ring_transactions = []
    tx_counter = 0

    all_accounts = accounts["account_id"].tolist()

    for ring_num in range(n_rings):
        # Sample unique accounts for this ring
        ring_pool = random.sample(all_accounts, ring_size + 5)

        compromised_origin = ring_pool[0]
        layer1_mules = ring_pool[1:5]       # 4 mules
        layer2_mules = ring_pool[5:10]      # 5 mules
        pre_cashout = ring_pool[10:12]      # 2 pre-cashout accounts
        cashout_node = ring_pool[12]        # 1 final cashout

        ring_accs = [compromised_origin] + layer1_mules + layer2_mules + pre_cashout + [cashout_node]
        ring_accounts_all.extend(ring_accs)
        cashout_nodes_all.append(cashout_node)

        ring_id = f"{ring_prefix}-{ring_num + 1:02d}"
        t = base_time + timedelta(minutes=ring_num * 30)  # stagger rings

        origin_amount = round(random.uniform(50000, 200000), 2)

        # ── Stage 1: Compromised origin → Layer 1 mules ────────────────────
        for i, mule in enumerate(layer1_mules):
            t += timedelta(minutes=random.uniform(1, 3))
            split_amount = round(origin_amount * random.uniform(0.18, 0.28), 2)
            ring_transactions.append({
                "tx_id": f"TX-{ring_id}-{tx_counter:04d}",
                "source": compromised_origin,
                "target": mule,
                "amount": split_amount,
                "timestamp": t,
                "tx_type": "layering",
                "is_suspicious": True,
                "ring_id": ring_id,
            })
            tx_counter += 1

        # ── Stage 2: Layer 1 → Layer 2 (deeper mixing) ─────────────────────
        for i, mule1 in enumerate(layer1_mules):
            receivers = random.sample(layer2_mules, random.randint(1, 3))
            for mule2 in receivers:
                t += timedelta(minutes=random.uniform(2, 5))
                amount = round(random.uniform(5000, 40000), 2)
                ring_transactions.append({
                    "tx_id": f"TX-{ring_id}-{tx_counter:04d}",
                    "source": mule1,
                    "target": mule2,
                    "amount": amount,
                    "timestamp": t,
                    "tx_type": "layering",
                    "is_suspicious": True,
                    "ring_id": ring_id,
                })
                tx_counter += 1

        # ── Stage 3: Layer 2 → Pre-cashout aggregation ──────────────────────
        for mule2 in layer2_mules:
            aggregator = random.choice(pre_cashout)
            t += timedelta(minutes=random.uniform(3, 8))
            amount = round(random.uniform(10000, 60000), 2)
            ring_transactions.append({
                "tx_id": f"TX-{ring_id}-{tx_counter:04d}",
                "source": mule2,
                "target": aggregator,
                "amount": amount,
                "timestamp": t,
                "tx_type": "aggregation",
                "is_suspicious": True,
                "ring_id": ring_id,
            })
            tx_counter += 1

        # ── Stage 4: Pre-cashout → Final cashout ────────────────────────────
        for aggregator in pre_cashout:
            t += timedelta(minutes=random.uniform(1, 4))
            cashout_amount = round(random.uniform(30000, 100000), 2)
            ring_transactions.append({
                "tx_id": f"TX-{ring_id}-{tx_counter:04d}",
                "source": aggregator,
                "target": cashout_node,
                "amount": cashout_amount,
                "timestamp": t,
                "tx_type": "cashout",
                "is_suspicious": True,
                "ring_id": ring_id,
            })
            tx_counter += 1

    ring_df = pd.DataFrame(ring_transactions)
    combined = pd.concat([normal_transactions, ring_df], ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    return combined, list(set(ring_accounts_all)), list(set(cashout_nodes_all))


# ────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL SIMULATION ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────

def run_simulation(
    mode: str = "attack",
    n_accounts: int = 1000,
    n_normal_tx: int = 3000,
    n_rings: int = 1,
    ring_size: int = 12,
) -> dict:
    """
    Run the full simulation and return a results dictionary.

    Args:
        mode        : "normal" or "attack"
        n_accounts  : number of synthetic accounts to generate
        n_normal_tx : number of normal background transactions
        n_rings     : number of mule rings to inject (attack mode only)
        ring_size   : number of accounts per ring

    Returns a dict with keys:
        accounts         : pd.DataFrame
        transactions     : pd.DataFrame
        ring_accounts    : list[str]
        cashout_nodes    : list[str]
        mode             : str
    """
    accounts = generate_accounts(n_accounts)
    base_time = datetime(2025, 1, 1, 0, 0, 0)

    normal_tx = generate_normal_transactions(accounts, n_normal_tx, base_time)

    if mode == "attack":
        transactions, ring_accounts, cashout_nodes = inject_mule_ring(
            normal_tx, accounts, ring_size=ring_size, n_rings=n_rings, base_time=base_time
        )
    else:
        transactions = normal_tx
        ring_accounts = []
        cashout_nodes = []

    return {
        "accounts": accounts,
        "transactions": transactions,
        "ring_accounts": ring_accounts,
        "cashout_nodes": cashout_nodes,
        "mode": mode,
    }