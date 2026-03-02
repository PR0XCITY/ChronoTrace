"""
blockchain_layer.py — ChronoTrace Immutable Audit Anchoring
=============================================================
Provides:
  hash_alert(alert_dict)         → SHA-256 hex string
  anchor_to_blockchain(hash_hex) → {"tx_hash": str, "mode": str, "etherscan_url": str}

Anchoring conditions (ALL must be true):
  - DNA score > DNA_THRESHOLD
  - laundering_stage == "Exit Imminent"
  - pqc_status == "Quantum Vulnerable"

Uses Web3 HTTPProvider with Sepolia testnet.
Falls back gracefully to deterministic mock tx hash if:
  - ETH_RPC_URL missing
  - PRIVATE_KEY missing
  - Web3 call fails for any reason

Auth priority:  os.environ → st.secrets → None (simulation mode)
No crash is possible — all errors are caught and simulated.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any

# ── Suppress Web3 noise ──────────────────────────────────────────────────────
logging.getLogger("web3").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# ── Thresholds ───────────────────────────────────────────────────────────────
DNA_THRESHOLD       = 25.0
REQUIRED_STAGE      = "Exit Imminent"
REQUIRED_PQC_STATUS = "Quantum Vulnerable"

SEPOLIA_ETHERSCAN   = "https://sepolia.etherscan.io/tx/{tx_hash}"


# ── Secret resolution ────────────────────────────────────────────────────────

def _secret(key: str) -> str | None:
    """Resolve a secret: env var first, then st.secrets, then None."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key) or None
    except Exception:
        return None


# ── Core hashing ─────────────────────────────────────────────────────────────

def hash_alert(alert_dict: dict) -> str:
    """
    Produce a deterministic SHA-256 fingerprint of an alert dict.
    Keys are sorted so insertion order doesn't affect the hash.

    Args:
        alert_dict: any JSON-serialisable dict (alert, DNA metrics, etc.)

    Returns:
        64-character lowercase hex string.
    """
    canonical = json.dumps(alert_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Mock tx hash (deterministic, no randomness) ──────────────────────────────

def _mock_tx_hash(hash_hex: str) -> str:
    """
    Produce a deterministic 66-char Ethereum-style tx hash from the alert hash.
    Reproducible for the same input — useful for demo consistency.
    """
    raw = hmac.new(b"chronotrace-mock-anchor", hash_hex.encode(), hashlib.sha256).hexdigest()
    return "0x" + raw


# ── Blockchain anchor ────────────────────────────────────────────────────────

def anchor_to_blockchain(hash_hex: str) -> dict[str, str]:
    """
    Attempt to anchor hash_hex to Ethereum Sepolia testnet as a 0-ETH tx.
    The hash is stored in the `data` field of the transaction.

    Returns a dict:
        {
            "tx_hash":       "0x...",
            "mode":          "Live Sepolia" | "Testnet Simulation Mode",
            "etherscan_url": "https://sepolia.etherscan.io/tx/0x...",
        }

    Never raises — all errors produce a simulation-mode fallback.
    """
    rpc_url   = _secret("ETH_RPC_URL")
    priv_key  = _secret("PRIVATE_KEY")
    wallet    = _secret("WALLET_ADDRESS")

    if not (rpc_url and priv_key and wallet):
        return _simulation_result(hash_hex, reason="credentials missing")

    try:
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 8}))
        if not w3.is_connected():
            return _simulation_result(hash_hex, reason="RPC not reachable")

        account  = w3.eth.account.from_key(priv_key)
        nonce    = w3.eth.get_transaction_count(account.address)
        gas_price = w3.eth.gas_price

        tx = {
            "from":     account.address,
            "to":       wallet,
            "value":    0,
            "gas":      21_500,
            "gasPrice": gas_price,
            "nonce":    nonce,
            "chainId":  11155111,   # Sepolia
            "data":     ("0x" + hash_hex.encode().hex()),
        }

        signed = w3.eth.account.sign_transaction(tx, priv_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction).hex()

        if not tx_hash.startswith("0x"):
            tx_hash = "0x" + tx_hash

        return {
            "tx_hash":       tx_hash,
            "mode":          "Live Sepolia",
            "etherscan_url": SEPOLIA_ETHERSCAN.format(tx_hash=tx_hash),
        }

    except Exception as exc:  # noqa: BLE001
        reason = str(exc)[:80]
        return _simulation_result(hash_hex, reason=reason)


def _simulation_result(hash_hex: str, reason: str = "") -> dict[str, str]:
    mock = _mock_tx_hash(hash_hex)
    return {
        "tx_hash":       mock,
        "mode":          "Testnet Simulation Mode",
        "etherscan_url": SEPOLIA_ETHERSCAN.format(tx_hash=mock),
        "_reason":       reason,   # internal only — not shown in UI
    }


# ── Anchoring eligibility gate ────────────────────────────────────────────────

def should_anchor(
    dna_score: float,
    laundering_stage: str,
    pqc_status: str,
) -> bool:
    """
    Return True only when ALL three conditions are met.
    Used by dashboard to decide whether to show the anchor button.
    """
    return (
        dna_score >= DNA_THRESHOLD
        and laundering_stage == REQUIRED_STAGE
        and pqc_status == REQUIRED_PQC_STATUS
    )


def anchor_if_eligible(
    alert_dict: dict,
    dna_score: float,
    laundering_stage: str,
    pqc_status: str,
) -> dict[str, Any] | None:
    """
    Hash the alert and anchor to blockchain if eligibility conditions pass.

    Returns:
        {
            "alert_hash":    "sha256 hex",
            "tx_hash":       "0x...",
            "mode":          "Live Sepolia" | "Testnet Simulation Mode",
            "etherscan_url": "...",
        }
        or None if conditions are not met.
    """
    if not should_anchor(dna_score, laundering_stage, pqc_status):
        return None

    alert_hash = hash_alert(alert_dict)
    anchor     = anchor_to_blockchain(alert_hash)

    return {
        "alert_hash":    alert_hash,
        "tx_hash":       anchor["tx_hash"],
        "mode":          anchor["mode"],
        "etherscan_url": anchor["etherscan_url"],
    }
