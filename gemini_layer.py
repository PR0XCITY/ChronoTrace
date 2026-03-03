"""
gemini_layer.py — ChronoTrace AI Intelligence Layer
Integrates Google Gemini via google-generativeai SDK.

Authentication priority:
  1. Environment variable GEMINI_API_KEY
  2. st.secrets["GEMINI_API_KEY"]
  3. Graceful fallback (returns placeholder data if key unavailable)
"""

import json
import os
import re
from typing import Optional

# ── Lazy import Streamlit secrets so this module works outside Streamlit too ─
def _get_api_key() -> Optional[str]:
    """Resolve API key from env → st.secrets → None."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


from google import genai

_client = None

def _get_client():
    global _client
    if _client:
        return _client

    api_key = _get_api_key()
    if not api_key:
        return None

    try:
        _client = genai.Client(api_key=api_key)
        return _client
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────────
# HELPER — safe JSON extraction from model response
# ────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Attempt to extract a JSON object from the model's text response.
    Falls back to a partial parse or empty dict on failure.
    """
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find a JSON block (```json ... ```)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


# ────────────────────────────────────────────────────────────────────────────
# A) GENERATE_INTELLIGENCE
# ────────────────────────────────────────────────────────────────────────────

_INTELLIGENCE_FALLBACK = {
    "laundering_stage":   "API key not configured",
    "risk_reasoning":     "Configure GEMINI_API_KEY to enable AI analysis.",
    "recommended_action": "Set GEMINI_API_KEY in environment or st.secrets.",
    "confidence_level":   "N/A",
}


def generate_intelligence(metrics_json: dict) -> dict:
    """
    Send structured transaction metrics to Gemini and receive AML intelligence.

    Args:
        metrics_json: dict with keys:
            hop_count, fan_out, burst_score, avg_hop_time,
            predicted_exit_minutes, compromised_status

    Returns:
        dict with keys:
            laundering_stage, risk_reasoning, recommended_action, confidence_level
    """
    model = _get_client()
    if model is None:
        return _INTELLIGENCE_FALLBACK.copy()

    prompt = f"""You are an expert Anti-Money Laundering (AML) analyst at a tier-1 financial institution.

The following transaction network metrics were computed by ChronoTrace's deterministic graph intelligence engine:

```json
{json.dumps(metrics_json, indent=2)}
```

Metric definitions:
- hop_count: number of hops from origin to nearest cashout node (-1 = unreachable)
- fan_out: out-degree / in-degree ratio (high values = rapid redistribution)
- burst_score: velocity anomaly index 0–1 (>0.6 = suspicious rapid-fire transfers)
- avg_hop_time: average minutes between hops in this sub-network
- predicted_exit_minutes: estimated minutes before funds leave the monitored network
- compromised_status: whether the origin account is flagged as breached
- deterministic_stage: laundering stage already classified by the rule engine — use this EXACTLY
- eta_minutes: same as predicted_exit_minutes, provided for clarity

⚠️ CRITICAL INSTRUCTION:
The field `deterministic_stage` has been computed by a deterministic rule engine and is the authoritative
classification. You MUST use this exact value in the `laundering_stage` field of your response.
Do NOT override, reinterpret, or substitute it with your own classification.
Your role is to provide reasoning and recommended action, not to reclassify the stage.

Based on the metrics above, respond with a JSON object containing exactly these four fields:

{{
  "laundering_stage": "{metrics_json.get('deterministic_stage', 'Normal')}",
  "risk_reasoning": "<2–3 sentence analysis explaining WHY this stage was assigned and key risk indicators observed in the metrics>",
  "recommended_action": "<concrete, specific action for the compliance team to take RIGHT NOW>",
  "confidence_level": "<one of: Low | Moderate | High | Critical>"
}}

Respond with ONLY the JSON object. No preamble, no explanation outside the JSON."""

    try:
        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        result = _extract_json(response.text)

        # Validate required keys
        required = {"laundering_stage", "risk_reasoning", "recommended_action", "confidence_level"}
        if not required.issubset(result.keys()):
            for key in required:
                if key not in result:
                    result[key] = "Analysis incomplete"

        # ── Enforce single source of truth ────────────────────────────────────
        # stage_predictor is authoritative; Gemini must not override it.
        deterministic = metrics_json.get("deterministic_stage")
        if deterministic:
            result["laundering_stage"] = deterministic

        return result

    except Exception as e:
        err_str = str(e)
        # 429 / quota exhausted
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            return {
                "laundering_stage":   metrics_json.get("deterministic_stage", "N/A"),
                "risk_reasoning":     "Gemini API quota temporarily exceeded (429). "
                                      "Using cached result if available. Try again in ~60 seconds.",
                "recommended_action": "Wait for quota reset, then click Refresh AI Analysis.",
                "confidence_level":   "N/A",
            }
        # API key missing / invalid
        if "API_KEY" in err_str.upper() or "401" in err_str or "403" in err_str or "invalid" in err_str.lower():
            return {
                "laundering_stage":   metrics_json.get("deterministic_stage", "N/A"),
                "risk_reasoning":     "Gemini API key is missing or invalid. "
                                      "Set GEMINI_API_KEY in your environment or .streamlit/secrets.toml.",
                "recommended_action": "Configure a valid GEMINI_API_KEY and reload.",
                "confidence_level":   "N/A",
            }
        # Model not found
        if "404" in err_str or "not found" in err_str.lower() or "model" in err_str.lower():
            return {
                "laundering_stage":   metrics_json.get("deterministic_stage", "N/A"),
                "risk_reasoning":     "Gemini model could not be resolved. "
                                      "The model name may have changed or the API is temporarily unavailable.",
                "recommended_action": "Check Gemini model availability and API key permissions.",
                "confidence_level":   "N/A",
            }
        # Generic fallback (truncated, never show raw traceback)
        return {
            "laundering_stage":   metrics_json.get("deterministic_stage", "N/A"),
            "risk_reasoning":     f"Gemini API call failed. Please check your connection and API key.",
            "recommended_action": "Verify API key, quota, and network connectivity.",
            "confidence_level":   "N/A",
        }


# ── Cached wrapper — deduped by metrics JSON string, TTL = 60 s ──────────────
# Takes a JSON *string* (not a dict) so st.cache_data can hash it.
# Only invoked when the user explicitly clicks "Generate AI Analysis".

import streamlit as _st  # available at module import time under Streamlit

@_st.cache_data(ttl=60, show_spinner=False)
def generate_intelligence_cached(metrics_json_str: str) -> dict:
    """
    Cached wrapper around generate_intelligence().

    Args:
        metrics_json_str: JSON-serialised metrics dict
                          (produce with json.dumps(metrics, sort_keys=True)).

    Returns:
        Same dict as generate_intelligence().
        Responses are cached for 60 s so identical metrics never re-hit the API.
    """
    try:
        metrics = json.loads(metrics_json_str)
    except (json.JSONDecodeError, TypeError):
        metrics = {}
    return generate_intelligence(metrics)


# ────────────────────────────────────────────────────────────────────────────
# B) GENERATE_INVESTIGATION_REPORT
# ────────────────────────────────────────────────────────────────────────────

def generate_investigation_report(summary_json: dict) -> str:
    """
    Generate a formatted AML investigation summary narrative using Gemini.

    Args:
        summary_json: dict with keys:
            accounts_involved, timeline_summary, estimated_loss

    Returns:
        Formatted AML investigation report as plain text.
    """
    model = _get_client()
    if model is None:
        return (
            "⚠️ GEMINI_API_KEY not configured.\n\n"
            "To enable AI-generated investigation reports:\n"
            "  • Set the GEMINI_API_KEY environment variable, OR\n"
            "  • Add it to .streamlit/secrets.toml as: GEMINI_API_KEY = \"your-key\"\n\n"
            "Investigation data summary:\n"
            f"  Accounts involved : {summary_json.get('accounts_involved', 'N/A')}\n"
            f"  Timeline          : {summary_json.get('timeline_summary', 'N/A')}\n"
            f"  Estimated loss    : ${summary_json.get('estimated_loss', 0):,.2f}"
        )

    prompt = f"""You are a Suspicious Activity Report (SAR) author at a financial intelligence unit.

Based on the following case summary data, draft a professional AML investigation report:

```json
{json.dumps(summary_json, indent=2)}
```

The report must include these sections:

1. CASE SUMMARY
   - Brief executive description of the detected financial crime pattern
   - Key risk indicators identified

2. TRANSACTION TIMELINE
   - Narrative description of how funds moved through the network
   - Estimated duration of the laundering operation

3. ACCOUNTS INVOLVED
   - Description of the roles (origin, mule accounts, cashout destination)
   - Estimated number of accounts compromised

4. FINANCIAL IMPACT
   - Estimated total funds at risk
   - Breakdown of what stage the funds are at

5. RECOMMENDED ACTIONS
   - Immediate steps (account freezes, notifications)
   - Medium-term steps (law enforcement referral, asset recovery)
   - Compliance obligations (SAR filing deadlines)

6. RISK CLASSIFICATION
   - Overall risk rating (Critical / High / Medium / Low)
   - Urgency score (1–10 with justification)

Write in professional regulatory language. Be specific and actionable.
Total length: 400–600 words."""

    try:
        client = _get_client()
        if client is None:
            return (
                "⚠️ GEMINI_API_KEY not configured.\n\n"
                "To enable AI-generated investigation reports:\n"
                "  • Set the GEMINI_API_KEY environment variable, OR\n"
                "  • Add it to .streamlit/secrets.toml as: GEMINI_API_KEY = \"your-key\"\n\n"
                "Investigation data summary:\n"
                f"  Accounts involved : {summary_json.get('accounts_involved', 'N/A')}\n"
                f"  Timeline          : {summary_json.get('timeline_summary', 'N/A')}\n"
                f"  Estimated loss    : ${summary_json.get('estimated_loss', 0):,.2f}"
            )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            return (
                "⚠️ Gemini API quota exceeded (429).\n\n"
                "Please wait approximately 60 seconds before generating the report again."
            )
        if "API_KEY" in err_str.upper() or "401" in err_str or "403" in err_str:
            return (
                "❌ Gemini API key is missing or invalid.\n\n"
                "Set GEMINI_API_KEY in your environment or .streamlit/secrets.toml."
            )
        return (
            "❌ Report generation failed. Please check your Gemini API key and network connection."
        )


# ────────────────────────────────────────────────────────────────────────────
# UTILITY — Build metrics_json from predictor + session data
# ────────────────────────────────────────────────────────────────────────────

def build_metrics_json(pred_df, ring_summary: dict) -> dict:
    """
    Build the metrics dict that generate_intelligence() expects.
    Includes the deterministic stage computed by stage_predictor so
    Gemini never needs to (re)classify it.
    """
    if pred_df is None or pred_df.empty:
        return {
            "hop_count": -1,
            "fan_out": 0.0,
            "burst_score": 0.0,
            "avg_hop_time": 0.0,
            "predicted_exit_minutes": -1,
            "compromised_status": False,
            "deterministic_stage": "Normal",
            "eta_minutes": -1,
        }

    top = pred_df.iloc[0]
    hops     = int(top.get("hops_to_cashout", -1))
    burst    = float(top.get("burst_score", 0))
    avg_hop_time = round((1.0 - burst) * 30, 1)  # heuristic

    ttc = ring_summary.get("min_time_to_cashout", -1)

    # Deterministic stage comes from stage_predictor, never from Gemini
    deterministic_stage = ring_summary.get("dominant_label", "Normal")
    eta_minutes = round(ttc, 1) if ttc >= 0 else -1

    return {
        "hop_count":               hops,
        "fan_out":                 round(float(top.get("fan_out_ratio", 0)), 3),
        "burst_score":             round(burst, 3),
        "avg_hop_time":            avg_hop_time,
        "predicted_exit_minutes":  eta_minutes,
        "compromised_status":      ring_summary.get("max_stage", 0) >= 1,
        "deterministic_stage":     deterministic_stage,   # ← single source of truth
        "eta_minutes":             eta_minutes,
    }


def build_summary_json(ring_summary: dict, transactions_df, ring_accounts: list) -> dict:
    """Build the summary dict that generate_investigation_report() expects."""
    n_accounts = len(ring_accounts)

    # Build a crude timeline from ring_summary stage
    stage = ring_summary.get("dominant_label", "Unknown")
    ttc = ring_summary.get("min_time_to_cashout", -1)
    timeline = f"Funds currently in '{stage}' phase. "
    if ttc > 0:
        timeline += f"Estimated {ttc:.0f} minutes until cashout."
    elif ttc == 0:
        timeline += "Cashout appears to be in progress or completed."

    estimated_loss = ring_summary.get("estimated_loss_usd", 0.0)

    return {
        "accounts_involved": n_accounts,
        "timeline_summary":  timeline,
        "estimated_loss":    round(estimated_loss, 2),
        "laundering_stage":  stage,
        "max_dna_score":     ring_summary.get("max_cashout_probability", 0),
    }
