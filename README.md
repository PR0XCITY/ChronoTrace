# ChronoTrace

> **Predicting Financial Crime Before Money Disappears.**

ChronoTrace is an enterprise-grade fintech cybersecurity intelligence platform that detects simulated mule ring behaviour using graph analytics and predicts **time-to-cashout** before laundering completes.

---

## Overview

| Feature | Description |
|---|---|
| 🕸 Graph Intelligence | NetworkX-powered directed transaction graph |
| 🧬 DNA Scoring | 6-factor behavioural anomaly engine |
| ⏱ Time-to-Cashout | Stage-aware cashout ETA predictor |
| 🚨 Alert Feed | Prioritised real-time threat intelligence |
| 🧊 Interventions | Simulate Freeze / Monitor outcomes |
| 📥 Export | Download JSON threat report |

---

## Project Structure

```
ChronoTrace/
│
├── simulate.py        # Simulation engine: accounts, normal tx, mule ring injection
├── dna_engine.py      # Graph analytics: DNA scores, cluster detection, layout
├── predictor.py       # Rule-based stage classifier & time-to-cashout estimator
├── alerts.py          # Alert generators & intervention outcome calculator
├── app.py             # Streamlit enterprise dashboard
├── requirements.txt   # Minimal Python dependencies
└── README.md
```

---

## Setup & Local Run

### 1. Clone / Download

```bash
git clone https://github.com/your-org/chronotrace.git
cd chronotrace
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**.

---

## Usage

1. Open the sidebar and select a **Simulation Mode**:
   - **Attack Simulation** — injects one or more mule rings into the transaction network
   - **Normal (Baseline)** — generates a clean transaction graph for comparison
2. For attack mode, choose the **number of mule rings** (1–3) using the slider.
3. Click **▶ Run Simulation**.
4. Explore:
   - **Network Graph** — red/orange nodes = ring members, blue = normal
   - **Alert Feed** — real-time prioritised threat events
   - **Laundering Stage Tracker** — current position in the attack lifecycle
   - **DNA Breakdown** — metric contribution of the top-risk node
   - **Risk Gauge** — composite threat index
   - **Velocity Chart** — transaction burst detection over time
   - **Intervention Panel** — simulate freeze actions and see financial outcomes
5. Download the threat report via **📥 Export Report (JSON)**.

---

## Deployment — Streamlit Community Cloud

1. Push the repository to GitHub (make sure all files are committed).
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in.
3. Click **New app** → select your repo and set:
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**.

> No additional secrets or configuration required.

---

## Deployment — Render.com

1. Create a new **Web Service** on [render.com](https://render.com).
2. Connect your GitHub repository.
3. Configure the service:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Click **Deploy**.

---

## Intelligence Architecture

```
simulate.py
  └── run_simulation()
        ├── generate_accounts()             # 1000 synthetic accounts
        ├── generate_normal_transactions()  # Benign background activity
        └── inject_mule_ring()              # 4-stage laundering pattern

dna_engine.py
  └── analyse()
        ├── build_graph()                   # NetworkX DiGraph from transactions
        ├── compute_dna_scores()            # 6 behavioural metrics per node
        │     ├── Fan-out ratio
        │     ├── Velocity score
        │     ├── Burst score
        │     ├── Circularity
        │     ├── Amount anomaly
        │     └── Hop proximity
        ├── detect_suspicious_clusters()    # High-risk connected components
        └── compute_layout()               # 2D Fruchterman-Reingold (no scipy)

predictor.py
  └── predict()
        ├── classify_stage()                # Stage 0–4 rule classifier
        ├── estimate_time_to_cashout()      # ETA in minutes
        └── compute_cashout_probability()   # 0–100% risk probability

alerts.py
  └── generate_all_alerts()
        ├── generate_system_alerts()
        ├── generate_ring_alerts()
        ├── generate_cashout_alerts()
        ├── generate_burst_alerts()
        ├── generate_compromise_alerts()
        └── generate_velocity_alerts()
```

---

## Laundering Stage Reference

| Stage | Label | Description |
|---|---|---|
| 0 | Normal | No suspicious activity |
| 1 | Compromised | Origin breach detected |
| 2 | Layering | Rapid redistribution through mules |
| 3 | Pre-Cashout | Funds aggregating near exit |
| 4 | Exit Imminent | Cashout transaction in progress |

---

## License

MIT License — © 2025 ChronoTrace Labs
