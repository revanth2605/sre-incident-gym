
---
title: SRE Incident Gym
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
# SRE-Incident-Gym — Benchmark Report

**SRE-Incident-Gym** is a Reinforcement Learning (RL) environment and benchmark for evaluating AI agents on automated incident response. The benchmark is designed to measure an agent's ability to minimize Mean Time To Recovery (MTTR) while controlling operational cost and avoiding unnecessary mitigations.

This document is a concise, judge-ready report describing the environment, the included UI, the action/observation spaces, canonical tasks, and benchmark results.

---

## Project Overview

SRE-Incident-Gym provides a compact but realistic testbed for automated SRE decision-making. Agents receive metric-based observations and textual logs, then choose actions from a small, meaningful action space. The goal: resolve incidents quickly and correctly while avoiding wasteful actions.

Key motivations:

- Reduce MTTR by training agents to identify and apply correct remediations.
- Penalize costly or unnecessary mitigations (cost-awareness).
- Force generalization using randomized initial conditions and noisy action effects.

Included components:

- `environment.py` — the RL environment with OpenEnv-style API.
- `models.py` — Pydantic models for observations, actions, and rewards.
- `main.py` — FastAPI server exposing `/reset`, `/step`, `/state`, `/health`.
- `inference.py` — evaluation harness (LLM-capable agent runner).
- `baseline_test.py` — random-policy baseline evaluator.
- `dashboard.py` — Streamlit "Mission Control" frontend for interactive evaluation.

---

## Recent Updates (April 2026)

These updates polish the runtime and deployment experience for a production-style hackathon submission:

- Backend now runs on port `7860` (primary external port), ensuring OpenEnv validators can directly access `/reset` and `/step` endpoints without proxy issues.
- `start.sh` launches FastAPI bound to `0.0.0.0:7860`, waits for the backend to become healthy, and then starts Streamlit on `0.0.0.0:8501`, keeping the API as the primary entry point.
- `Dockerfile` updated to run using a non-root user (UID `1000`) with `--chown` permissions to prevent filesystem and permission issues on hosted platforms like Hugging Face Spaces.
- Streamlit dashboard (`dashboard.py`) now retries `/state` up to 5 times with a short delay and uses Plotly gauges for improved visualization of CPU load and reward metrics.
- The environment (`environment.py`) includes a lightweight `close()` method, and the FastAPI server handles graceful shutdown using SIGTERM signals for better container lifecycle management.
- LLM integration includes a fallback to a deterministic heuristic when `OPENAI_API_KEY` is not provided, ensuring stable execution in environments without API access. Set the `OPENAI_API_KEY` secret to enable full LLM-driven evaluation.

These changes fix common runtime failures (permission, port collisions, and missing dependencies) and improve demo reliability.

---

## The "Mission Control" Dashboard

The Streamlit dashboard provides an interactive mission control for runs:

- Real-time metrics: Status Code, CPU Load, and Current Reward.
- Live log console with the most recent `last_log_entry`.
- One-click AI step: trigger the agent and update metrics immediately.
- Alerts and visual feedback: critical, warning, and success indicators.
- Comparison panel: quick performance comparison (Smart vs Baseline).

This UI is designed for judges to reproduce and inspect agent behavior quickly.

---

## Action Space

Agents select one action per step. Actions are JSON objects matching the following set:

- `nop` — No operation (observe and wait).
- `restart_service` — Restart the service to recover from 5xx errors.
- `scale_up` — Increase replicas; parameter: `replicas` (int).
- `apply_patch` — Deploy a patch; parameter: `patch_id` (str).

Examples:

```json
{"action_type": "nop"}
{"action_type": "restart_service"}
{"action_type": "scale_up", "replicas": 4}
{"action_type": "apply_patch", "patch_id": "CVE-2026-1234"}
```

---

## Observation Space

Each step the agent receives an `Observation` object containing:

- `status_code` (int): HTTP status code of the service (200, 500, 503, ...).
- `cpu_load` (float): CPU utilization percentage (0.0–100.0).
- `memory_usage` (float): Memory utilization percentage (0.0–100.0).
- `last_log_entry` (str): Latest free-text log message from the service.
- `is_patched` (bool): Whether the system has been patched.

Agents must combine numeric telemetry and textual logs to make correct decisions.

---

## Task Definitions (Canonical Benchmark)

Four tasks evaluate distinct SRE competencies. Each task includes randomized initial conditions to enforce generalization.

- **Task 1 — Service Recovery (Easy)**
  - Scenario: service returns 5xx (500/503). Decision: `restart_service`.
  - Primary competency: incident triage and recovery.

- **Task 2 — Resource Management (Medium)**
  - Scenario: sustained high CPU load. Decision: `scale_up` with an adequate `replicas` value.
  - Primary competency: capacity management and resource trade-offs.

- **Task 3 — Complex Remediation (Hard)**
  - Scenario: performance issue plus a security finding (CVE). Decisions: sequence of `scale_up` (to reach healthy CPU) and `apply_patch` (to remediate vulnerability).
  - Primary competency: multi-step remediation and sequencing.

- **Task 4 — Noise Filtering (Expert)**
  - Scenario: scheduled batch/maintenance window causing expected load. Decision: `nop` (detect and refrain from mitigation).
  - Primary competency: contextual restraint and log parsing.

---

## Reward Design (Summary)

The environment uses reward shaping to create clear signal for learning:

- **Fixed completion bonus** for correct remediation (e.g., +0.80).
- **Step decay**: per-step penalty to favor faster solutions (e.g., -0.10 per step).
- **Wrong-move penalty**: negative reward when action mismatches observation (e.g., scaling a downed service: -0.40).

This standardized engine is intended to produce reproducible, discriminative scores between smart and naive agents.

---

## Benchmarking Results — The Performance Gap

This table reflects representative scores obtained during evaluation runs (averaged across seeds):

| Agent | Overall Score | Success Rate |
| :--- | :---: | :---: |
| **GPT-4 Agent (Smart)** | **0.53** | **100%** |
| **Random Baseline** | **0.18** | **25%** |

> **Interpretation:** The ~3× gap demonstrates the environment's ability to reward efficient, context-aware remediation while penalizing reckless or uninformed actions.

---

## Installation & Quick Usage

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the backend server:

```bash
python main.py
```

4. Start the Streamlit dashboard (optional):

```bash
streamlit run dashboard.py
```

5. Run evaluations:

```bash
python inference.py         # LLM-driven evaluation
python baseline_test.py     # Random baseline
```

---

## Scientific Notes

- The environment intentionally includes stochastic initial states and variable action impacts to model real-world uncertainty.
- Repeat experiments to measure variance; use fixed seeds for deterministic reproducibility when needed.
- The compact action and observation spaces make this a focused benchmark suitable for rapid iteration and demonstration.

---

## Contact & Contribution

Contributions, issues, and benchmark reproducibility reports are welcome. Please open an issue or submit a pull request.

**License:** MIT — see `LICENSE`.

---

*Prepared as a concise benchmark report for judges and reviewers.*
