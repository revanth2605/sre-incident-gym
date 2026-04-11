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

**SRE-Incident-Gym** is a Reinforcement Learning (RL) environment and benchmark for evaluating AI agents on automated incident response. The benchmark measures an agent's ability to minimize Mean Time To Recovery (MTTR) while controlling operational cost and avoiding unnecessary mitigations.

---

## Table of Contents

1. [Environment Description & Motivation](#environment-description--motivation)
2. [Architecture & Key Files](#architecture--key-files)
3. [Observation Space](#observation-space)
4. [Action Space](#action-space)
5. [Task Definitions](#task-definitions)
6. [Reward Design](#reward-design)
7. [RL Training](#rl-training)
8. [Benchmark Results](#benchmark-results)
9. [Setup & Usage](#setup--usage)
10. [Dashboard](#the-mission-control-dashboard)

---

## Environment Description & Motivation

Modern cloud systems fail in complex, context-dependent ways. On-call engineers must triage incidents rapidly — but human response is slow, inconsistent, and expensive at scale. SRE-Incident-Gym simulates this challenge as a sequential decision-making problem.

The environment presents an agent with real-time telemetry (HTTP status, CPU load, memory, logs) and asks it to choose remediation actions. The reward signal is shaped to:

- **Minimise MTTR** — step penalties push agents toward fast solutions.
- **Penalise wrong moves** — unnecessary restarts, over-provisioning, and misapplied patches are costly.
- **Reward correct sequencing** — multi-step tasks (scale then patch) require the agent to reason about order and dependencies.
- **Test noise filtering** — scheduled batch windows should be left alone; agents that intervene are penalised.

This makes SRE-Incident-Gym a focused, discriminative benchmark that separates context-aware agents from naive or random policies.

---

## Architecture & Key Files

```
sre-incident-gym/
├── environment.py      # Core RL environment logic and reward engine
├── models.py           # Pydantic models for observations, actions, rewards
├── server/
│   ├── __init__.py
│   └── app.py          # OpenEnv-compliant FastAPI server (validator entry point)
├── rl_env.py           # Gymnasium wrapper for RL training
├── rl_train.py         # PPO training script (stable-baselines3)
├── dashboard.py        # Streamlit Mission Control UI
├── inference.py        # LLM-based agent evaluation harness
├── baseline_test.py    # Random policy baseline evaluator
├── start.sh            # Container startup script
├── Dockerfile          # Docker build (trains RL model at build time)
├── requirements.txt
├── pyproject.toml
└── uv.lock
```

### Why `environment.py`?

`environment.py` is the single source of truth for all environment logic — state transitions, reward computation, task initialisation, and success conditions. Keeping it separate from the API layer means:

- The reward engine can be tested and iterated independently of the server.
- `rl_env.py` wraps it directly into a Gymnasium-compatible interface for RL training without duplicating logic.
- Any change to reward shaping is made in one place and propagates everywhere automatically.
- It follows the standard Gym pattern (`reset()`, `step()`, `state()`) making it easy to swap in alternative backends or extend with new tasks.

### Why `models.py`?

`models.py` defines strict Pydantic V2 models for every data structure crossing a boundary — observations, actions, and rewards. This provides:

- **Type safety** — invalid actions are rejected at the API boundary before reaching environment logic.
- **Serialisation** — Pydantic handles JSON encoding/decoding for FastAPI endpoints automatically.
- **Discriminated union actions** — the `Action` union type allows the API to accept any of the five action types through a single endpoint, validated by `action_type` literal fields.
- **Reuse** — both `environment.py` and `rl_env.py` import from `models.py`, ensuring consistent types across the entire stack with no duplication.

---

## Observation Space

At each step the agent receives an `Observation` object with five fields:

| Field | Type | Range | Description |
|---|---|---|---|
| `status_code` | int | 100–599 | HTTP status code of the service |
| `cpu_load` | float | 0.0–100.0 | CPU utilisation percentage |
| `memory_usage` | float | 0.0–100.0 | Memory utilisation percentage |
| `last_log_entry` | str | — | Most recent log line from the service |
| `is_patched` | bool | True/False | Whether the security patch has been applied |

The RL agent additionally encodes these into a normalised 6-float vector for the neural network policy:

```
[status_ok, cpu_norm, mem_norm, is_patched, task_id_norm, step_norm]
```

All values are in `[0, 1]`, compatible with standard MLP policies in stable-baselines3.

---

## Action Space

Agents select one action per step from a discrete set of five:

| Action | Parameters | Description |
|---|---|---|
| `nop` | — | Do nothing; observe and wait |
| `check_logs` | — | Retrieve latest log entry (+0.05 diagnostic reward) |
| `restart_service` | — | Restart the service to recover from 5xx errors |
| `scale_up` | `replicas` (int, 1–10) | Add replicas to reduce CPU load |
| `apply_patch` | `patch_id` (str) | Deploy a security or bug-fix patch |

**Example JSON actions:**
```json
{"action_type": "nop"}
{"action_type": "restart_service"}
{"action_type": "scale_up", "replicas": 5}
{"action_type": "apply_patch", "patch_id": "CVE-2026-1234"}
```

---

## Task Definitions

Four tasks evaluate distinct SRE competencies. Initial conditions are randomised within each task to enforce generalisation over memorisation.

### Task 1 — Service Recovery (Easy)
- **Scenario:** Service returns HTTP 500. CPU is moderate, system is patched.
- **Correct action:** `restart_service`
- **Success condition:** `status_code == 200`
- **Competency:** Incident triage — recognising a service crash and applying the correct recovery action immediately.
- **Difficulty:** Easy. Single correct action, unambiguous signal from status code.

### Task 2 — Resource Management (Medium)
- **Scenario:** Service is healthy (200) but CPU is critically high (70–99%, randomised per episode).
- **Correct action:** `scale_up` with 4–5 replicas
- **Success condition:** `cpu_load ≤ 20%`
- **Competency:** Capacity management — identifying CPU saturation and scaling without over-provisioning.
- **Difficulty:** Medium. Randomised starting CPU requires generalisation; restart is penalised as a wrong move.

### Task 3 — Complex Remediation (Hard)
- **Scenario:** CPU critically high (70–99%) AND a CVE vulnerability is unpatched.
- **Correct actions:** `scale_up` first (brings CPU ≤ 20%), then `apply_patch`
- **Success condition:** `cpu_load ≤ 20% AND is_patched == True`
- **Competency:** Multi-step sequencing — both conditions must be satisfied in order. Either action alone does not complete the task.
- **Difficulty:** Hard. Requires correct ordering across two steps; wrong-move penalties apply if actions are misapplied.

### Task 4 — Noise Filtering (Expert)
- **Scenario:** CPU spike caused by a scheduled batch job, not an incident. The log explicitly states the batch window is active.
- **Correct action:** `nop`
- **Success condition:** Agent takes no mitigation action
- **Competency:** Contextual restraint — reading the log and recognising that intervention is incorrect. Any mitigation action incurs a penalty and ends the episode immediately.
- **Difficulty:** Expert. Tests whether the agent suppresses action based on textual context rather than numeric telemetry alone.

---

## Reward Design

The reward engine uses dense shaping to produce a clear learning signal at every step:

| Event | Reward |
|---|---|
| Per step taken (always applied) | −0.10 |
| Correct remediation action | +0.80 |
| Wrong / unnecessary action | −0.40 |
| `check_logs` diagnostic action | +0.05 |
| Extra `nop` penalty (non-scheduled task) | −0.10 |
| Correct `nop` on scheduled task | +0.80 |
| Wrong action on scheduled task | −0.20, episode ends immediately |

All rewards are clamped to `[−1.0, 1.0]`. The per-step penalty is the core MTTR proxy — agents that solve incidents in one step score higher than agents that take three steps to reach the same outcome.

---

## RL Training

The project includes a fully trained **PPO (Proximal Policy Optimisation)** agent built with `stable-baselines3`.

### How it works

`rl_env.py` wraps `SREIncidentGym` from `environment.py` into a `gymnasium.Env`:

- **Observation space:** `Box(6,)` — normalised float vector (see above)
- **Action space:** `Discrete(5)` — maps integers 0–4 to the five action types

`rl_train.py` trains a PPO agent with a 2-layer MLP policy `[64, 64]` with task randomised per episode (tasks 1–4 sampled uniformly), forcing the agent to generalise across all incident types rather than specialise on one.

The model is trained automatically during `docker build` and saved to `models/ppo_sre_agent.zip`, so it is ready immediately when the Space starts.

### Training commands

```bash
python rl_train.py                    # train all tasks, 50k timesteps (default)
python rl_train.py --timesteps 30000  # faster training run
python rl_train.py --task 3           # train on task 3 only
python rl_train.py --eval             # evaluate saved model (20 episodes per task)
python rl_train.py --demo --task 1    # watch agent solve task 1 step by step
python rl_train.py --compare          # PPO vs random baseline side-by-side
```

### Training progression (30k timesteps)

| Episode | Mean Reward (last 20) | Mean Steps |
|---|---|---|
| 20 | −0.60 | 6.5 |
| 60 | +0.07 | 2.9 |
| 100 | +0.30 | 3.1 |
| 300 | +0.59 | 1.9 |
| 500 | +0.69 | 1.4 |
| 780 | +0.95 | 1.6 |
| 1100+ | +0.90 (stable) | 1.1–1.4 |

The agent converges to near-optimal behaviour — solving most tasks in a single step — by episode ~300.

---

## Benchmark Results

### Overall agent comparison

| Agent | Overall Score | Success Rate |
|:---|:---:|:---:|
| **PPO RL Agent (trained)** | **0.875** | **100%** |
| **GPT-4 Rule Agent** | **0.530** | **100%** |
| **Random Baseline** | **0.180** | **25%** |

### Per-task PPO performance (20 episodes each)

| Task | PPO Mean Reward | Random Mean Reward | Lift |
|:---|:---:|:---:|:---:|
| Task 1 — Restart | +0.70 | +0.07 | **+0.63** |
| Task 2 — Scale CPU | +0.70 | −0.39 | **+1.09** |
| Task 3 — Scale + Patch | +1.40 | −0.47 | **+1.87** |
| Task 4 — Scheduled nop | +0.70 | −0.07 | **+0.77** |

> The ~5× gap on Task 3 is the most telling result — random agents almost always apply wrong-move penalties on the multi-step task, while the PPO agent learns the correct two-action sequence reliably.

---

## Setup & Usage

### Local development

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/yourusername/sre-incident-gym
cd sre-incident-gym

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the RL model (one time)
python rl_train.py --timesteps 30000

# 5. Start the API server (terminal 1)
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# 6. Start the dashboard (terminal 2)
streamlit run dashboard.py
# Open http://localhost:7860
```

### Docker

```bash
# Build image (automatically trains RL model during build ~60s extra)
docker build -t sre-incident-gym .

# Run container
docker run -p 7860:7860 sre-incident-gym

# Open dashboard
open http://localhost:7860
```

### API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Reset environment to a task (body: `{"task": 1}`) |
| `POST` | `/step` | Execute one action (body: `{"action": {...}}`) |
| `GET` | `/state` | Get full environment state |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive Swagger UI |

```bash
# Quick API test
curl http://localhost:8000/health

curl -X POST http://localhost:8000/api/reset \
  -H "Content-Type: application/json" \
  -d '{"task": 1}'

curl -X POST http://localhost:8000/api/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "restart_service"}}'
```

### Run evaluations

```bash
python inference.py           # LLM-driven evaluation (set OPENAI_API_KEY env var)
python baseline_test.py       # Random policy baseline
python rl_train.py --eval     # PPO agent evaluation across all tasks
python rl_train.py --compare  # PPO vs random baseline comparison
```

---

## The Mission Control Dashboard

The Streamlit dashboard at port `7860` provides interactive mission control:

- **Live metrics:** Status Code, CPU Load, Memory, per-agent reward
- **Two agent controllers side by side:**
  - 🧠 **RL Agent** — the trained PPO model selects actions from the normalised observation vector
  - 📋 **Rule Agent** — deterministic heuristic (`if status==500 → restart`, `if cpu>60 → scale`, etc.)
- **Live reward chart** — sidebar line chart updates after every step showing RL vs Rule rewards
- **Service log console** — shows `last_log_entry` after each action
- **Alerts** — critical / warning / success indicators with 🎉 balloons on task completion

---

## Scientific Notes

- Initial states are randomised within each task (CPU load, memory usage) to prevent memorisation and enforce generalisation to unseen starting conditions.
- Fixed seeds (`SREIncidentGym(seed=42)`) are supported for deterministic reproducibility when needed.
- The compact 5-action, 6-feature space makes this suitable for rapid iteration and ablation studies on reward shaping.
- Task 3 is intentionally unsolvable in one step — it specifically tests whether agents learn action sequencing, not just action classification.

---

## Contact & Contribution

Contributions, issues, and benchmark reproducibility reports are welcome. Please open an issue or submit a pull request.

**License:** MIT

---

*Prepared as a benchmark report for judges and reviewers — OpenEnv Hackathon 2026.*