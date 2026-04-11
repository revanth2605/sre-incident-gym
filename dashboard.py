import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import requests

# ── API base ────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:7860/api")

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="SRE Incident Mission Control",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.reportview-container { background: #0e1117; color: #c9d1d9; }
.sidebar .sidebar-content { background: #0b0f14; }
.stButton>button { background-color: #1f6feb; color: white; border-radius: 6px; }
pre { background: #0b0f14; color: #d3d7de; }
</style>
""", unsafe_allow_html=True)

# ── Load PPO model (once, cached) ───────────────────────────────────
@st.cache_resource
def load_rl_model():
    try:
        from stable_baselines3 import PPO
        model_path = "models/ppo_sre_agent.zip"
        if os.path.exists(model_path):
            return PPO.load(model_path), True
        return None, False
    except Exception as e:
        return None, False

rl_model, rl_available = load_rl_model()

# ── Session state defaults ───────────────────────────────────────────
for key, default in {
    "last_reward":         0.0,
    "rl_reward":           0.0,
    "rule_reward":         0.0,
    "rl_last_action":      "—",
    "rule_last_action":    "—",
    "run_rl_step":         False,
    "run_rule_step":       False,
    "reward_history_rl":   [],
    "reward_history_rule": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("🛡️ Mission Control")
task_id = st.sidebar.selectbox(
    "Select Task ID", [1, 2, 3, 4], index=0,
    help="1=Restart  2=Scale CPU  3=Scale+Patch  4=Scheduled nop"
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Environment", use_container_width=True):
    try:
        r = requests.post(f"{API_BASE}/reset", json={"task": task_id}, timeout=5)
        r.raise_for_status()
        st.session_state.rl_reward           = 0.0
        st.session_state.rule_reward         = 0.0
        st.session_state.rl_last_action      = "—"
        st.session_state.rule_last_action    = "—"
        st.session_state.reward_history_rl   = []
        st.session_state.reward_history_rule = []
        st.sidebar.success(f"✅ Reset to Task {task_id}")
    except Exception as e:
        st.sidebar.error(f"Reset failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Agent Controllers")
col_a, col_b = st.sidebar.columns(2)
if col_a.button("🧠 RL Agent", use_container_width=True, disabled=not rl_available):
    st.session_state.run_rl_step = True
if col_b.button("📋 Rule Agent", use_container_width=True):
    st.session_state.run_rule_step = True

if not rl_available:
    st.sidebar.warning("⚠️ Model not found. Rebuild the Docker image.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Live Reward Comparison")
if st.session_state.reward_history_rl or st.session_state.reward_history_rule:
    max_len  = max(len(st.session_state.reward_history_rl),
                   len(st.session_state.reward_history_rule))
    chart_df = pd.DataFrame({
        "RL Agent":   st.session_state.reward_history_rl   + [None] * (max_len - len(st.session_state.reward_history_rl)),
        "Rule Agent": st.session_state.reward_history_rule + [None] * (max_len - len(st.session_state.reward_history_rule)),
    })
    st.sidebar.line_chart(chart_df, height=180)
else:
    bench_df = pd.DataFrame(
        {"Score": [0.875, 0.53, 0.18]},
        index=["PPO (trained)", "Rule-based", "Random"]
    )
    st.sidebar.bar_chart(bench_df, height=180)

# ── Helpers ──────────────────────────────────────────────────────────
def get_state():
    try:
        r = requests.get(f"{API_BASE}/state", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to fetch state: {e}")
        return None

def post_step(action: dict):
    try:
        r = requests.post(f"{API_BASE}/step", json={"action": action}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def parse_reward(resp: dict) -> float:
    raw = resp.get("reward", 0.0)
    return raw.get("value", 0.0) if isinstance(raw, dict) else float(raw or 0.0)

def rule_based_action(obs: dict) -> dict:
    last_log = obs.get("last_log_entry", "") or ""
    if any(w in last_log.lower() for w in ["scheduled", "batch", "window"]):
        return {"action_type": "nop"}
    status  = obs.get("status_code", 200)
    cpu     = obs.get("cpu_load", 0)
    patched = obs.get("is_patched", True)
    if status != 200:
        return {"action_type": "restart_service"}
    if not patched:
        return {"action_type": "apply_patch", "patch_id": "CVE-2026-SRE-FIX"}
    if cpu and cpu > 60:
        return {"action_type": "scale_up", "replicas": min(6, max(2, int(cpu / 20)))}
    return {"action_type": "check_logs"}

def rl_action(obs: dict) -> dict:
    ACTION_NAMES = {0:"check_logs", 1:"restart_service", 2:"scale_up", 3:"apply_patch", 4:"nop"}
    if rl_model is None:
        return {"action_type": "nop"}
    task_map = {"EASY": 0, "MEDIUM": 1, "HARD": 2, "SCHEDULED": 3}
    task_idx = task_map.get(obs.get("task_level", "EASY"), 0)
    obs_vec  = np.array([
        1.0 if obs.get("status_code", 200) == 200 else 0.0,
        obs.get("cpu_load",     0.0) / 100.0,
        obs.get("memory_usage", 0.0) / 100.0,
        1.0 if obs.get("is_patched", False) else 0.0,
        task_idx / 3.0,
        min(obs.get("step_count", 0) / 15.0, 1.0),
    ], dtype=np.float32)
    action_int, _ = rl_model.predict(obs_vec, deterministic=True)
    at = ACTION_NAMES[int(action_int)]
    if at == "scale_up":
        return {"action_type": "scale_up", "replicas": 5}
    if at == "apply_patch":
        return {"action_type": "apply_patch", "patch_id": "CVE-2026-SRE-FIX"}
    return {"action_type": at}

def render_obs(obs, reward, agent_label):
    status = obs.get("status_code", "n/a")
    cpu    = obs.get("cpu_load",    0.0)
    done   = obs.get("done", False)
    if status == 500:
        st.error(f"{agent_label} 🔴 Service DOWN | Reward: {reward:+.2f}")
    elif cpu > 80:
        st.warning(f"{agent_label} 🟡 High CPU {cpu:.0f}% | Reward: {reward:+.2f}")
    elif done:
        st.success(f"{agent_label} ✅ Task Solved! | Reward: {reward:+.2f}")
    else:
        st.info(f"{agent_label} 🟢 Running | Reward: {reward:+.2f}")

# ── Main layout ──────────────────────────────────────────────────────
st.title("🛡️ SRE-Incident-Gym — Mission Control")
st.caption(f"API: {API_BASE}")

# Metrics row
m1, m2, m3, m4, m5 = st.columns(5)
status_card   = m1.empty()
cpu_card      = m2.empty()
mem_card      = m3.empty()
rl_rwd_card   = m4.empty()
rule_rwd_card = m5.empty()

st.markdown("---")

# Agent comparison panels
left_col, right_col = st.columns(2)
with left_col:
    st.markdown("#### 🧠 PPO RL Agent")
    rl_action_display  = st.empty()
    rl_status_display  = st.empty()
with right_col:
    st.markdown("#### 📋 Rule-Based Agent")
    rule_action_display  = st.empty()
    rule_status_display  = st.empty()

st.markdown("---")
st.markdown("**📟 Service Log**")
log_box = st.empty()

# ── Initial render ───────────────────────────────────────────────────
state = get_state()
if state:
    status_card.metric("Status Code", str(state.get("status_code", "n/a")))
    cpu_card.metric("CPU Load",       f"{state.get('cpu_load', 0.0):.1f}%")
    mem_card.metric("Memory",         f"{state.get('memory_usage', 0.0):.1f}%")
    rl_rwd_card.metric("RL Reward",   f"{st.session_state.rl_reward:+.2f}")
    rule_rwd_card.metric("Rule Reward",f"{st.session_state.rule_reward:+.2f}")
    rl_action_display.info(f"Last action: **{st.session_state.rl_last_action}**")
    rule_action_display.info(f"Last action: **{st.session_state.rule_last_action}**")
    log_box.code(state.get("last_log_entry", "") or "(no logs)")
    if state.get("done", False):
        st.success("🎉 Task already solved! Reset to start a new episode.")

# ── RL Agent step ────────────────────────────────────────────────────
if st.session_state.get("run_rl_step", False):
    st.session_state.run_rl_step = False
    state = get_state()
    if state:
        action = rl_action(state)
        st.session_state.rl_last_action = action["action_type"]
        with st.spinner("🧠 RL agent stepping..."):
            resp = post_step(action)
            time.sleep(0.2)
        if "error" in resp:
            st.error(f"RL step error: {resp['error']}")
        else:
            reward = parse_reward(resp)
            st.session_state.rl_reward = reward
            st.session_state.reward_history_rl.append(reward)
            obs  = resp.get("observation", {})
            done = resp.get("done", False)
            # update metrics
            status_card.metric("Status Code", str(obs.get("status_code", "n/a")))
            cpu_card.metric("CPU Load",       f"{obs.get('cpu_load', 0.0):.1f}%")
            mem_card.metric("Memory",         f"{obs.get('memory_usage', 0.0):.1f}%")
            rl_rwd_card.metric("RL Reward",   f"{reward:+.2f}")
            rl_action_display.info(f"Last action: **{action['action_type']}**")
            log_box.code(obs.get("last_log_entry", "") or "(no logs)")
            render_obs({"status_code": obs.get("status_code"), "cpu_load": obs.get("cpu_load", 0), "done": done},
                       reward, "🧠 RL Agent")
            if done:
                st.balloons()

# ── Rule-based Agent step ─────────────────────────────────────────────
if st.session_state.get("run_rule_step", False):
    st.session_state.run_rule_step = False
    state = get_state()
    if state:
        action = rule_based_action(state)
        st.session_state.rule_last_action = action["action_type"]
        with st.spinner("📋 Rule agent stepping..."):
            resp = post_step(action)
            time.sleep(0.2)
        if "error" in resp:
            st.error(f"Rule step error: {resp['error']}")
        else:
            reward = parse_reward(resp)
            st.session_state.rule_reward = reward
            st.session_state.reward_history_rule.append(reward)
            obs  = resp.get("observation", {})
            done = resp.get("done", False)
            # update metrics
            status_card.metric("Status Code",  str(obs.get("status_code", "n/a")))
            cpu_card.metric("CPU Load",        f"{obs.get('cpu_load', 0.0):.1f}%")
            mem_card.metric("Memory",          f"{obs.get('memory_usage', 0.0):.1f}%")
            rule_rwd_card.metric("Rule Reward",f"{reward:+.2f}")
            rule_action_display.info(f"Last action: **{action['action_type']}**")
            log_box.code(obs.get("last_log_entry", "") or "(no logs)")
            render_obs({"status_code": obs.get("status_code"), "cpu_load": obs.get("cpu_load", 0), "done": done},
                       reward, "📋 Rule Agent")
            if done:
                st.balloons()

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Tip:** Reset → pick a task → click **🧠 RL Agent** or **📋 Rule Agent** "
    "to step through. Live reward chart updates in the sidebar."
)