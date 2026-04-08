import os
import streamlit as st
import requests
import pandas as pd
import time

# Prefer environment variable; fallback to local loopback (used in Docker/HF Spaces)
# The FastAPI backend runs internally on port 8000; Streamlit uses 7860 as the external port.
# Use 'localhost' instead of 127.0.0.1 so inter-process requests resolve inside containers.
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="SRE Incident Mission Control", layout="wide")

# Minimal dark theme tweaks
st.markdown(
    """
    <style>
    .reportview-container { background: #0e1117; color: #c9d1d9; }
    .sidebar .sidebar-content { background: #0b0f14; }
    .stButton>button { background-color: #1f6feb; color: white; }
    .metric-label { color: #9aa4ad; }
    pre { background: #0b0f14; color: #d3d7de; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
st.sidebar.title("Mission Control")
task_id = st.sidebar.selectbox("Select Task ID", [1, 2, 3, 4], index=0)
if "last_reward" not in st.session_state:
    st.session_state.last_reward = 0.0
if "last_action" not in st.session_state:
    st.session_state.last_action = {"action_type": "nop"}

if st.sidebar.button("Reset Environment"):
    try:
        r = requests.post(f"{API_BASE}/reset", json={"task": task_id}, timeout=5)
        r.raise_for_status()
        payload = r.json()
        obs = payload.get("observation", {})
        st.session_state.last_reward = 0.0
        st.session_state.last_action = {"action_type": "nop"}
        st.sidebar.success(f"Environment reset to Task {task_id}")
    except Exception as e:
        st.sidebar.error(f"Reset failed: {e}")

st.sidebar.markdown("---")
st.sidebar.write("AI Controller")
if st.sidebar.button("Run AI Agent Step"):
    st.session_state.run_step = True

# --- Main layout ---
st.title("SRE-Incident-Gym — Mission Control")
st.caption("Live dashboard connected to the FastAPI environment at %s" % API_BASE)

col1, col2, col3 = st.columns([1, 1, 1])
status_card = col1.empty()
cpu_card = col2.empty()
reward_card = col3.empty()

console = st.container()
log_box = console.empty()

# Comparison Mode
comp_col = st.sidebar.container()
comp_col.markdown("**Performance Comparison**")
comp_df = pd.DataFrame({"Agent": ["GPT-4 Agent", "Random Baseline"], "Score": [0.53, 0.18]})
comp_df = comp_df.set_index("Agent")
comp_col.bar_chart(comp_df)

# Helper functions

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


# Local simple policy (used to drive /step when "Run AI Agent Step" is clicked)
def select_action_from_obs(obs_state: dict, step_count: int = 0) -> dict:
    # Check logs for schedule keywords
    last_log = obs_state.get("last_log_entry", "") or ""
    try:
        if isinstance(last_log, str) and ("scheduled" in last_log.lower() or "batch" in last_log.lower() or "window" in last_log.lower()):
            return {"action_type": "nop"}
    except Exception:
        pass

    status = obs_state.get("status_code", 200)
    cpu = obs_state.get("cpu_load", 0)
    patched = obs_state.get("is_patched", True)

    if status != 200:
        return {"action_type": "restart_service"}
    if not patched:
        return {"action_type": "apply_patch", "patch_id": obs_state.get("required_patch", "CVE-UNKNOWN")}
    if cpu and cpu > 60:
        # choose replicas proportional to CPU
        replicas = min(6, max(2, int(cpu / 20)))
        return {"action_type": "scale_up", "replicas": replicas}
    # Fallback
    return {"action_type": "check_logs"}


# Main update loop (on load and when user triggers a step)
state = get_state()
if state:
    s = state.get("state", {})
    status = s.get("status_code", "n/a")
    cpu = s.get("cpu_load", 0.0)
    reward_val = st.session_state.get("last_reward", 0.0)
    last_log = s.get("last_log_entry", "")

    # Top metrics
    status_card.metric("Status Code", value=str(status))
    cpu_card.metric("CPU Load", value=f"{cpu:.1f}%")
    reward_card.metric("Current Reward", value=f"{reward_val:.2f}")

    # Alerts
    if status == 500:
        st.error("CRITICAL: Service DOWN (500)")
    elif cpu > 80:
        st.warning("WARNING: High CPU > 80%")

    if s.get("done", False):
        st.success("SUCCESS — Task solved!")
        st.balloons()

    # Console
    log_box.code(last_log or "(no logs)")

# If user clicked Run AI Agent Step, perform one step using local policy and display results
if st.session_state.get("run_step", False):
    st.session_state.run_step = False
    # Re-fetch full state
    state = get_state()
    if state:
        obs = state.get("state", {})
        action = select_action_from_obs(obs)
        st.session_state.last_action = action
        with st.spinner("Running agent step..."):
            resp = post_step(action)
            time.sleep(0.2)
        if "error" in resp:
            st.error(f"Step failed: {resp['error']}")
        else:
            # Update last reward and display
            reward_val = resp.get("reward", {}).get("value", 0.0)
            st.session_state.last_reward = reward_val
            observation = resp.get("observation", {})
            last_log = observation.get("last_log_entry", "") if isinstance(observation, dict) else ""
            status = observation.get("status_code", status)
            cpu = observation.get("cpu_load", cpu)
            done = resp.get("done", False)

            # Refresh metrics
            status_card.metric("Status Code", value=str(status))
            cpu_card.metric("CPU Load", value=f"{cpu:.1f}%")
            reward_card.metric("Current Reward", value=f"{reward_val:.2f}")

            # Console
            log_box.code(last_log or "(no logs)")

            # Alerts / success
            if status == 500:
                st.error("CRITICAL: Service DOWN (500)")
            elif cpu > 80:
                st.warning("WARNING: High CPU > 80%")

            if done:
                st.success("SUCCESS — Task solved!")
                st.balloons()

# Footer
st.markdown("---")
st.markdown("**Tip:** Use the sidebar to change tasks, reset the environment, and step the AI agent. Refresh the page to reconnect.")
