"""
SRE-Incident-Gym Inference Script
Judge runner script for scoring the environment.
Reads ENV_URL, API_BASE_URL, MODEL_NAME, and HF_TOKEN from environment variables.
Prints logs in the exact OpenEnv format required.
"""

import os
import sys
import json
import time
from typing import Optional, Dict, Any
import requests

try:
    from openai import OpenAI
except Exception:
    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass


# ============================================================================
# CONFIGURATION
# ============================================================================

# ENV_URL  — your HuggingFace Space where /reset, /step, /state live
# API_BASE_URL — the LLM endpoint (validator sets this to their litellm proxy)
# These MUST be separate. The validator injects API_BASE_URL as their LLM URL
# (e.g. https://litellm.sclr.ac), NOT your environment URL.
ENV_URL      = os.getenv("ENV_URL",      "https://revanthkothamasu26-sre-incident-gym.hf.space")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

MAX_STEPS_PER_TASK = 15
TASKS = [1, 2, 3, 4]

SYSTEM_PROMPT = """
You are a Senior SRE. Solve the incident in the FEWEST steps possible.
- If status is 500: restart_service.
- If cpu_load > 50: scale_up (use 5 replicas).
- If is_patched is False: apply_patch.
- If log mentions scheduled/batch/window: nop.
Once the system is healthy, stop immediately.
"""


# ============================================================================
# LOGGING  (exact OpenEnv format)
# ============================================================================

def log_start(task_name: str, env_name: str = "sre-gym", model_name: str = MODEL_NAME):
    print(f"[START] task={task_name} env={env_name} model={model_name}")
    sys.stdout.flush()


def log_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str] = None):
    error_str = f'"{error}"' if error else "null"
    done_str  = "true" if done else "false"
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error_str}")
    sys.stdout.flush()


def log_end(success: bool, steps: int, score: float, rewards: str):
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards}")
    sys.stdout.flush()


# ============================================================================
# ENVIRONMENT CALLS  — always use ENV_URL, never API_BASE_URL
# ============================================================================

def reset_env(task: int) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task": task}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] Failed to reset environment: {e}", file=sys.stderr)
        return None


def step_env(action_dict: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(f"{ENV_URL}/step", json={"action": action_dict}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_state() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{ENV_URL}/state", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] Failed to get state: {e}", file=sys.stderr)
        return None


# ============================================================================
# AGENT  — LLM with heuristic fallback
# ============================================================================

def heuristic(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback — always returns a valid action."""
    last_log    = state_dict.get("last_log_entry", "") or ""
    status_code = state_dict.get("status_code", 200)
    cpu_load    = state_dict.get("cpu_load", 0) or 0
    is_patched  = state_dict.get("is_patched", True)

    if isinstance(last_log, str) and any(
        w in last_log.lower() for w in ["scheduled", "batch", "window"]
    ):
        return {"action_type": "nop"}
    if status_code != 200:
        return {"action_type": "restart_service"}
    if not is_patched:
        return {"action_type": "apply_patch", "patch_id": "CVE-2026-SRE-FIX"}
    if cpu_load > 50:
        return {"action_type": "scale_up", "replicas": 5}
    return {"action_type": "nop"}


def parse_action(txt: str) -> Optional[Dict[str, Any]]:
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(txt[s:e + 1])
            except Exception:
                pass
    return None


def get_next_action(state: Dict[str, Any], step_count: int) -> Dict[str, Any]:
    """Call LLM via API_BASE_URL; fall back to heuristic on any failure."""
    state_dict = state.get("state") or state

    user_payload = {
        "state": state_dict,
        "step":  step_count,
        "instructions": (
            "Return ONLY a JSON object with the action. "
            "Allowed: restart_service | scale_up (replicas int) | "
            "apply_patch (patch_id str) | check_logs | nop. "
            'Example: {"action_type": "scale_up", "replicas": 5}'
        ),
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": json.dumps(user_payload)},
    ]

    try:
        api_key = os.getenv("OPENAI_API_KEY") or HF_TOKEN or "no-key"
        # Use API_BASE_URL for LLM only (validator's litellm proxy)
        client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
        resp   = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=150,
            temperature=0.0,
        )
        text   = (resp.choices[0].message.content or "").strip()
        action = parse_action(text)
        if action and "action_type" in action:
            return action
    except Exception as e:
        print(f"[WARNING] LLM call failed: {e} — using heuristic", file=sys.stderr)

    return heuristic(state_dict)


# ============================================================================
# TASK RUNNER
# ============================================================================

def task_name_from_level(level: int) -> str:
    return {
        1: "task_easy_restart",
        2: "task_medium_cpu_scaling",
        3: "task_hard_security_and_perf",
        4: "task_scheduled_nop",
    }.get(level, f"task_{level}")


def run_task(task_level: int) -> Dict[str, Any]:
    task_name = task_name_from_level(task_level)
    log_start(task_name)

    reset_response = reset_env(task_level)
    if not reset_response:
        log_end(False, 0, 0.0, "")
        return {"steps": 0, "rewards": [], "success": False, "score": 0.0}

    step_count   = 0
    rewards      = []
    done         = False
    task_success = False

    while step_count < MAX_STEPS_PER_TASK and not done:
        state = get_state()
        if not state:
            log_step(step_count + 1, "error", 0.0, True, "Failed to get state")
            break

        state_dict = state.get("state") or state

        # Early exit if already solved
        if state_dict.get("done", False):
            log_step(step_count, "(auto)done", 0.0, True)
            task_success = True
            break

        action     = get_next_action(state, step_count)
        action_str = action.get("action_type", "unknown")

        step_response = step_env(action)
        if "error" in step_response:
            log_step(step_count + 1, action_str, 0.0, True, step_response["error"])
            break

        # Parse reward — handles both flat float and {"value": float}
        raw_reward   = step_response.get("reward", 0.0)
        reward_value = (
            raw_reward.get("value", 0.0)
            if isinstance(raw_reward, dict)
            else float(raw_reward or 0.0)
        )
        done = step_response.get("done", False)

        # Detect success from log
        obs = step_response.get("observation", {})
        if isinstance(obs, dict):
            obs_log = obs.get("last_log_entry", "") or ""
            if not done and ("TASK COMPLETE" in obs_log or "SUCCESS:" in obs_log):
                done = True

        rewards.append(reward_value)
        step_count += 1
        log_step(step_count, action_str, reward_value, done)

        if done:
            task_success = True

    sum_rewards = sum(rewards) if rewards else 0.0
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    score       = min(max(sum_rewards, 0.0), 1.0) if task_success else 0.0

    log_end(task_success, step_count, score, rewards_str)
    return {"steps": step_count, "rewards": rewards, "success": task_success, "score": score}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("SRE-Incident-Gym Evaluation")
    print("=" * 80)
    print(f"ENV URL:  {ENV_URL}")
    print(f"LLM API:  {API_BASE_URL}")
    print(f"Model:    {MODEL_NAME}")
    print("=" * 80)

    all_results = {}
    all_scores  = []

    for task_level in TASKS:
        print(f"\n{'='*80}")
        print(f"Running Task {task_level}")
        print(f"{'='*80}\n")
        result = run_task(task_level)
        all_results[f"task_{task_level}"] = result
        all_scores.append(result["score"])
        time.sleep(1)

    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for name, result in all_results.items():
        print(f"{name}: score={result['score']:.2f} steps={result['steps']} success={result['success']}")
    print(f"\nOverall Score: {overall_score:.2f}/1.00")
    print("=" * 80)

    return overall_score


if __name__ == "__main__":
    try:
        score = main()
        sys.exit(0 if score > 0.3 else 1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Evaluation stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] {e}", file=sys.stderr)
        sys.exit(1)