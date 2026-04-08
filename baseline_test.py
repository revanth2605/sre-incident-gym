"""
Random Baseline Test for SRE-Incident-Gym
Runs tasks 1-4 using a random policy to demonstrate baseline performance.
"""

import os
import sys
import time
import random
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MAX_STEPS = 15
TASKS = [1, 2, 3, 4]

ACTION_TYPES = ["nop", "check_logs", "restart_service", "apply_patch", "scale_up"]


def reset_env(task: int):
    try:
        r = requests.post(f"{API_BASE_URL}/reset", json={"task": task}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Reset failed: {e}")
        return None


def step_env(action: dict):
    try:
        r = requests.post(f"{API_BASE_URL}/step", json={"action": action}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def random_action():
    at = random.choice(ACTION_TYPES)
    if at == "scale_up":
        return {"action_type": "scale_up", "replicas": random.randint(1, 8)}
    if at == "apply_patch":
        return {"action_type": "apply_patch", "patch_id": f"patch-{random.randint(1,100)}"}
    return {"action_type": at}


def run_task(task):
    print(f"Running random baseline for task {task}")
    res = reset_env(task)
    if not res:
        return {"success": False, "score": 0.0, "steps": 0}

    steps = 0
    rewards = []
    done = False
    while steps < MAX_STEPS and not done:
        action = random_action()
        resp = step_env(action)
        if "error" in resp:
            print("Step error:", resp["error"])
            break
        reward = resp.get("reward", {}).get("value", 0.0)
        done = resp.get("done", False)
        rewards.append(reward)
        steps += 1
    score = min(max(sum(rewards), 0.0), 1.0)
    print(f"Task {task} done. steps={steps} score={score:.2f}")
    return {"success": done, "score": score, "steps": steps}


if __name__ == '__main__':
    results = {}
    scores = []
    for t in TASKS:
        r = run_task(t)
        results[f"task_{t}"] = r
        scores.append(r["score"])
        time.sleep(0.5)

    overall = sum(scores) / len(scores) if scores else 0.0
    print("\nBaseline Summary:")
    for k, v in results.items():
        print(f"{k}: score={v['score']:.2f} steps={v['steps']} success={v['success']}")
    print(f"Overall baseline score: {overall:.2f}")
