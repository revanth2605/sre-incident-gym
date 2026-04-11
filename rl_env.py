"""
rl_env.py — Gymnasium wrapper around SREIncidentGym for RL training.

Wraps the existing environment.py into a proper gymnasium.Env so that
stable-baselines3 (PPO, A2C, DQN etc.) can train on it directly.

Observation vector (6 floats, all normalised to [0,1]):
  [status_ok, cpu_norm, mem_norm, is_patched, task_id_norm, step_norm]

Action space (Discrete 5):
  0 = check_logs
  1 = restart_service
  2 = scale_up  (always uses 5 replicas)
  3 = apply_patch
  4 = nop
"""

import sys
import os
# Ensure the app directory is always on sys.path regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment import SREIncidentGym
from models import (
    CheckLogsAction, RestartServiceAction,
    ScaleUpAction, ApplyPatchAction, NopAction,
)

# Maps integer action → Action object
ACTION_MAP = {
    0: CheckLogsAction(),
    1: RestartServiceAction(),
    2: ScaleUpAction(replicas=5),
    3: ApplyPatchAction(patch_id="CVE-2026-SRE-FIX"),
    4: NopAction(),
}

ACTION_NAMES = {
    0: "check_logs",
    1: "restart_service",
    2: "scale_up",
    3: "apply_patch",
    4: "nop",
}


class SREGymEnv(gym.Env):
    """
    Gymnasium-compatible SRE Incident environment.

    Each episode is a single task (randomly sampled from 1-4 unless
    task_id is fixed). The agent receives a normalised float observation
    and a scalar reward from the existing reward engine in environment.py.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, task_id: int = None, seed: int = None):
        super().__init__()

        # task_id=None → randomise each episode (better generalisation)
        self.task_id   = task_id
        self._seed     = seed
        self._gym      = SREIncidentGym(seed=seed)
        self._current_task = task_id or 1
        self._step_count   = 0
        self._max_steps    = 15

        # ── spaces ──────────────────────────────────────────────────
        # Obs: [status_ok, cpu_norm, mem_norm, is_patched, task_norm, step_norm]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        # 5 discrete actions
        self.action_space = spaces.Discrete(5)

    # ----------------------------------------------------------------
    # reset
    # ----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise task each episode if no fixed task_id
        if self.task_id is None:
            self._current_task = np.random.randint(1, 5)  # 1..4
        else:
            self._current_task = self.task_id

        self._step_count = 0
        obs = self._gym.reset(task=self._current_task)
        return self._encode(obs), {}

    # ----------------------------------------------------------------
    # step
    # ----------------------------------------------------------------

    def step(self, action: int):
        action_obj = ACTION_MAP[int(action)]
        obs, reward, done, info = self._gym.step(action_obj)

        self._step_count += 1
        truncated = self._step_count >= self._max_steps

        return self._encode(obs), float(reward.value), done, truncated, info

    # ----------------------------------------------------------------
    # render
    # ----------------------------------------------------------------

    def render(self):
        s = self._gym._state
        print(
            f"Task={self._current_task} | "
            f"Step={self._step_count} | "
            f"Status={s.get('status_code')} | "
            f"CPU={s.get('cpu_load'):.1f}% | "
            f"Patched={s.get('is_patched')}"
        )

    # ----------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------

    def _encode(self, obs) -> np.ndarray:
        """Encode Observation into a normalised float32 vector."""
        return np.array([
            1.0 if obs.status_code == 200 else 0.0,   # status OK?
            obs.cpu_load    / 100.0,                   # cpu  [0,1]
            obs.memory_usage / 100.0,                  # mem  [0,1]
            1.0 if obs.is_patched else 0.0,            # patched?
            (self._current_task - 1) / 3.0,           # task [0,1]
            self._step_count / self._max_steps,        # progress [0,1]
        ], dtype=np.float32)