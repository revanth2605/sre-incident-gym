"""
rl_train.py — Train a PPO agent on SRE-Incident-Gym.

Usage:
    python rl_train.py              # train all tasks (random per episode)
    python rl_train.py --task 1     # train on a single task only
    python rl_train.py --eval       # evaluate saved model

The trained model is saved to models/ppo_sre_agent.zip
A training curve is saved to models/training_curve.png
"""

import os
import sys
import argparse
import numpy as np

# Ensure app directory is on path so environment.py / models.py are found
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── optional matplotlib (graceful fallback if not installed) ────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from rl_env import SREGymEnv, ACTION_NAMES

os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/ppo_sre_agent"


# ============================================================================
# Reward-tracking callback
# ============================================================================

class RewardLoggerCallback(BaseCallback):
    """Logs mean episode reward every N steps and saves training curve."""

    def __init__(self, log_interval: int = 500, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval  = log_interval
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int]   = []
        self._ep_reward = 0.0
        self._ep_len    = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]

        self._ep_reward += reward
        self._ep_len    += 1

        if done:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_len)
            self._ep_reward = 0.0
            self._ep_len    = 0

            if self.verbose and len(self.episode_rewards) % 20 == 0:
                mean_r = np.mean(self.episode_rewards[-20:])
                mean_l = np.mean(self.episode_lengths[-20:])
                print(
                    f"  Episode {len(self.episode_rewards):4d} | "
                    f"mean_reward (last 20): {mean_r:+.3f} | "
                    f"mean_steps: {mean_l:.1f}"
                )

        # Save curve every log_interval steps
        if self.num_timesteps % self.log_interval == 0 and HAS_PLOT:
            self._save_curve()

        return True

    def _save_curve(self):
        if len(self.episode_rewards) < 2:
            return
        plt.figure(figsize=(10, 4))
        plt.plot(self.episode_rewards, alpha=0.3, color="steelblue", label="Episode reward")
        # Smoothed
        window = min(50, len(self.episode_rewards))
        smoothed = np.convolve(
            self.episode_rewards,
            np.ones(window) / window,
            mode="valid"
        )
        plt.plot(range(window - 1, len(self.episode_rewards)), smoothed,
                 color="steelblue", linewidth=2, label=f"Smoothed ({window}ep)")
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("PPO Training on SRE-Incident-Gym")
        plt.legend()
        plt.tight_layout()
        plt.savefig("models/training_curve.png", dpi=100)
        plt.close()


# ============================================================================
# Training
# ============================================================================

def train(task_id: int = None, total_timesteps: int = 50_000):
    """Train a PPO agent. task_id=None trains across all tasks."""

    task_label = f"task-{task_id}" if task_id else "all-tasks"
    print(f"\n{'='*60}")
    print(f"  Training PPO on SRE-Incident-Gym  [{task_label}]")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"{'='*60}\n")

    # Vectorised env (4 parallel workers for speed)
    def make_env():
        env = SREGymEnv(task_id=task_id)
        return Monitor(env)

    vec_env = make_vec_env(make_env, n_envs=4)

    # PPO hyperparameters tuned for this small env
    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = 3e-4,
        n_steps         = 256,        # steps per env per update
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,       # discount factor
        gae_lambda      = 0.95,       # GAE lambda
        clip_range      = 0.2,        # PPO clip
        ent_coef        = 0.01,       # entropy bonus (explore)
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = dict(net_arch=[64, 64]),  # 2-layer MLP
        verbose         = 0,
    )

    callback = RewardLoggerCallback(log_interval=500, verbose=1)

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(MODEL_PATH)

    if HAS_PLOT:
        callback._save_curve()
        print(f"\n  Training curve saved → models/training_curve.png")

    print(f"\n  Model saved → {MODEL_PATH}.zip")
    return model, callback.episode_rewards


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model_path: str = MODEL_PATH, n_episodes: int = 20):
    """Evaluate a saved PPO model across all 4 tasks."""

    print(f"\n{'='*60}")
    print(f"  Evaluating PPO agent: {model_path}.zip")
    print(f"{'='*60}\n")

    model = PPO.load(model_path)

    results = {}
    for task in [1, 2, 3, 4]:
        env      = SREGymEnv(task_id=task)
        rewards  = []
        successes = 0

        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = truncated = False

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(int(action))
                ep_reward += reward
                if env.verbose if hasattr(env, 'verbose') else False:
                    env.render()

            rewards.append(ep_reward)
            if ep_reward > 0:
                successes += 1

        mean_r   = np.mean(rewards)
        success_rate = successes / n_episodes * 100
        results[task] = {"mean_reward": mean_r, "success_rate": success_rate}

        print(
            f"  Task {task} | mean_reward: {mean_r:+.3f} | "
            f"success_rate: {success_rate:.0f}%  ({successes}/{n_episodes})"
        )

    overall = np.mean([v["mean_reward"] for v in results.values()])
    print(f"\n  Overall mean reward: {overall:+.3f}")
    print(f"{'='*60}\n")
    return results


# ============================================================================
# Compare RL vs random baseline
# ============================================================================

def compare_vs_random(model_path: str = MODEL_PATH, n_episodes: int = 20):
    """Compare trained PPO against a random policy."""

    print(f"\n{'='*60}")
    print("  PPO vs Random Baseline Comparison")
    print(f"{'='*60}\n")

    model = PPO.load(model_path)

    for task in [1, 2, 3, 4]:
        ppo_rewards    = []
        random_rewards = []

        for _ in range(n_episodes):
            # PPO
            env = SREGymEnv(task_id=task)
            obs, _ = env.reset()
            ep_r = 0.0
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, _ = env.step(int(action))
                ep_r += r
            ppo_rewards.append(ep_r)

            # Random
            env2 = SREGymEnv(task_id=task)
            obs2, _ = env2.reset()
            ep_r2 = 0.0
            done2 = truncated2 = False
            while not (done2 or truncated2):
                action2 = env2.action_space.sample()
                obs2, r2, done2, truncated2, _ = env2.step(action2)
                ep_r2 += r2
            random_rewards.append(ep_r2)

        ppo_mean    = np.mean(ppo_rewards)
        random_mean = np.mean(random_rewards)
        lift        = ppo_mean - random_mean

        print(
            f"  Task {task} | PPO: {ppo_mean:+.3f} | "
            f"Random: {random_mean:+.3f} | Lift: {lift:+.3f}"
        )

    print(f"{'='*60}\n")


# ============================================================================
# Single-episode demo (shows action-by-action decisions)
# ============================================================================

def demo(model_path: str = MODEL_PATH, task_id: int = 1):
    """Run one episode and print every action + reward."""

    model = PPO.load(model_path)
    env   = SREGymEnv(task_id=task_id)
    obs, _ = env.reset()

    print(f"\n{'='*60}")
    print(f"  PPO Agent Demo  —  Task {task_id}")
    print(f"{'='*60}")
    env.render()
    print()

    total = 0.0
    done = truncated = False
    step = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total += reward
        step  += 1
        print(
            f"  Step {step:2d} | action: {ACTION_NAMES[int(action)]:18s} | "
            f"reward: {reward:+.2f} | done: {done}"
        )
        env.render()

    print(f"\n  Episode finished — total reward: {total:+.3f}")
    print(f"{'='*60}\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate PPO on SRE-Incident-Gym")
    parser.add_argument("--task",       type=int, default=None,
                        help="Fix task id 1-4 (default: random each episode)")
    parser.add_argument("--timesteps",  type=int, default=50_000,
                        help="Total training timesteps (default: 50000)")
    parser.add_argument("--eval",       action="store_true",
                        help="Evaluate saved model instead of training")
    parser.add_argument("--demo",       action="store_true",
                        help="Run single episode demo")
    parser.add_argument("--compare",    action="store_true",
                        help="Compare PPO vs random baseline")
    args = parser.parse_args()

    if args.eval:
        evaluate()
    elif args.demo:
        demo(task_id=args.task or 1)
    elif args.compare:
        compare_vs_random()
    else:
        model, rewards = train(task_id=args.task, total_timesteps=args.timesteps)
        print("\nRunning post-training evaluation...")
        evaluate()
        compare_vs_random()
        demo(task_id=1)