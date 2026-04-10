"""
SRE-Incident-Gym Environment
OpenEnv-compliant environment for training incident response agents.
Implements reset(), step(), and state() methods with dense reward logic.
"""

from typing import Optional, Dict, Any, Tuple
from enum import Enum
import random
from models import Observation, Action, Reward, CheckLogsAction, RestartServiceAction, ScaleUpAction, ApplyPatchAction, NopAction


class TaskLevel(Enum):
    """Task difficulty levels."""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    SCHEDULED = 4


class SREIncidentGym:
    """
    SRE Incident Gymnasium Environment
    
    Task Definitions:
    - Task 1 (Easy): Server is DOWN (status_code=500)
    - Task 2 (Medium): CPU load is at 99%
    - Task 3 (Hard): CPU load is 99% AND is_patched is False
    
    Success Conditions:
    - Task 1: Reach status_code = 200
    - Task 2: Reduce cpu_load to <= 20%
    - Task 3: Reduce cpu_load to <= 20% AND is_patched = True
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the environment."""
        self.seed_value = seed
        if seed is not None:
            random.seed(seed)
        
        # Initialize with default keys to prevent KeyErrors in the API
        self._state: Dict[str, Any] = {
            "status_code": 200,
            "cpu_load": 10.0,
            "memory_usage": 15.0,
            "last_log_entry": "System healthy",
            "is_patched": False
        }
        # Match your TaskLevel enum (ensure it's TaskLevel.EASY or TaskLevel.LEVEL_1)
        self._task_level: TaskLevel = TaskLevel.EASY 
        self._step_count: int = 0
        self._max_steps: int = 15
        self._done: bool = False
        self._episode_reward: float = 0.0
        
    def reset(self, task: int = 1) -> Observation:
        """
        Reset the environment to initial state for a given task.
        
        Args:
            task: Task level (1=Easy, 2=Medium, 3=Hard)
            
        Returns:
            Initial Observation
        """
        self._step_count = 0
        self._done = False
        self._episode_reward = 0.0
        
        if task == 1:
            self._task_level = TaskLevel.EASY
            self._state = {
                "status_code": 500,  # Server DOWN
                "cpu_load": 45.0,
                "memory_usage": 60.0,
                "last_log_entry": "Internal Server Error. Inspecting stack trace... null pointer at handler.process()",
                "is_patched": True,
                "logs": [
                    "Internal Server Error: null pointer in handler.process()",
                    "Stacktrace: module.service -> handler.process -> memory access",
                    "Symptoms: delayed responses, intermittent 500s",
                ],
            }
        elif task == 2:
            self._task_level = TaskLevel.MEDIUM
            self._state = {
                "status_code": 200,
                # Randomize CPU and memory to introduce variability
                "cpu_load": random.uniform(70.0, 99.0),  # High CPU (randomized)
                "memory_usage": random.uniform(50.0, 90.0),
                "last_log_entry": "Internal alert: high CPU utilization detected; investigating workload and threadpool behavior.",
                "is_patched": True,
                "logs": [
                    "Alert: sustained high CPU utilization observed",
                    "Symptoms: increased latency and retry errors",
                    "Note: workload patterns may indicate batch activity or spike",
                ],
            }
        elif task == 3:
            self._task_level = TaskLevel.HARD
            self._state = {
                "status_code": 200,
                # Randomized starting load for robustness
                "cpu_load": random.uniform(70.0, 99.0),
                "memory_usage": random.uniform(60.0, 90.0),
                "last_log_entry": "Critical: multiple anomalies detected; security scan flagged potential vulnerability and CPU is elevated.",
                "is_patched": False,  # Unpatched
                "logs": [
                    "Security scan: potential CVE-XXXX detected",
                    "Performance: CPU spike observed during probe",
                    "Investigation: simultaneous signals from security and performance monitors",
                ],
            }
        elif task == 4:
            self._task_level = TaskLevel.SCHEDULED
            self._state = {
                "status_code": 200,
                "cpu_load": random.uniform(50.0, 70.0),
                "memory_usage": random.uniform(40.0, 60.0),
                "last_log_entry": "NOTICE: Scheduled batch window active; throughput spike expected.",
                "is_patched": True,
                "logs": ["Job scheduler: batch-run started", "Expected CPU spike: batch_job_id=42"],
                "scheduled_batch": True,
            }
        else:
            raise ValueError(f"Invalid task level: {task}. Must be 1, 2, 3, or 4.")
        
        return self._get_observation()
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Safety Shield: If episode already done, return early with zero reward
        if self._done:
            return self._get_observation(), Reward(value=0.0), True, {"info": "Task already completed"}

        self._step_count += 1
        # State validation: ensure required keys exist and are not None to avoid runtime errors
        defaults = {
            "status_code": 200,
            "cpu_load": 0.0,
            "memory_usage": 0.0,
            "last_log_entry": "",
            "is_patched": False,
            "logs": [],
        }
        for k, v in defaults.items():
            if self._state.get(k) is None:
                self._state[k] = v

        # Standardized reward engine
        # Step decay: punish wasted steps to favor efficient solutions
        step_penalty = -0.10
        reward_value = step_penalty
        info = {"step": self._step_count, "action_type": action.action_type}

        # Fixed completion bonus for any correct fix
        FIX_BONUS = 0.80
        WRONG_MOVE_PENALTY = -0.40

        # Execute action with standardized rewards
        if isinstance(action, CheckLogsAction):
            # Small positive for diagnostics
            reward_value += 0.05
            self._state["last_log_entry"] = self._state["logs"][0] if self._state["logs"] else "No new logs"
            info["action_info"] = "Logs retrieved - analysis shows root cause"

        elif isinstance(action, RestartServiceAction):
            # Correct when server is DOWN
            if int(self._state.get("status_code", 200)) == 500:
                self._state["status_code"] = 200
                reward_value += FIX_BONUS
                info["action_info"] = "Service restarted - correct fix"
            else:
                reward_value += WRONG_MOVE_PENALTY
                info["action_info"] = "Unnecessary restart on healthy service - wrong move"

        elif isinstance(action, ScaleUpAction):
            # Correct when CPU is very high
            cpu_before = float(self._state.get("cpu_load", 100.0))
            if cpu_before > 70.0:
                # Apply a deterministic improvement to reach healthy floor
                self._state["cpu_load"] = 20.0
                reward_value += FIX_BONUS
                info["action_info"] = f"Scale-up correct: CPU {cpu_before:.1f}% -> 20.0%"
            else:
                reward_value += WRONG_MOVE_PENALTY
                info["action_info"] = f"Over-provisioning penalty: CPU {cpu_before:.1f}% - wrong move"

        elif isinstance(action, ApplyPatchAction):
            # Correct when task requires patching (HARD)
            is_patched = bool(self._state.get("is_patched", False))
            if (self._task_level == TaskLevel.HARD) and (not is_patched):
                self._state["is_patched"] = True
                reward_value += FIX_BONUS
                info["action_info"] = f"Patch {action.patch_id} applied - correct fix"
                info["patch_id"] = action.patch_id
            else:
                reward_value += WRONG_MOVE_PENALTY
                info["action_info"] = "Unnecessary patch or not required - wrong move"

        elif isinstance(action, NopAction):
            # No-op: correct only for scheduled window
            if self._task_level == TaskLevel.SCHEDULED:
                reward_value += FIX_BONUS
                info["action_info"] = "No-op during scheduled window - correct"
                # Finish episode immediately for scheduled success
                observation = self._get_observation()
                self._done = True
                final_reward = min(1.0, max(-1.0, reward_value))
                self._episode_reward += final_reward
                return observation, Reward(value=final_reward), True, info
            else:
                # Discourage needless inaction
                reward_value += -0.10
                if random.random() < 0.3:
                    self._state["cpu_load"] = min(100.0, self._state.get("cpu_load", 0.0) + 5.0)
                info["action_info"] = "No-op taken - likely wrong unless scheduled"

        # Special handling for scheduled batch task: correct action is NOP
        if self._task_level == TaskLevel.SCHEDULED:
            if isinstance(action, NopAction):
                # Reward correct no-op and finish episode
                observation = self._get_observation()
                info["action_info"] = "Scheduled window - nop was correct"
                self._done = True
                self._episode_reward += 1.0
                return observation, Reward(value=1.0), True, info
            else:
                # Incorrect mitigation - penalize and finish
                observation = self._get_observation()
                info["action_info"] = "Unnecessary mitigation during scheduled job"
                self._done = True
                self._episode_reward += -0.2
                return observation, Reward(value=-0.2), True, info

        # Immediate success check: if the action just solved the task, return immediately
        # Extra explicit zero-latency checks to avoid any 'bleed' when actions clearly satisfy success
        cpu_now = 0.0
        try:
            cpu_now = float(self._state.get("cpu_load", 100.0))
        except Exception:
            cpu_now = 100.0
        patched_now = bool(self._state.get("is_patched", False))

        just_solved = False
        # Primary check: general task solved predicate
        if self._check_task_solved():
            just_solved = True

        # Targeted checks to ensure immediate termination when a ScaleUp or ApplyPatch clearly finishes the task
        if isinstance(action, ScaleUpAction):
            if (self._task_level == TaskLevel.MEDIUM and cpu_now <= 50.0) or (
                self._task_level == TaskLevel.HARD and patched_now and cpu_now <= 50.0
            ):
                just_solved = True

        if isinstance(action, ApplyPatchAction):
            # If applying a patch completes Task 3 (patch + already-low-cpu), finish immediately
            if self._task_level == TaskLevel.HARD and patched_now and cpu_now <= 50.0:
                just_solved = True

        if isinstance(action, RestartServiceAction):
            if self._task_level == TaskLevel.EASY and int(self._state.get("status_code", 0)) == 200:
                just_solved = True

        if just_solved:
            # Mark done and craft the clear success observation/message
            self._done = True
            self._state["last_log_entry"] = "TASK COMPLETE. System is healthy. Stop all actions."

            # Final reward is based on accumulated reward_value (no artificial top-up)
            final_reward = min(1.0, max(-1.0, reward_value))
            self._episode_reward += final_reward

            observation = self._get_observation()
            info["action_info"] = info.get("action_info", "") + " ✓ TASK COMPLETED!"
            return observation, Reward(value=final_reward), True, info

        # If not solved immediately, apply step penalty as usual
        # Accumulate episode reward
        self._episode_reward += reward_value

        # If the environment now satisfies the success predicate, mark done and annotate
        task_solved = self._check_task_solved()
        if task_solved:
            self._done = True
            if self._task_level == TaskLevel.EASY:
                self._state["last_log_entry"] = "SUCCESS: Service recovered and stable. Task complete."
            elif self._task_level == TaskLevel.MEDIUM:
                self._state["last_log_entry"] = f"SUCCESS: CPU reduced to {self._state.get('cpu_load', 0):.1f}%. Task complete."
            elif self._task_level == TaskLevel.HARD:
                self._state["last_log_entry"] = f"SUCCESS: CPU reduced to {self._state.get('cpu_load', 0):.1f}% and patch applied. Task complete."

            info["action_info"] = info.get("action_info", "") + " ✓ TASK COMPLETED!"
        
        # Check if max steps reached (increased to 30 for more chances, but agent should finish much sooner)
        if self._step_count >= self._max_steps:
            self._done = True
            info["action_info"] = info.get("action_info", "") + " [MAX STEPS REACHED - EPISODE TERMINATED]"
        
        observation = self._get_observation()
        reward = Reward(value=min(1.0, max(-1.0, reward_value)))  # Clamp to [-1.0, 1.0]
        
        return observation, reward, self._done, info
    
    def state(self) -> Dict[str, Any]:
        """
        Return the full internal state dictionary safely.
        """
        # 1. Safely check if task is solved
        try:
            solved = self._check_task_solved()
        except Exception:
            solved = False

        # 2. Use .get(key, default) to prevent KeyErrors
        return {
            "status_code": self._state.get("status_code", 200),
            "cpu_load": self._state.get("cpu_load", 0.0),
            "memory_usage": self._state.get("memory_usage", 0.0),
            "last_log_entry": self._state.get("last_log_entry", "System Initializing..."),
            "is_patched": self._state.get("is_patched", False),
            "step_count": self._step_count,
            "done": bool(self._done or solved),
            "episode_reward": self._episode_reward,
            # Handle task_level name safely
            "task_level": getattr(self._task_level, 'name', str(self._task_level)),
        }
    
    def _get_observation(self) -> Observation:
        """Get the current observation."""
        return Observation(
            status_code=int(self._state["status_code"]),
            cpu_load=float(self._state["cpu_load"]),
            memory_usage=float(self._state["memory_usage"]),
            last_log_entry=str(self._state["last_log_entry"]),
            is_patched=bool(self._state["is_patched"]),
        )
    
    def _check_task_solved(self) -> bool:
        """Check if the current task has been solved."""
        if self._task_level == TaskLevel.EASY:
            # Task 1: status_code must be 200
            return int(self._state.get("status_code", 0)) == 200
        
        elif self._task_level == TaskLevel.MEDIUM:
            # Task 2: require CPU <= 20% to be considered solved (realistic threshold)
            try:
                cpu = float(self._state.get("cpu_load", 100.0))
            except Exception:
                cpu = 100.0
            return cpu <= 20.0
        
        elif self._task_level == TaskLevel.HARD:
            # Task 3: cpu_load <= 20% AND is_patched == True
            # Be defensive: coerce types and ensure both conditions are checked explicitly
            try:
                cpu = float(self._state.get("cpu_load", 100.0))
            except Exception:
                cpu = 100.0
            is_patched = bool(self._state.get("is_patched", False))
            # For Task 3, require the patch and CPU <= 20% (realistic success)
            return is_patched and (cpu <= 20.0)
        
        return False
    
    def get_task_description(self) -> str:
        """Get a description of the current task."""
        descriptions = {
            TaskLevel.EASY: "Task 1 (Easy): Restart the service to bring status from 500 to 200. Use 'restart_service' action.",
            TaskLevel.MEDIUM: "Task 2 (Medium): Scale up to reduce CPU load from 99% to ≤20%. Use 'scale_up' action with 4-5 replicas.",
            TaskLevel.HARD: "Task 3 (Hard): BOTH (1) Reduce CPU to ≤20% using 'scale_up' (4-5 replicas) AND (2) Apply security patch using 'apply_patch'. Both required!",
            TaskLevel.SCHEDULED: "Task 4 (Scheduled): Scheduled batch window — do nothing (use 'nop' action) and allow the batch to complete.",
        }
        return descriptions[self._task_level]
