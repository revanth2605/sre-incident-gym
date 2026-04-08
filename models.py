"""
SRE-Incident-Gym OpenEnv Models
Strict Pydantic V2 models for the Meta OpenEnv Hackathon project.
Defines the environment's observation, action, and reward structures.
"""

from typing import Literal, Union
from pydantic import BaseModel, Field


# ============================================================================
# OBSERVATION MODEL
# ============================================================================

class Observation(BaseModel):
    """
    Observation state from the SRE incident environment.
    
    Represents the current system state that the agent observes at each step.
    """
    
    status_code: int = Field(
        ...,
        description="HTTP status code of the service",
        ge=100,
        le=599
    )
    cpu_load: float = Field(
        ...,
        description="CPU load percentage (0-100)",
        ge=0.0,
        le=100.0
    )
    memory_usage: float = Field(
        ...,
        description="Memory usage percentage (0-100)",
        ge=0.0,
        le=100.0
    )
    last_log_entry: str = Field(
        ...,
        description="The most recent log entry from the service"
    )
    is_patched: bool = Field(
        ...,
        description="Whether the current system has the security patch applied"
    )
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        use_enum_values = True


# ============================================================================
# ACTION MODELS (Discriminated Union)
# ============================================================================

class CheckLogsAction(BaseModel):
    """Retrieve and analyze service logs."""
    
    action_type: Literal["check_logs"] = "check_logs"
    
    class Config:
        description = "Fetch and inspect the latest logs from the service"


class RestartServiceAction(BaseModel):
    """Restart the microservice."""
    
    action_type: Literal["restart_service"] = "restart_service"
    
    class Config:
        description = "Restart the service to recover from transient failures"


class ScaleUpAction(BaseModel):
    """Scale up the service replicas."""
    
    action_type: Literal["scale_up"] = "scale_up"
    replicas: int = Field(
        default=1,
        description="Number of additional replicas to provision",
        ge=1,
        le=10
    )
    
    class Config:
        description = "Horizontally scale the service to handle increased load"


class ApplyPatchAction(BaseModel):
    """Apply a security or bug fix patch."""
    
    action_type: Literal["apply_patch"] = "apply_patch"
    patch_id: str = Field(
        ...,
        description="Identifier of the patch to apply"
    )
    
    class Config:
        description = "Deploy a patch to fix vulnerabilities or bugs"


class NopAction(BaseModel):
    """No operation - do nothing."""
    
    action_type: Literal["nop"] = "nop"
    
    class Config:
        description = "Take no action and observe the environment's behavior"


# Discriminated Union of all possible actions
Action = Union[
    CheckLogsAction,
    RestartServiceAction,
    ScaleUpAction,
    ApplyPatchAction,
    NopAction
]


# ============================================================================
# REWARD MODEL
# ============================================================================

class Reward(BaseModel):
    """
    Reward signal from the SRE incident environment.
    
    A single scalar value indicating the quality of the agent's action.
    Higher values indicate better outcomes.
    """
    
    value: float = Field(
        ...,
        description="Reward value in the range [-1.0, 1.0] (penalties allowed)",
        ge=-1.0,
        le=1.0
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
