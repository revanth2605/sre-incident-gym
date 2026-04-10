"""
SRE-Incident-Gym FastAPI Server
Exposes the environment as a web API on port 8000 (internal API port).
Endpoints: /reset, /step, /state, /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import json
import traceback
from contextlib import asynccontextmanager

from models import Observation, Action, Reward, CheckLogsAction, RestartServiceAction, ScaleUpAction, ApplyPatchAction, NopAction
from environment import SREIncidentGym


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ResetRequest(BaseModel):
    """Request model for /reset endpoint."""
    task: int = Field(default=1, description="Task level (1, 2, 3, or 4)", ge=1, le=4)


class ResetResponse(BaseModel):
    """Response model for /reset endpoint."""
    observation: Observation
    task_description: str


class StepRequest(BaseModel):
    """Request model for /step endpoint."""
    action: Action = Field(..., description="Action to execute")


class StepResponse(BaseModel):
    """Response model for /step endpoint."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    """Response model for /state endpoint."""
    state: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    version: str


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler to initialize and cleanup resources."""
    global env
    # Use None seed to enable randomized resets for robustness testing
    env = SREIncidentGym(seed=None)
    try:
        yield
    finally:
        # place for cleanup if needed
        env = None

# Global environment instance (initialized in lifespan)
env: Optional[SREIncidentGym] = None

# Create FastAPI app with modern lifespan handler
app = FastAPI(
    title="SRE-Incident-Gym",
    description="OpenEnv-compliant SRE incident response training environment",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest):
    """
    Reset the environment to initial state for a given task.
    
    Args:
        request: ResetRequest with task level (1, 2, or 3)
        
    Returns:
        ResetResponse with initial observation and task description
    """
    global env
    try:
        observation = env.reset(task=request.task)
        task_description = env.get_task_description()
        return ResetResponse(
            observation=observation,
            task_description=task_description
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """
    Execute one step in the environment.
    
    Args:
        request: StepRequest with action
        
    Returns:
        StepResponse with observation, reward, done flag, and info
    """
    global env
    try:
        # Parse action - it comes as a discriminated union
        action_dict = request.action.model_dump()
        action_type = action_dict.get("action_type")
        
        # Reconstruct the proper action object
        if action_type == "check_logs":
            action = CheckLogsAction()
        elif action_type == "restart_service":
            action = RestartServiceAction()
        elif action_type == "scale_up":
            replicas = action_dict.get("replicas", 1)
            action = ScaleUpAction(replicas=replicas)
        elif action_type == "apply_patch":
            patch_id = action_dict.get("patch_id", "default")
            action = ApplyPatchAction(patch_id=patch_id)
        elif action_type == "nop":
            action = NopAction()
        else:
            raise ValueError(f"Unknown action type: {action_type}")
        
        observation, reward, done, info = env.step(action)
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info
        )
    except RuntimeError as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Print full traceback to console to aid debugging of 500 errors
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state", response_model=StateResponse)
async def state():
    global env
    
    # 1. Safety check: Is the environment initialized?
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized. Please call /reset first.")
    
    try:
        # 2. Try to find the state method (handles different naming styles)
        if hasattr(env, 'state'):
            current_state = env.state()
        elif hasattr(env, 'get_state'):
            current_state = env.get_state()
        else:
            # Fallback: manually return the internal dict if possible
            current_state = getattr(env, '_state', {"status": "initialized"})
            
        return StateResponse(state=current_state)
        
    except Exception as e:
        # This will print the actual error to your Hugging Face Logs
        print(f"CRITICAL ERROR in /state: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with status and version
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "name": "SRE-Incident-Gym",
        "version": "1.0.0",
        "description": "OpenEnv-compliant SRE incident response training environment",
        "endpoints": {
            "POST /reset": "Reset environment to initial state",
            "POST /step": "Execute one step with an action",
            "GET /state": "Get full environment state",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation (Swagger UI)",
        },
        "tasks": {
            "1": "Easy - Restart service (status 500 → 200)",
            "2": "Medium - Scale up (reduce CPU 99% → ≤20%)",
            "3": "Hard - Scale up AND apply patch (CPU ≤20% AND patched)",
            "4": "Scheduled - No-op during scheduled batch window (do nothing)",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
