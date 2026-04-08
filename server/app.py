"""
FastAPI HTTP Server for the Sprint Task Scheduler OpenEnv
==========================================================
Uses openenv-core's ``create_fastapi_app`` to expose the standard
OpenEnv API (``/reset``, ``/step``, ``/state``, ``/health``, ``/schema``,
``/metadata``, ``/ws``) and adds custom endpoints for task listing
and baseline grading.

Start locally:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path so env/tasks/graders imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openenv.core.env_server.http_server import create_fastapi_app

from env import TaskSchedulerEnv, SchedulerAction, SchedulerObservation
from tasks import TASK_DEFINITIONS


# ──────────────────────────────────────────────────────────────
#  Create standard OpenEnv app
# ──────────────────────────────────────────────────────────────

app = create_fastapi_app(
    env=TaskSchedulerEnv,
    action_cls=SchedulerAction,
    observation_cls=SchedulerObservation,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
#  Custom request / response models
# ──────────────────────────────────────────────────────────────

class TaskNameRequest(BaseModel):
    task_name: str = "easy"


class GradeResponse(BaseModel):
    task: str
    score: float
    method: str = "heuristic"


# ──────────────────────────────────────────────────────────────
#  Custom endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Custom"])
def root():
    """Identity endpoint."""
    return {
        "name": "task_scheduler_env",
        "display_name": "Sprint Task Scheduler",
        "spec": "openenv",
        "framework": "openenv-core",
        "status": "running",
    }


@app.get("/tasks", tags=["Custom"])
def list_tasks():
    """List available task scenarios with metadata."""
    return {"tasks": TASK_DEFINITIONS}


@app.post("/grade", tags=["Custom"], response_model=GradeResponse)
def grade_baseline(req: TaskNameRequest):
    """Run the heuristic baseline internally and return its grade."""
    from inference import heuristic_action
    from graders import grade

    env = TaskSchedulerEnv()
    obs = env.reset(task_name=req.task_name)

    while not obs.done:
        action = heuristic_action(obs)
        obs = env.step(action)

    score = grade(req.task_name, env)
    return GradeResponse(task=req.task_name, score=round(score, 4))


# ──────────────────────────────────────────────────────────────
#  Entrypoint
# ──────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
