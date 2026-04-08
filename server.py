"""
FastAPI HTTP Server for the Sprint Task Scheduler OpenEnv
==========================================================
Uses openenv-core's ``create_fastapi_app`` to expose the standard
OpenEnv API (``/reset``, ``/step``, ``/state``, ``/health``, ``/schema``,
``/metadata``, ``/ws``) and adds custom endpoints for task listing
and baseline grading.

Start locally:
    uvicorn server:app --host 0.0.0.0 --port 7860 --reload
"""

from __future__ import annotations

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openenv.core.env_server.http_server import create_fastapi_app

from env import TaskSchedulerEnv, SchedulerAction, SchedulerObservation
from tasks import TASK_DEFINITIONS


# ──────────────────────────────────────────────────────────────
#  Create standard OpenEnv app
# ──────────────────────────────────────────────────────────────
#
# create_fastapi_app expects an env *factory* (callable), not an instance.
# It provides:
#   POST /reset         — create env, reset, return observation
#   POST /step          — create env, step, return observation
#   GET  /state         — create env, return state
#   GET  /health        — health check
#   GET  /schema        — JSON schemas for Action / Observation / State
#   GET  /metadata      — environment metadata
#   WS   /ws            — stateful WebSocket session (persistent env per conn)

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
#  Custom endpoints (on top of the standard OpenEnv routes)
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
    """Run the heuristic baseline internally and return its grade.

    Since the standard OpenEnv HTTP endpoints are stateless (new env per
    request), this endpoint spins up a full episode with the greedy
    heuristic and reports the resulting score.
    """
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
