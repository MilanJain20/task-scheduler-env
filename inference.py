"""
Baseline Inference Script for the Sprint Task Scheduler
========================================================
Runs a deterministic heuristic (or an LLM via the OpenAI API)
against all three tasks and prints reproducible baseline scores.

Follows the OpenEnv submission checklist:
- Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN, LOCAL_IMAGE_NAME
- All LLM calls use the OpenAI client configured via these variables
- Stdout logs follow the required structured format (START / STEP / END)

Usage
-----
    # Heuristic baseline (no API key needed):
    python inference.py

    # LLM baseline via HF Inference:
    HF_TOKEN=hf_... python inference.py

    # LLM baseline via custom endpoint:
    API_BASE_URL=http://localhost:8000/v1 HF_TOKEN=sk-... MODEL_NAME=gpt-4o python inference.py
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Optional

from openai import OpenAI

from env import (
    TaskSchedulerEnv,
    SchedulerAction as Action,
    SchedulerObservation as Observation,
    render_observation,
)
from graders import grade


# ──────────────────────────────────────────────────────────────
#  Environment Variables (OpenEnv submission checklist)
# ──────────────────────────────────────────────────────────────

# Defaults are set ONLY for API_BASE_URL and MODEL_NAME.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B")

# HF_TOKEN has NO default — must be provided via env var for LLM mode.
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


# ──────────────────────────────────────────────────────────────
#  Heuristic Agent (deterministic, no API key required)
# ──────────────────────────────────────────────────────────────

def heuristic_action(obs: Observation) -> Action:
    """
    Greedy heuristic:
    1. Sort pending tasks by priority (desc), deadline (asc), value (desc).
    2. For each task whose dependencies are all assigned:
       a. Find a skill-matched developer with the earliest available day
          that respects dependencies and deadline.
       b. If found → assign.
    3. If no assignable task exists, defer the lowest-value pending task.
    4. If nothing is pending → finish.
    """
    pending = [t for t in obs.tasks if t.status == "pending"]
    if not pending:
        return Action(action_type="finish")

    # Sort: highest priority first, then earliest deadline, then highest value
    pending.sort(key=lambda t: (-t.priority, t.deadline_day, -t.business_value))

    assigned_map = {t.id: t for t in obs.tasks if t.status == "assigned"}

    for task in pending:
        # Check all dependencies are assigned
        deps_ok = True
        max_dep_day = 0
        for dep_id in task.dependencies:
            dep = assigned_map.get(dep_id)
            if dep is None:
                deps_ok = False
                break
            if dep.scheduled_day is not None:
                max_dep_day = max(max_dep_day, dep.scheduled_day)
        if not deps_ok:
            continue

        earliest_day = max(1, max_dep_day + 1)

        # Find best (developer, day) slot
        best = None  # (dev_id, day, load)
        for dev in obs.developers:
            skill_ok = (task.required_skill == "any"
                        or task.required_skill in dev.skills)
            if not skill_ok:
                continue
            for day in range(earliest_day,
                             min(task.deadline_day, obs.sprint_days) + 1):
                load = obs.developer_load.get(dev.id, {}).get(str(day), 0.0)
                if load + task.estimated_hours <= dev.hours_per_day:
                    if best is None or day < best[1] or (day == best[1] and load < best[2]):
                        best = (dev.id, day, load)
            # take first valid dev with best day
            if best is not None:
                break

        if best is not None:
            return Action(
                action_type="assign",
                task_id=task.id,
                developer_id=best[0],
                scheduled_day=best[1],
            )

    # Nothing assignable → defer the lowest-value pending task
    lowest = min(pending, key=lambda t: (t.business_value, -t.priority))
    return Action(action_type="defer", task_id=lowest.id)


# ──────────────────────────────────────────────────────────────
#  LLM Agent (requires HF_TOKEN)
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior engineering manager scheduling a development sprint.

RULES:
• Assign every task to a developer on a specific day (1–{sprint_days}).
• A developer can only work up to {hours_per_day} hours per day.
• Tasks with dependencies must be scheduled AFTER all their dependencies finish (on a later day).
• Tasks must be scheduled on or before their deadline_day.
• Assign tasks to developers who have the required skill (unless skill="any").
• Defer tasks only when they truly cannot fit.

Respond with EXACTLY ONE JSON action per turn (no markdown, no explanation):
  {{"action_type":"assign","task_id":"T1","developer_id":"D1","scheduled_day":1}}
  {{"action_type":"defer","task_id":"T1"}}
  {{"action_type":"finish"}}
"""


def llm_action(client: OpenAI, model: str, obs: Observation) -> Action:
    """Call the OpenAI-compatible API to get the next action."""
    system = SYSTEM_PROMPT.format(
        sprint_days=obs.sprint_days,
        hours_per_day=obs.developers[0].hours_per_day if obs.developers else 8,
    )
    user_msg = render_observation(obs)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=256,
            timeout=30.0,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  [WARN] LLM call failed: {e}, using heuristic.")
        return heuristic_action(obs)

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(raw)
        return Action(**data)
    except Exception:
        # Fallback to heuristic on parse failure
        print(f"  [WARN] Could not parse LLM response, using heuristic. Raw: {raw[:120]}")
        return heuristic_action(obs)


# ──────────────────────────────────────────────────────────────
#  Main runner
# ──────────────────────────────────────────────────────────────

def run_task(
    env: TaskSchedulerEnv,
    task_name: str,
    client: Optional[OpenAI] = None,
    model: str = MODEL_NAME,
) -> dict:
    """Run one episode with structured START / STEP / END logging."""

    obs = env.reset(task_name=task_name)
    total_reward = 0.0
    steps = 0

    # ── [START] ──
    print(f"[START] task={task_name} num_tasks={len(obs.tasks)} "
          f"num_devs={len(obs.developers)} sprint_days={obs.sprint_days}")

    while not obs.done:
        if client is not None:
            action = llm_action(client, model, obs)
        else:
            action = heuristic_action(obs)

        obs = env.step(action)
        reward = obs.reward or 0.0
        total_reward += reward
        steps += 1

        # ── [STEP] ──
        step_detail = f"action={action.action_type}"
        if action.task_id:
            step_detail += f" task_id={action.task_id}"
        if action.developer_id:
            step_detail += f" developer_id={action.developer_id}"
        if action.scheduled_day is not None:
            step_detail += f" day={action.scheduled_day}"
        print(f"[STEP] step={steps} {step_detail} reward={reward:.4f} done={obs.done}")

    score = grade(task_name, env)

    # ── [END] ──
    print(f"[END] task={task_name} score={score:.4f} "
          f"total_reward={total_reward:.4f} steps={steps}")

    return {
        "task": task_name,
        "score": round(score, 4),
        "total_reward": round(total_reward, 4),
        "steps": steps,
    }


def main() -> None:
    client: Optional[OpenAI] = None

    if HF_TOKEN:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        print(f"[MODE] LLM agent  (model={MODEL_NAME}, api_base={API_BASE_URL})")
    else:
        print("[MODE] Heuristic baseline  (set HF_TOKEN for LLM mode)")

    env = TaskSchedulerEnv()
    results = {}

    for task_name in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}")
        print(f"  Task: {task_name.upper()}")
        print(f"{'='*50}")
        try:
            result = run_task(env, task_name, client=client, model=MODEL_NAME)
            results[task_name] = result
            print(f"  Score : {result['score']:.4f}")
            print(f"  Reward: {result['total_reward']:.4f}")
            print(f"  Steps : {result['steps']}")
        except Exception:
            traceback.print_exc()
            results[task_name] = {"score": 0.0, "error": "exception"}

    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    for name, r in results.items():
        s = r.get("score", 0.0)
        print(f"  {name:8s}  score={s:.4f}")

    avg = sum(r.get("score", 0.0) for r in results.values()) / max(len(results), 1)
    print(f"  {'average':8s}  score={avg:.4f}")

    print(f"\n[RESULTS_JSON] {json.dumps(results)}")


if __name__ == "__main__":
    main()
