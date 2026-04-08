---
title: Sprint Task Scheduler
emoji: 📋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Sprint Task Scheduler — OpenEnv Environment

> A real-world sprint-planning environment where an AI agent must schedule software development tasks across team members, respecting dependencies, deadlines, skill requirements, and capacity constraints.

[![OpenEnv](https://img.shields.io/badge/spec-OpenEnv-blue)](https://github.com/meta-pytorch/OpenEnv)
[![openenv-core](https://img.shields.io/badge/framework-openenv--core-purple)](https://pypi.org/project/openenv-core/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## Motivation

Every sprint, engineering managers and scrum masters face the same optimisation problem: assign a backlog of tasks to team members across a fixed number of days, while juggling:

- **Dependencies** — some tasks block others
- **Deadlines** — business commitments set hard cutoffs
- **Skills** — not every developer can do every task
- **Capacity** — each person has a limited number of hours per day

This environment faithfully models that problem so RL agents and LLMs can be trained and benchmarked on it. It fills a real gap — no existing OpenEnv covers project-management scheduling.

---

## Action Space

Each step, the agent submits **one JSON action**:

| Action | Schema | Description |
|--------|--------|-------------|
| **assign** | `{"action_type":"assign", "task_id":"T1", "developer_id":"D1", "scheduled_day":2}` | Assign a task to a developer on a specific sprint day |
| **defer** | `{"action_type":"defer", "task_id":"T1"}` | Defer a task to the next sprint (penalty scales with priority) |
| **finish** | `{"action_type":"finish"}` | Signal that scheduling is complete |

All fields are validated via Pydantic (`env.Action`).

---

## Observation Space

The observation (`env.Observation`) returned after every step contains:

| Field | Type | Description |
|-------|------|-------------|
| `tasks` | `List[TaskItem]` | All tasks with id, title, priority (1–5), estimated_hours, deadline_day, required_skill, dependencies, status, assigned_to, scheduled_day, business_value (0–10) |
| `developers` | `List[Developer]` | Team members with id, name, skills list, hours_per_day |
| `schedule` | `Dict[day → Dict[dev_id → List[task_ids]]]` | Current schedule grid |
| `developer_load` | `Dict[dev_id → Dict[day → hours_used]]` | Per-developer load per day |
| `messages` | `List[str]` | Feedback / warnings from the last action |
| `current_step` | `int` | Steps taken so far |
| `max_steps` | `int` | Episode step limit |
| `sprint_days` | `int` | Length of the sprint (5) |

---

## Reward Function

The reward is **not sparse** — it provides signal on every step:

| Component | Reward | When |
|-----------|--------|------|
| Progress | +0.01 to +0.08 | Proportional to the assigned task's business value |
| Deadline met | +0.03 | Task scheduled on or before its deadline |
| Dependencies respected | +0.04 | All prerequisite tasks already assigned on earlier days |
| Skill mismatch | −0.06 | Developer lacks the required skill |
| Overload | −0.06 | Developer exceeds capacity on that day |
| Dependency violation | −0.08 | Prerequisite not assigned or scheduled on same/later day |
| Deadline violation | −0.08 | Scheduled after the deadline |
| Defer penalty | −0.02 × priority | Higher-priority deferrals cost more |
| Invalid action | −0.05 | Unknown task/developer, already-handled task |
| Episode complete | +bonus | Final quality score (value delivery, dep compliance, deadline compliance, skill match) × 0.3 |
| Timeout | −0.10 | Hit max_steps without finishing |

Reward is clipped to **[-1, 1]** and accompanied by a `breakdown` dict for interpretability.

---

## Tasks

### 1. Easy — Basic Sprint Planning
- **5 tasks**, 2 developers, 5-day sprint
- No dependencies, all deadlines on day 5
- Broad skill coverage — any reasonable assignment works
- **Expected score**: ~1.0

### 2. Medium — Backend Bottleneck with Dependencies
- **10 tasks**, 3 developers (1 backend specialist), 5-day sprint
- Dependency chains (design → API → frontend, bugfix → security audit → order rewrite)
- Backend bottleneck: 48h of backend work, Bob (sole backend dev) has 40h capacity
- Anti-greedy trap: T3 (`Emergency metrics hotfix`, P5, value=3) consumes Bob's day 2, pushing the T5→T7 chain past its deadline
- Must respect task ordering and skill matching
- **Heuristic score**: 0.89

### 3. Hard — Sprint Optimisation Under Constraints
- **15 tasks**, 4 developers (only 1 backend specialist), 5-day sprint
- Complex dependency DAG, severe backend bottleneck (56h of backend work, 40h capacity)
- Anti-greedy traps: two high-priority (P5) but low-value hotfixes (T3, T4) consume critical early backend capacity
- Taking them causes a 9-task cascade failure, blocking the entire T5→T9→T14 value chain
- An intelligent agent must **defer the hotfixes** to unlock the high-value dependency chains
- Optimal solution scores ~0.90 vs heuristic's 0.65 — requires genuine strategic reasoning
- **Heuristic score**: 0.65

---

## Grading

Each grader returns a deterministic score in **[0.0, 1.0]**:

| Task | Scoring Formula |
|------|-----------------|
| **Easy** | `assigned_count / total_count` |
| **Medium** | 45% value delivered + 25% assignment ratio + 15% dep compliance + 15% deadline compliance |
| **Hard** | 45% value delivered + 20% assignment ratio + 15% dep compliance + 10% deadline compliance + 10% skill+capacity |

---

## Setup & Usage

### Local (no Docker)

```bash
# Install openenv-core (--no-deps avoids heavy gradio resolution)
pip install openenv-core --no-deps
pip install fastmcp --no-deps

# Install runtime dependencies
pip install -r requirements.txt

# Run the heuristic baseline
python inference.py

# Run with an LLM agent
OPENAI_API_KEY=sk-... python inference.py

# Start the HTTP server (standard OpenEnv API + custom endpoints)
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t task-scheduler-env .
docker run -p 7860:7860 task-scheduler-env
```

### API Endpoints

The server uses `openenv-core`'s `create_fastapi_app` for the standard OpenEnv API plus custom task-specific routes:

| Method | Path | Source | Description |
|--------|------|--------|-------------|
| `POST` | `/reset` | OpenEnv | Reset environment (body: `{"task_name":"easy"}`) |
| `POST` | `/step` | OpenEnv | Take an action (body: SchedulerAction JSON) |
| `GET` | `/state` | OpenEnv | Current state summary |
| `GET` | `/health` | OpenEnv | Health check |
| `GET` | `/schema` | OpenEnv | JSON schemas for Action / Observation / State |
| `GET` | `/metadata` | OpenEnv | Environment metadata |
| `WS` | `/ws` | OpenEnv | **Stateful** WebSocket session |
| `GET` | `/` | Custom | Identity endpoint |
| `GET` | `/tasks` | Custom | List available task scenarios |
| `POST` | `/grade` | Custom | Run heuristic baseline and return score |

---

## Baseline Scores (Heuristic)

Deterministic greedy heuristic (priority-first, earliest-fit):

| Task | Score | Reward | Steps |
|------|-------|--------|-------|
| Easy | 1.0000 | 0.618 | 5 |
| Medium | 0.8927 | 1.029 | 10 |
| Hard | 0.6467 | 0.290 | 15 |
| **Average** | **0.8465** | — | — |

*Scores are fully reproducible — no randomness in scenarios or heuristic.*

The hard task has a **0.25-point gap** between the heuristic (0.65) and the optimal solution (~0.90), rewarding agents that reason about value chains rather than just following priority ordering. An LLM agent that correctly identifies and avoids the anti-greedy traps can significantly outperform the baseline.

---

## Project Structure

```
task_scheduler_env/
├── env.py              # Environment: OpenEnv base classes + Pydantic models + step/reset/state
├── tasks.py            # Three deterministic task scenarios
├── graders.py          # Grading functions (0.0–1.0)
├── inference.py        # Baseline script (heuristic + OpenAI API)
├── server.py           # OpenEnv HTTP server (create_fastapi_app + custom routes)
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile          # HF Spaces container
├── requirements.txt    # Python dependencies
├── .dockerignore       # Docker build exclusions
└── README.md           # This file
```

---

## Environment Design Highlights

- **Built on openenv-core** — inherits from `Environment[Action, Observation, State]` base classes, uses `create_fastapi_app` for the server
- **Deterministic scenarios** — no randomness, fully reproducible grading
- **Rich reward shaping** — partial credit on every step, not just end-of-episode
- **Anti-greedy traps** — high-priority items aren't always high-value; the hard task rewards strategic reasoning over simple priority-following
- **Typed Pydantic models** — `SchedulerAction`, `SchedulerObservation`, `SchedulerState` with full validation
- **WebSocket support** — stateful agent interaction via standard OpenEnv `/ws` endpoint
- **Human-readable rendering** — `render_observation()` produces text suitable for LLM prompts
- **Genuine difficulty progression** — easy is straightforward, medium requires dependency awareness, hard requires strategic deferral of high-priority traps to unlock value chains
- **Real-world domain** — sprint planning is a $B+ industry problem that every software team faces
