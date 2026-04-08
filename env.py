"""
Sprint Task Scheduler — OpenEnv Environment
=============================================
Simulates real-world sprint planning where an AI agent must schedule
software development tasks across team members, respecting dependencies,
deadlines, skill requirements, and capacity constraints.

Built on the openenv-core framework (https://github.com/meta-pytorch/OpenEnv).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
    EnvironmentMetadata,
)


# ──────────────────────────────────────────────────────────────
#  Domain models (unchanged)
# ──────────────────────────────────────────────────────────────

class TaskItem(BaseModel):
    """A software-development work item to schedule in the sprint."""
    id: str = Field(description="Unique task identifier, e.g. 'T1'")
    title: str = Field(description="Short descriptive title")
    description: str = Field(default="", description="Details about the task")
    priority: int = Field(ge=1, le=5, description="Business priority 1 (low) – 5 (critical)")
    estimated_hours: float = Field(gt=0, description="Estimated hours to complete")
    deadline_day: int = Field(ge=1, description="Must be scheduled on or before this sprint day")
    category: Literal["bug", "feature", "tech_debt", "docs", "infra"]
    required_skill: Literal["frontend", "backend", "devops", "design", "qa", "any"]
    dependencies: List[str] = Field(default_factory=list, description="IDs of prerequisite tasks")
    status: Literal["pending", "assigned", "deferred"] = "pending"
    assigned_to: Optional[str] = None
    scheduled_day: Optional[int] = None
    business_value: float = Field(ge=0, le=10, description="Business value 0–10")


class Developer(BaseModel):
    """A team member available for task assignment during the sprint."""
    id: str
    name: str
    skills: List[str]
    hours_per_day: float = Field(default=8.0, description="Available hours per day")


# ──────────────────────────────────────────────────────────────
#  OpenEnv typed models (inherit from openenv-core base classes)
# ──────────────────────────────────────────────────────────────

class SchedulerAction(BaseAction):
    """
    An action the agent takes to manage the sprint backlog.

    Inherits `metadata: Dict[str, Any]` from OpenEnv Action base.
    """
    action_type: Literal["assign", "defer", "finish"]
    task_id: Optional[str] = None
    developer_id: Optional[str] = None
    scheduled_day: Optional[int] = None


class SchedulerObservation(BaseObservation):
    """
    Complete observable state returned to the agent at each step.

    Inherits from OpenEnv Observation base which provides:
      - done: bool          (whether the episode has terminated)
      - reward: float|None  (reward signal from the last action)
      - metadata: Dict      (additional metadata)
    """
    tasks: List[TaskItem]
    developers: List[Developer]
    current_step: int
    max_steps: int
    sprint_days: int
    schedule: Dict[str, Dict[str, List[str]]]   # day -> dev_id -> [task_ids]
    developer_load: Dict[str, Dict[str, float]]  # dev_id -> day -> hours used
    messages: List[str] = Field(default_factory=list, description="Feedback from last action")
    task_name: str = ""
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown",
    )


class SchedulerState(BaseState):
    """
    Summary state for the sprint scheduler environment.

    Inherits from OpenEnv State base which provides:
      - episode_id: Optional[str]
      - step_count: int
    """
    task_name: str = ""
    current_step: int = 0
    max_steps: int = 30
    sprint_days: int = 5
    num_tasks: int = 0
    num_assigned: int = 0
    num_deferred: int = 0
    num_pending: int = 0
    is_done: bool = False


# Backward-compatibility aliases so existing code using
# `from env import Action, Observation` keeps working.
Action = SchedulerAction
Observation = SchedulerObservation


# ──────────────────────────────────────────────────────────────
#  Helper: render observation as readable text for LLM agents
# ──────────────────────────────────────────────────────────────

def render_observation(obs: SchedulerObservation) -> str:
    """Convert a SchedulerObservation to a human/LLM-readable text block."""
    lines: list[str] = []
    lines.append(f"=== Sprint Planner  (step {obs.current_step}/{obs.max_steps}) ===")
    lines.append(f"Sprint length: {obs.sprint_days} days\n")

    # Feedback messages
    if obs.messages:
        lines.append("--- Messages ---")
        for m in obs.messages:
            lines.append(f"  • {m}")
        lines.append("")

    # Team members & load
    lines.append("--- Team Members ---")
    for dev in obs.developers:
        load_parts = []
        for d in range(1, obs.sprint_days + 1):
            h = obs.developer_load.get(dev.id, {}).get(str(d), 0.0)
            load_parts.append(f"D{d}:{h:.1f}h")
        lines.append(
            f"  {dev.id} ({dev.name}): skills={dev.skills}, "
            f"capacity={dev.hours_per_day}h/day | {', '.join(load_parts)}"
        )
    lines.append("")

    # Pending tasks
    pending = [t for t in obs.tasks if t.status == "pending"]
    assigned = [t for t in obs.tasks if t.status == "assigned"]
    deferred = [t for t in obs.tasks if t.status == "deferred"]

    if pending:
        lines.append("--- Pending Tasks ---")
        for t in pending:
            deps = f", deps={t.dependencies}" if t.dependencies else ""
            lines.append(
                f"  {t.id}: \"{t.title}\" | {t.category} | skill={t.required_skill} | "
                f"{t.estimated_hours}h | deadline=day{t.deadline_day} | "
                f"priority={t.priority} | value={t.business_value}{deps}"
            )
        lines.append("")

    if assigned:
        lines.append("--- Assigned Tasks ---")
        for t in assigned:
            lines.append(f"  {t.id}: \"{t.title}\" → {t.assigned_to} on day {t.scheduled_day}")
        lines.append("")

    if deferred:
        lines.append("--- Deferred Tasks ---")
        for t in deferred:
            lines.append(f"  {t.id}: \"{t.title}\" (value={t.business_value})")
        lines.append("")

    lines.append(
        "Available actions:\n"
        '  assign  → {"action_type":"assign","task_id":"T1","developer_id":"D1","scheduled_day":1}\n'
        '  defer   → {"action_type":"defer","task_id":"T1"}\n'
        '  finish  → {"action_type":"finish"}'
    )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  Environment  (inherits openenv.core Environment)
# ──────────────────────────────────────────────────────────────

class TaskSchedulerEnv(
    Environment[SchedulerAction, SchedulerObservation, SchedulerState]
):
    """
    OpenEnv-compliant sprint-planning environment.

    API  (OpenEnv standard)
    -----------------------
    reset(task_name)  → SchedulerObservation
    step(action)      → SchedulerObservation   (reward & done on the observation)
    state             → SchedulerState          (property)
    close()           → None
    get_metadata()    → EnvironmentMetadata
    """

    def __init__(self) -> None:
        super().__init__()  # no transform / rubric
        self.tasks: List[TaskItem] = []
        self.developers: List[Developer] = []
        self.sprint_days: int = 5
        self.max_steps: int = 30
        self.current_step: int = 0
        self._done: bool = False
        self.task_name: str = ""
        self.messages: List[str] = []
        self.schedule: Dict[str, Dict[str, List[str]]] = {}
        self.developer_load: Dict[str, Dict[str, float]] = {}
        self.action_history: List[SchedulerAction] = []

    # convenience -----------------------------------------------------------
    @property
    def done(self) -> bool:
        """Whether the current episode has ended."""
        return self._done

    # ── reset ──────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "easy",
        **kwargs: Any,
    ) -> SchedulerObservation:
        """Reset the environment to a fresh sprint with the given scenario."""
        from tasks import get_scenario

        scenario = get_scenario(task_name)
        self.tasks = [TaskItem(**t) for t in scenario["tasks"]]
        self.developers = [Developer(**d) for d in scenario["developers"]]
        self.sprint_days = scenario.get("sprint_days", 5)
        self.max_steps = scenario.get("max_steps", 30)
        self.current_step = 0
        self._done = False
        self.task_name = task_name
        self.messages = [
            f"Sprint started. {len(self.tasks)} tasks, "
            f"{len(self.developers)} developers, {self.sprint_days} days."
        ]
        self.action_history = []

        # Build empty schedule grid
        self.schedule = {}
        self.developer_load = {}
        for day in range(1, self.sprint_days + 1):
            ds = str(day)
            self.schedule[ds] = {dev.id: [] for dev in self.developers}
        for dev in self.developers:
            self.developer_load[dev.id] = {
                str(d): 0.0 for d in range(1, self.sprint_days + 1)
            }

        return self._obs()

    # ── state (property – OpenEnv standard) ────────────────

    @property
    def state(self) -> SchedulerState:
        """Get the current environment state summary."""
        return SchedulerState(
            task_name=self.task_name,
            current_step=self.current_step,
            max_steps=self.max_steps,
            sprint_days=self.sprint_days,
            step_count=self.current_step,
            num_tasks=len(self.tasks),
            num_assigned=sum(1 for t in self.tasks if t.status == "assigned"),
            num_deferred=sum(1 for t in self.tasks if t.status == "deferred"),
            num_pending=sum(1 for t in self.tasks if t.status == "pending"),
            is_done=self._done,
        )

    # ── step ───────────────────────────────────────────────

    def step(
        self,
        action: SchedulerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SchedulerObservation:
        """Execute one agent action and return updated observation.

        The returned SchedulerObservation carries `.reward` and `.done`
        following the OpenEnv convention (no separate tuple).
        """
        if self._done:
            return self._obs(
                reward=0.0, done=True,
                breakdown={},
            )

        self.current_step += 1
        self.messages = []
        reward_val = 0.0
        breakdown: Dict[str, float] = {}

        # ---- dispatch action ----
        if action.action_type == "finish":
            self._done = True
            self.messages.append("Agent chose to finish scheduling.")
            bonus = self._final_quality() * 0.3
            reward_val += bonus
            breakdown["finish_bonus"] = bonus

        elif action.action_type == "assign":
            reward_val, breakdown = self._do_assign(action)

        elif action.action_type == "defer":
            reward_val, breakdown = self._do_defer(action)

        # ---- auto-end: all tasks handled ----
        if not self._done and all(
            t.status in ("assigned", "deferred") for t in self.tasks
        ):
            self._done = True
            self.messages.append("All tasks have been scheduled or deferred.")
            bonus = self._final_quality() * 0.3
            reward_val += bonus
            breakdown["all_handled_bonus"] = bonus

        # ---- auto-end: max steps ----
        if not self._done and self.current_step >= self.max_steps:
            self._done = True
            self.messages.append("Maximum steps reached — episode ended.")
            reward_val -= 0.1
            breakdown["timeout_penalty"] = -0.1

        clipped_reward = max(-1.0, min(1.0, reward_val))
        self.action_history.append(action)
        return self._obs(reward=clipped_reward, done=self._done, breakdown=breakdown)

    # ── get_metadata ───────────────────────────────────────

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Sprint Task Scheduler",
            description=(
                "A real-world sprint-planning environment where an AI agent "
                "schedules development tasks across team members, respecting "
                "dependencies, deadlines, skills, and capacity."
            ),
            version="1.0.0",
        )

    # ── close ──────────────────────────────────────────────

    def close(self) -> None:
        """Clean up resources (no-op for this environment)."""
        pass

    # ──────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────

    def _do_assign(self, action: SchedulerAction) -> tuple[float, Dict[str, float]]:
        bd: Dict[str, float] = {}

        task = self._task(action.task_id)
        if task is None:
            self.messages.append(f"Error: Task '{action.task_id}' not found.")
            bd["invalid_task"] = -0.05
            return -0.05, bd

        if task.status != "pending":
            self.messages.append(f"Error: Task '{task.id}' is already {task.status}.")
            bd["already_handled"] = -0.05
            return -0.05, bd

        dev = self._dev(action.developer_id)
        if dev is None:
            self.messages.append(f"Error: Developer '{action.developer_id}' not found.")
            bd["invalid_developer"] = -0.05
            return -0.05, bd

        day = action.scheduled_day
        if day is None or day < 1 or day > self.sprint_days:
            self.messages.append(f"Error: Day {day} out of range 1–{self.sprint_days}.")
            bd["invalid_day"] = -0.05
            return -0.05, bd

        day_s = str(day)
        reward = 0.0

        # Skill check
        if task.required_skill != "any" and task.required_skill not in dev.skills:
            self.messages.append(
                f"Warning: {dev.name} lacks '{task.required_skill}' skill for {task.id}."
            )
            bd["skill_mismatch"] = -0.06
            reward -= 0.06

        # Capacity check
        cur_load = self.developer_load[dev.id][day_s]
        if cur_load + task.estimated_hours > dev.hours_per_day:
            self.messages.append(
                f"Warning: {dev.name} overloaded on day {day} "
                f"({cur_load + task.estimated_hours:.1f}/{dev.hours_per_day}h)."
            )
            bd["overload"] = -0.06
            reward -= 0.06

        # Dependency check
        deps_ok = True
        for dep_id in task.dependencies:
            dep = self._task(dep_id)
            if dep is None:
                continue
            if dep.status != "assigned":
                deps_ok = False
                self.messages.append(f"Warning: Dependency '{dep_id}' not yet assigned.")
            elif dep.scheduled_day is not None and dep.scheduled_day >= day:
                deps_ok = False
                self.messages.append(
                    f"Warning: Dependency '{dep_id}' scheduled on day "
                    f"{dep.scheduled_day}, same or after day {day}."
                )
        if not deps_ok:
            bd["dependency_violation"] = -0.08
            reward -= 0.08
        elif task.dependencies:
            bd["dependencies_respected"] = 0.04
            reward += 0.04

        # Deadline check
        if day > task.deadline_day:
            self.messages.append(
                f"Warning: Task '{task.id}' deadline is day {task.deadline_day}, "
                f"but scheduled for day {day}."
            )
            bd["deadline_violation"] = -0.08
            reward -= 0.08
        else:
            bd["deadline_met"] = 0.03
            reward += 0.03

        # Commit assignment
        task.status = "assigned"
        task.assigned_to = dev.id
        task.scheduled_day = day
        self.schedule[day_s][dev.id].append(task.id)
        self.developer_load[dev.id][day_s] += task.estimated_hours

        # Progress reward proportional to business value
        progress = 0.08 * (task.business_value / 10.0)
        bd["progress"] = progress
        reward += progress

        self.messages.append(
            f"Assigned {task.id} → {dev.name} on day {day}."
        )
        return max(-1.0, min(1.0, reward)), bd

    # ── internal: defer ────────────────────────────────────

    def _do_defer(self, action: SchedulerAction) -> tuple[float, Dict[str, float]]:
        bd: Dict[str, float] = {}
        task = self._task(action.task_id)
        if task is None:
            self.messages.append(f"Error: Task '{action.task_id}' not found.")
            bd["invalid_task"] = -0.05
            return -0.05, bd
        if task.status != "pending":
            self.messages.append(f"Error: Task '{task.id}' is already {task.status}.")
            bd["already_handled"] = -0.05
            return -0.05, bd

        task.status = "deferred"
        penalty = -0.02 * task.priority  # higher-priority deferrals hurt more
        bd["defer_penalty"] = penalty
        self.messages.append(
            f"Deferred {task.id} (priority={task.priority}, value={task.business_value})."
        )
        return penalty, bd

    # ── internal: final schedule quality ───────────────────

    def _final_quality(self) -> float:
        """Score the overall schedule quality at episode end (0–1)."""
        if not self.tasks:
            return 0.0

        total_val = sum(t.business_value for t in self.tasks)
        delivered_val = sum(t.business_value for t in self.tasks if t.status == "assigned")
        value_ratio = delivered_val / max(total_val, 1.0)

        assigned = [t for t in self.tasks if t.status == "assigned"]
        if assigned:
            deadline_ok = sum(
                1 for t in assigned
                if t.scheduled_day is not None and t.scheduled_day <= t.deadline_day
            )
            deadline_ratio = deadline_ok / len(assigned)
        else:
            deadline_ratio = 0.0

        dep_total = 0
        dep_violations = 0
        for t in assigned:
            for dep_id in t.dependencies:
                dep_total += 1
                dep = self._task(dep_id)
                if dep is None:
                    continue
                if dep.status != "assigned":
                    dep_violations += 1
                elif (dep.scheduled_day is not None
                      and t.scheduled_day is not None
                      and dep.scheduled_day >= t.scheduled_day):
                    dep_violations += 1
        dep_ratio = 1.0 - (dep_violations / max(dep_total, 1))

        # Skill match ratio
        skill_ok = sum(
            1 for t in assigned
            if t.required_skill == "any"
            or (t.assigned_to and any(
                d.id == t.assigned_to and t.required_skill in d.skills
                for d in self.developers
            ))
        )
        skill_ratio = skill_ok / max(len(assigned), 1)

        return (0.40 * value_ratio
                + 0.25 * deadline_ratio
                + 0.20 * dep_ratio
                + 0.15 * skill_ratio)

    # ── helpers ────────────────────────────────────────────

    def _task(self, tid: Optional[str]) -> Optional[TaskItem]:
        if tid is None:
            return None
        return next((t for t in self.tasks if t.id == tid), None)

    def _dev(self, did: Optional[str]) -> Optional[Developer]:
        if did is None:
            return None
        return next((d for d in self.developers if d.id == did), None)

    def _obs(
        self,
        reward: Optional[float] = None,
        done: bool = False,
        breakdown: Optional[Dict[str, float]] = None,
    ) -> SchedulerObservation:
        return SchedulerObservation(
            tasks=self.tasks,
            developers=self.developers,
            current_step=self.current_step,
            max_steps=self.max_steps,
            sprint_days=self.sprint_days,
            schedule=self.schedule,
            developer_load=self.developer_load,
            messages=self.messages,
            task_name=self.task_name,
            done=done,
            reward=reward,
            reward_breakdown=breakdown or {},
        )
