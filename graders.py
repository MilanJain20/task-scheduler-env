"""
Graders for the Sprint Task Scheduler Environment
===================================================
Each grader returns a deterministic score in (0, 1) exclusive.
Grading criteria increase in sophistication from easy → hard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env import TaskSchedulerEnv


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────

def grade(task_name: str, env: "TaskSchedulerEnv") -> float:
    """
    Grade the current environment state for the given task.

    Returns a float strictly in (0, 1) — never exactly 0.0 or 1.0.
    """
    graders = {
        "easy": _grade_easy,
        "medium": _grade_medium,
        "hard": _grade_hard,
    }
    fn = graders.get(task_name)
    if fn is None:
        raise ValueError(f"Unknown task '{task_name}'. Choose from: {list(graders)}")
    # Clamp to strictly within (0, 1) — neither 0.0 nor 1.0 are allowed.
    EPS = 1e-4
    raw = fn(env)
    return round(max(EPS, min(1.0 - EPS, raw)), 4)


# ──────────────────────────────────────────────────────────────
#  EASY  — Score = fraction of tasks successfully assigned
# ──────────────────────────────────────────────────────────────

def _grade_easy(env: "TaskSchedulerEnv") -> float:
    """
    Simple metric: what fraction of tasks did the agent assign?
    Deferred or pending tasks reduce the score proportionally.

    Score components:
        • 100% — tasks assigned ratio
    """
    total = len(env.tasks)
    if total == 0:
        return 0.0
    assigned = sum(1 for t in env.tasks if t.status == "assigned")
    return assigned / total


# ──────────────────────────────────────────────────────────────
#  MEDIUM — Score considers assignment + deps + deadlines
# ──────────────────────────────────────────────────────────────

def _grade_medium(env: "TaskSchedulerEnv") -> float:
    """
    Balanced metric that rewards correct scheduling.

    Score components weighted:
        • 45% — business value delivered / total possible value
        • 25% — task assignment ratio
        • 15% — dependency compliance (all deps scheduled before the task)
        • 15% — deadline compliance (task scheduled on or before deadline)
    """
    total = len(env.tasks)
    if total == 0:
        return 0.0

    assigned = [t for t in env.tasks if t.status == "assigned"]
    assign_ratio = len(assigned) / total

    # Value delivery
    total_value = sum(t.business_value for t in env.tasks)
    delivered_value = sum(t.business_value for t in assigned)
    value_ratio = delivered_value / max(total_value, 1.0)

    # Dependency compliance
    dep_total = 0
    dep_ok = 0
    for t in assigned:
        for dep_id in t.dependencies:
            dep_total += 1
            dep = _find(env, dep_id)
            if dep is None:
                continue
            if (dep.status == "assigned"
                    and dep.scheduled_day is not None
                    and t.scheduled_day is not None
                    and dep.scheduled_day < t.scheduled_day):
                dep_ok += 1
    dep_ratio = dep_ok / max(dep_total, 1)

    # Deadline compliance
    dl_ok = sum(
        1 for t in assigned
        if t.scheduled_day is not None and t.scheduled_day <= t.deadline_day
    )
    dl_ratio = dl_ok / max(len(assigned), 1)

    return (0.45 * value_ratio
            + 0.25 * assign_ratio
            + 0.15 * dep_ratio
            + 0.15 * dl_ratio)


# ──────────────────────────────────────────────────────────────
#  HARD — Multi-dimensional optimisation score
# ──────────────────────────────────────────────────────────────

def _grade_hard(env: "TaskSchedulerEnv") -> float:
    """
    Comprehensive metric rewarding optimal Sprint management:

    Score components weighted:
        • 45% — business value delivered / total possible value
        • 20% — task assignment ratio (fraction of tasks scheduled)
        • 15% — dependency compliance
        • 10% — deadline compliance
        • 10% — combined skill-match + capacity compliance
    """
    total = len(env.tasks)
    if total == 0:
        return 0.0

    assigned = [t for t in env.tasks if t.status == "assigned"]
    assign_ratio = len(assigned) / total

    # ── Value delivery ──
    total_value = sum(t.business_value for t in env.tasks)
    delivered_value = sum(t.business_value for t in assigned)
    value_ratio = delivered_value / max(total_value, 1.0)

    # ── Dependency compliance ──
    dep_total = 0
    dep_ok = 0
    for t in assigned:
        for dep_id in t.dependencies:
            dep_total += 1
            dep = _find(env, dep_id)
            if dep is None:
                continue
            if (dep.status == "assigned"
                    and dep.scheduled_day is not None
                    and t.scheduled_day is not None
                    and dep.scheduled_day < t.scheduled_day):
                dep_ok += 1
    dep_ratio = dep_ok / max(dep_total, 1)

    # ── Deadline compliance ──
    dl_ok = sum(
        1 for t in assigned
        if t.scheduled_day is not None and t.scheduled_day <= t.deadline_day
    )
    dl_ratio = dl_ok / max(len(assigned), 1)

    # ── Skill match ──
    skill_ok = 0
    for t in assigned:
        if t.required_skill == "any":
            skill_ok += 1
            continue
        dev = next((d for d in env.developers if d.id == t.assigned_to), None)
        if dev and t.required_skill in dev.skills:
            skill_ok += 1
    skill_ratio = skill_ok / max(len(assigned), 1)

    # ── Capacity compliance (no overload on any dev-day) ──
    overload_slots = 0
    total_slots = 0
    for dev in env.developers:
        for day in range(1, env.sprint_days + 1):
            total_slots += 1
            load = env.developer_load.get(dev.id, {}).get(str(day), 0.0)
            if load > dev.hours_per_day:
                overload_slots += 1
    capacity_ratio = 1.0 - (overload_slots / max(total_slots, 1))

    return (0.45 * value_ratio
            + 0.20 * assign_ratio
            + 0.15 * dep_ratio
            + 0.10 * dl_ratio
            + 0.10 * (skill_ratio * 0.5 + capacity_ratio * 0.5))


# ──────────────────────────────────────────────────────────────
#  Helper
# ──────────────────────────────────────────────────────────────

def _find(env: "TaskSchedulerEnv", task_id: str):
    """Look up a task by ID in the environment."""
    return next((t for t in env.tasks if t.id == task_id), None)
