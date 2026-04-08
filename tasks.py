"""
Task Scenarios for the Sprint Task Scheduler Environment
=========================================================
Three deterministic scenarios of increasing difficulty.
Each scenario returns a fixed set of tasks and developers —
no randomness, so grading is fully reproducible.
"""

from typing import Any, Dict, List

# ──────────────────────────────────────────────────────────────
#  Public metadata exposed through /tasks and openenv.yaml
# ──────────────────────────────────────────────────────────────

TASK_DEFINITIONS: List[Dict[str, str]] = [
    {
        "name": "easy",
        "display_name": "Basic Sprint Planning",
        "description": (
            "Schedule 5 independent tasks across 2 developers in a 5-day "
            "sprint. No dependencies, generous deadlines, broad skill "
            "coverage. Any reasonable assignment scores well."
        ),
        "difficulty": "easy",
    },
    {
        "name": "medium",
        "display_name": "Backend Bottleneck with Dependencies",
        "description": (
            "Schedule 10 tasks with dependency chains across 3 developers. "
            "Backend bottleneck with 48h of backend work for a single "
            "backend developer (40h capacity). An anti-greedy trap: "
            "a high-priority low-value hotfix competes for critical capacity."
        ),
        "difficulty": "medium",
    },
    {
        "name": "hard",
        "display_name": "Sprint Optimisation Under Constraints",
        "description": (
            "Optimise scheduling of 15 tasks across 4 developers with "
            "severe backend bottleneck (56h work, 40h capacity), "
            "anti-greedy traps (P5 low-value hotfixes steal capacity), "
            "and cascading dependency failures. Strategic deferral is "
            "essential — the heuristic scores 0.65 while optimal is 0.90."
        ),
        "difficulty": "hard",
    },
]


# ──────────────────────────────────────────────────────────────
#  Scenario factory
# ──────────────────────────────────────────────────────────────

def get_scenario(name: str) -> Dict[str, Any]:
    """Return the full scenario dict for a given task name."""
    builders = {
        "easy": _easy_scenario,
        "medium": _medium_scenario,
        "hard": _hard_scenario,
    }
    if name not in builders:
        raise ValueError(f"Unknown task '{name}'. Choose from: {list(builders)}")
    return builders[name]()


# ──────────────────────────────────────────────────────────────
#  EASY — 5 tasks, 2 developers, no dependencies
# ──────────────────────────────────────────────────────────────

def _easy_scenario() -> Dict[str, Any]:
    return {
        "sprint_days": 5,
        "max_steps": 20,
        "developers": [
            {"id": "D1", "name": "Alice", "skills": ["frontend", "backend", "design"]},
            {"id": "D2", "name": "Bob",   "skills": ["backend", "devops", "qa"]},
        ],
        "tasks": [
            {
                "id": "T1", "title": "Fix login button styling",
                "description": "The login button clips on mobile viewports.",
                "priority": 3, "estimated_hours": 2.0, "deadline_day": 5,
                "category": "bug", "required_skill": "frontend",
                "dependencies": [], "business_value": 5.0,
            },
            {
                "id": "T2", "title": "Update project documentation",
                "description": "Refresh the README and API docs for v2.1.",
                "priority": 1, "estimated_hours": 1.5, "deadline_day": 5,
                "category": "docs", "required_skill": "any",
                "dependencies": [], "business_value": 2.0,
            },
            {
                "id": "T3", "title": "Add user avatar upload",
                "description": "Allow users to upload a profile picture.",
                "priority": 2, "estimated_hours": 3.0, "deadline_day": 5,
                "category": "feature", "required_skill": "frontend",
                "dependencies": [], "business_value": 4.0,
            },
            {
                "id": "T4", "title": "Fix API timeout on /search",
                "description": "The search endpoint times out on large datasets.",
                "priority": 4, "estimated_hours": 2.5, "deadline_day": 5,
                "category": "bug", "required_skill": "backend",
                "dependencies": [], "business_value": 7.0,
            },
            {
                "id": "T5", "title": "Write unit tests for auth service",
                "description": "Increase auth module coverage to 80%.",
                "priority": 2, "estimated_hours": 3.0, "deadline_day": 5,
                "category": "tech_debt", "required_skill": "backend",
                "dependencies": [], "business_value": 3.0,
            },
        ],
    }


# ──────────────────────────────────────────────────────────────
#  MEDIUM — 10 tasks, 3 developers, backend bottleneck
#  Anti-greedy trap: T3 (pri5, val=3) steals Bob's slot on day 2,
#  blocking T7 (val=5) from meeting its deadline. An intelligent
#  agent should defer T3 to save the higher-value chain.
# ──────────────────────────────────────────────────────────────

def _medium_scenario() -> Dict[str, Any]:
    return {
        "sprint_days": 5,
        "max_steps": 30,
        "developers": [
            {"id": "D1", "name": "Alice", "skills": ["frontend", "design"]},
            {"id": "D2", "name": "Bob",   "skills": ["backend"]},
            {"id": "D3", "name": "Carol", "skills": ["devops", "qa"]},
        ],
        "tasks": [
            {
                "id": "T1", "title": "Design dashboard mockups",
                "description": "Create high-fidelity mockups for the analytics dashboard.",
                "priority": 4, "estimated_hours": 7.0, "deadline_day": 1,
                "category": "feature", "required_skill": "design",
                "dependencies": [], "business_value": 6.0,
            },
            {
                "id": "T2", "title": "Critical payment bugfix",
                "description": "Double-charge issue affecting 2% of transactions.",
                "priority": 5, "estimated_hours": 8.0, "deadline_day": 1,
                "category": "bug", "required_skill": "backend",
                "dependencies": [], "business_value": 9.0,
            },
            {
                "id": "T3", "title": "Emergency metrics hotfix",
                "description": "Dashboard metrics showing stale data since last deploy.",
                "priority": 5, "estimated_hours": 8.0, "deadline_day": 2,
                "category": "bug", "required_skill": "backend",
                "dependencies": [], "business_value": 3.0,
            },
            {
                "id": "T4", "title": "Payment security audit",
                "description": "Run OWASP checks after the payment bugfix.",
                "priority": 4, "estimated_hours": 8.0, "deadline_day": 3,
                "category": "tech_debt", "required_skill": "backend",
                "dependencies": ["T2"], "business_value": 8.0,
            },
            {
                "id": "T5", "title": "Dashboard REST API",
                "description": "Implement /api/dashboard endpoints per the mockups.",
                "priority": 4, "estimated_hours": 8.0, "deadline_day": 4,
                "category": "feature", "required_skill": "backend",
                "dependencies": ["T1"], "business_value": 7.0,
            },
            {
                "id": "T6", "title": "Order service rewrite",
                "description": "Rewrite legacy order processing with new payment flow.",
                "priority": 3, "estimated_hours": 8.0, "deadline_day": 5,
                "category": "feature", "required_skill": "backend",
                "dependencies": ["T4"], "business_value": 6.0,
            },
            {
                "id": "T7", "title": "Database migration",
                "description": "Schema migration for new analytics tables.",
                "priority": 3, "estimated_hours": 8.0, "deadline_day": 4,
                "category": "infra", "required_skill": "backend",
                "dependencies": ["T5"], "business_value": 5.0,
            },
            {
                "id": "T8", "title": "Dashboard frontend",
                "description": "Build React components consuming the dashboard API.",
                "priority": 3, "estimated_hours": 7.0, "deadline_day": 5,
                "category": "feature", "required_skill": "frontend",
                "dependencies": ["T1", "T5"], "business_value": 5.0,
            },
            {
                "id": "T9", "title": "Integration test suite",
                "description": "End-to-end API tests for payment and order flows.",
                "priority": 2, "estimated_hours": 6.0, "deadline_day": 5,
                "category": "tech_debt", "required_skill": "qa",
                "dependencies": ["T4", "T5"], "business_value": 4.0,
            },
            {
                "id": "T10", "title": "API documentation",
                "description": "Auto-generate OpenAPI docs for new endpoints.",
                "priority": 1, "estimated_hours": 3.0, "deadline_day": 5,
                "category": "docs", "required_skill": "any",
                "dependencies": ["T5"], "business_value": 2.0,
            },
        ],
    }


# ──────────────────────────────────────────────────────────────
#  HARD — 15 tasks, 4 developers, severe capacity pressure
#
#  Why this is genuinely hard:
#  - Total backend work: 56h across Bob(40h) + Dave(40h) = 80h backend
#    capacity, but skill+dependency constraints limit effective capacity.
#  - Total frontend work: 30h, Alice has 40h but day 1 is consumed by
#    design, so effectively 32h frontend capacity with a day-1 gap.
#  - DevOps/QA: Carol is sole QA+devops → 40h for ~28h of work, BUT
#    dependencies push QA tasks to late sprint days, creating day-5 jam.
#  - Anti-greedy traps:
#    * T3 (pri5, val=2) and T4 (pri5, val=2) are urgent but low-value
#      hotfixes that consume early backend capacity. A greedy agent
#      takes them, blocking the high-value T5→T9→T14 chain.
#    * Deferring T3+T4 (lose val 4) saves T9+T14 (val 13). Net +9.
#  - At least 3-4 tasks MUST be deferred even in optimal solution.
#  - Optimal requires non-greedy reasoning about value chains.
# ──────────────────────────────────────────────────────────────

def _hard_scenario() -> Dict[str, Any]:
    return {
        "sprint_days": 5,
        "max_steps": 50,
        "developers": [
            {"id": "D1", "name": "Alice", "skills": ["frontend", "design"]},
            {"id": "D2", "name": "Bob",   "skills": ["backend"]},
            {"id": "D3", "name": "Carol", "skills": ["devops", "qa"]},
            {"id": "D4", "name": "Dave",  "skills": ["frontend"]},
        ],
        "tasks": [
            # ── Foundation tasks (day 1) ──
            {
                "id": "T1", "title": "Design system architecture",
                "description": "Architecture diagrams and component specs for new platform.",
                "priority": 5, "estimated_hours": 8.0, "deadline_day": 1,
                "category": "feature", "required_skill": "design",
                "dependencies": [], "business_value": 8.0,
            },
            {
                "id": "T2", "title": "Fix critical data corruption",
                "description": "Race condition causing data loss in order service.",
                "priority": 5, "estimated_hours": 8.0, "deadline_day": 1,
                "category": "bug", "required_skill": "backend",
                "dependencies": [], "business_value": 10.0,
            },
            # ── Anti-greedy trap: urgent but low-value hotfixes ──
            {
                "id": "T3", "title": "Hotfix: stale cache headers",
                "description": "CDN serving stale assets after last deploy.",
                "priority": 5, "estimated_hours": 8.0, "deadline_day": 2,
                "category": "bug", "required_skill": "backend",
                "dependencies": [], "business_value": 2.0,
            },
            {
                "id": "T4", "title": "Hotfix: logging overflow",
                "description": "Log rotation broken, disks filling up.",
                "priority": 5, "estimated_hours": 8.0, "deadline_day": 2,
                "category": "bug", "required_skill": "backend",
                "dependencies": [], "business_value": 2.0,
            },
            # ── High-value backend chain ──
            {
                "id": "T5", "title": "Implement user auth API",
                "description": "OAuth2 + JWT authentication endpoints.",
                "priority": 4, "estimated_hours": 8.0, "deadline_day": 3,
                "category": "feature", "required_skill": "backend",
                "dependencies": ["T1"], "business_value": 7.0,
            },
            {
                "id": "T6", "title": "Payment processing API",
                "description": "Stripe integration for new checkout flow.",
                "priority": 4, "estimated_hours": 8.0, "deadline_day": 3,
                "category": "feature", "required_skill": "backend",
                "dependencies": ["T2"], "business_value": 8.0,
            },
            {
                "id": "T7", "title": "Setup monitoring & alerts",
                "description": "Datadog dashboards and PagerDuty integrations.",
                "priority": 3, "estimated_hours": 8.0, "deadline_day": 2,
                "category": "infra", "required_skill": "devops",
                "dependencies": [], "business_value": 4.0,
            },
            # ── Mid-sprint tasks ──
            {
                "id": "T8", "title": "Build auth UI",
                "description": "Login/register forms, password reset flow.",
                "priority": 3, "estimated_hours": 8.0, "deadline_day": 4,
                "category": "feature", "required_skill": "frontend",
                "dependencies": ["T5"], "business_value": 5.0,
            },
            {
                "id": "T9", "title": "Order service rewrite",
                "description": "Rewrite legacy order processing with new payment flow.",
                "priority": 3, "estimated_hours": 8.0, "deadline_day": 4,
                "category": "feature", "required_skill": "backend",
                "dependencies": ["T5", "T6"], "business_value": 7.0,
            },
            {
                "id": "T10", "title": "Deploy staging environment",
                "description": "Provision staging infra with new services.",
                "priority": 4, "estimated_hours": 8.0, "deadline_day": 4,
                "category": "infra", "required_skill": "devops",
                "dependencies": ["T6", "T7"], "business_value": 7.0,
            },
            # ── Late-sprint tasks ──
            {
                "id": "T11", "title": "Search feature backend",
                "description": "Full-text search with Elasticsearch.",
                "priority": 2, "estimated_hours": 8.0, "deadline_day": 5,
                "category": "feature", "required_skill": "backend",
                "dependencies": ["T5"], "business_value": 4.0,
            },
            {
                "id": "T12", "title": "Search UI",
                "description": "Auto-complete search bar, results, filters.",
                "priority": 2, "estimated_hours": 7.0, "deadline_day": 5,
                "category": "feature", "required_skill": "frontend",
                "dependencies": ["T11"], "business_value": 3.0,
            },
            {
                "id": "T13", "title": "Integration test suite",
                "description": "End-to-end API tests for auth and order flows.",
                "priority": 3, "estimated_hours": 8.0, "deadline_day": 5,
                "category": "tech_debt", "required_skill": "qa",
                "dependencies": ["T5", "T6"], "business_value": 6.0,
            },
            {
                "id": "T14", "title": "Performance load testing",
                "description": "k6 load tests against staging with new services.",
                "priority": 3, "estimated_hours": 8.0, "deadline_day": 5,
                "category": "tech_debt", "required_skill": "qa",
                "dependencies": ["T9", "T10"], "business_value": 6.0,
            },
            {
                "id": "T15", "title": "API documentation",
                "description": "OpenAPI docs for auth, payment, and order endpoints.",
                "priority": 1, "estimated_hours": 4.0, "deadline_day": 5,
                "category": "docs", "required_skill": "any",
                "dependencies": ["T5", "T6"], "business_value": 2.0,
            },
        ],
    }
