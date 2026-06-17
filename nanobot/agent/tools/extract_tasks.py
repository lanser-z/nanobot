"""ExtractTasksTool — first sample capability (P0 demo).

spec: capability-business-contract-bridge#8
"""

from __future__ import annotations

import hashlib

from nanobot.agent.tools.base import Tool, ToolMeta


class ExtractTasksTool(Tool):
    """Extract actionable tasks from a meeting transcript.

    v1 implementation: deterministic mock that returns predictable tasks
    based on the input meeting_id (so tests are reproducible). In v2 this
    is replaced with a real LLM call (via HttpAiProvider).
    """

    name = "extract_tasks"
    description = "Extract actionable tasks from a meeting transcript"
    parameters: dict = {
        "type": "object",
        "properties": {
            "meeting_id": {"type": "string", "description": "会议编号"},
            "max_tasks": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
        },
        "required": ["meeting_id"],
    }

    # P3-compatible metadata — P0 implementation defaults work as-is
    invocation_mode = "both"
    is_long_running = False
    is_idempotent = True
    requires_auth = frozenset()
    timeout_s = 120.0
    side_effect_class = "read"
    owner = "ai-team"
    version = "0.1.0"
    status = "active"
    tags = frozenset({"extract", "meeting"})

    async def execute(self, meeting_id: str, max_tasks: int = 20, **kwargs) -> dict:
        """Deterministic mock — hash-based task generation for tests."""
        # Use a stable hash so the same meeting_id always yields the same tasks
        seed = int(hashlib.sha256(meeting_id.encode()).hexdigest()[:8], 16)
        n_tasks = min(max_tasks, 5)  # cap mock at 5
        tasks = []
        for i in range(n_tasks):
            task_id = f"t-{seed:08x}-{i}"
            confidence = 0.95 - (i * 0.05)  # decreasing confidence
            tasks.append({
                "id": task_id,
                "title": f"Task {i+1} for meeting {meeting_id}",
                "owner": f"user-{(seed + i) % 10}",
                "due_date": None,
                "confidence": confidence,
            })
        return {
            "tasks": tasks,
            "total": n_tasks,
            "avg_confidence": sum(t["confidence"] for t in tasks) / n_tasks if tasks else 0.0,
        }
