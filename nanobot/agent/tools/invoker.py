"""Direct tool invoker — bypass LLM, reuse registry's prepare_call + execute.

spec: tool-invocation-interface#2, #13, #14
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry


@dataclass(slots=True)
class ToolResult:
    """Result of a single direct tool invocation (no LLM in the loop)."""

    ok: bool
    tool_name: str
    content: Any = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: int = 0
    warnings: list[str] = field(default_factory=list)


# --- P3-compatibility read-side dataclasses (spec tool-invocation-interface#13, #14) ---


@dataclass(frozen=True, slots=True)
class CapabilityInfo:
    """Public view of a capability (subset of ToolMeta for SDK/HTTP responses)."""

    name: str
    description: str
    parameters: dict
    meta: ToolMeta


@dataclass(frozen=True, slots=True)
class CapabilityMetadata:
    """Read-side metadata. P3 will switch storage to DB; signature MUST stay stable."""

    name: str
    invocation_mode: str
    is_long_running: bool
    is_idempotent: bool
    requires_auth: frozenset[str]
    timeout_s: float
    side_effect_class: str
    owner: str
    version: str
    status: str
    tags: frozenset[str]


@dataclass(frozen=True, slots=True)
class AuthContext:
    """Caller identity. P3 will expand (tenant_id / dept hierarchy / ...)."""

    user_id: str | None = None
    dept_ids: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class AuthResult:
    """Auth check result. `allowed=True` short-circuits invocation."""

    allowed: bool
    missing: frozenset[str] = frozenset()
    reason: str | None = None


async def invoke(
    registry: "ToolRegistry",
    name: str,
    params: Any,
    *,
    timeout: float | None = None,
    ctx_overrides: dict | None = None,
) -> ToolResult:
    """Invoke a tool directly, bypassing LLM.

    Reuses ``registry.prepare_call`` for JSON Schema validation and
    ``Tool.execute(**cast_params)`` for actual execution.

    Returns a ToolResult even on failure (does NOT raise). Timeouts and
    exceptions are captured into ``ToolResult.error`` so callers can
    degrade gracefully (spec tool-invocation-interface#2).

    The ``ctx_overrides`` dict is reserved for future use (P3 SHOULD_INVOKE
    hook, AuthContext propagation) — currently a no-op pass-through.
    """
    started = time.monotonic()

    def _fire_audit(ok: bool, content: Any, err: str | None) -> None:
        from nanobot.agent.tools.registry import HookType, RegistryEvent
        registry._invoke_hooks(RegistryEvent(
            hook_type=HookType.AUDIT,
            tool_name=name,
            timestamp=started,
            ctx=ctx_overrides or {},
            duration_ms=int((time.monotonic() - started) * 1000),
            result_summary=_summary(content if ok else None),
            source="external",
        ))

    # Resolve + validate (no execution)
    tool, cast_params, error = registry.prepare_call(name, params)
    if error:
        result = ToolResult(
            ok=False,
            tool_name=name,
            error=error,
            started_at=started,
            finished_at=time.monotonic(),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        _fire_audit(False, None, error)
        return result

    # At this point tool is guaranteed non-None (guarded by prepare_call)
    assert tool is not None
    effective_timeout = timeout if timeout is not None else tool.meta.timeout_s

    async def _run() -> Any:
        return await tool.execute(**cast_params)

    try:
        if effective_timeout is not None and effective_timeout > 0:
            content = await asyncio.wait_for(_run(), timeout=effective_timeout)
        else:
            content = await _run()
    except asyncio.TimeoutError:
        result = ToolResult(
            ok=False,
            tool_name=name,
            error=f"timeout after {effective_timeout}s",
            started_at=started,
            finished_at=time.monotonic(),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        _fire_audit(False, None, result.error)
        return result
    except Exception as e:  # noqa: BLE001  — capture ALL exceptions per spec
        err = f"Error executing {name}: {e}"
        result = ToolResult(
            ok=False,
            tool_name=name,
            error=err,
            started_at=started,
            finished_at=time.monotonic(),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        _fire_audit(False, None, err)
        return result

    finished = time.monotonic()
    duration_ms = int((finished - started) * 1000)
    _fire_audit(True, content, None)

    return ToolResult(
        ok=True,
        tool_name=name,
        content=content,
        started_at=started,
        finished_at=finished,
        duration_ms=duration_ms,
    )


def _summary(value: Any, limit: int = 200) -> str:
    """Best-effort truncated string summary for AUDIT hook payload."""
    if value is None:
        return ""
    try:
        s = repr(value)
    except Exception:
        s = f"<{type(value).__name__}>"
    return s if len(s) <= limit else s[:limit] + "..."
