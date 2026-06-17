"""Tool registry for dynamic tool management."""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from nanobot.agent.tools.base import Tool


class HookType(str, Enum):
    """ToolRegistry hook types — 7 values for P3 compat (spec tool-invocation-interface#12).

    v1 implementation only ACTUALLY triggers ``AUDIT``. The other 6 are defined
    for P3 future use; P3 will replace the no-op path with real handlers.
    """

    AUDIT = "audit"
    ON_REGISTER = "on_register"
    ON_UNREGISTER = "on_unregister"
    ON_ACTIVATE = "on_activate"
    ON_DEPRECATE = "on_deprecate"
    SHOULD_INVOKE = "should_invoke"
    QUOTA_CHECK = "quota_check"


@dataclass(slots=True)
class RegistryEvent:
    """Event payload passed to registered hooks.

    spec tool-invocation-interface#12
    """

    hook_type: HookType
    tool_name: str
    timestamp: float
    ctx: dict = field(default_factory=dict)
    duration_ms: int | None = None
    result_summary: Any | None = None
    source: str | None = None  # "llm" | "external" | "sdk" | "http" | "cli"


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._cached_definitions: list[dict[str, Any]] | None = None
        # spec tool-invocation-interface#12
        self._hooks: dict[HookType, list[Callable[[RegistryEvent], None]]] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        self._cached_definitions = None
        # v1: ON_REGISTER defined but no-op (P3 will wire real impl)
        self._invoke_hooks(RegistryEvent(
            hook_type=HookType.ON_REGISTER,
            tool_name=tool.name,
            timestamp=time.time(),
        ))

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
        self._cached_definitions = None
        # v1: ON_UNREGISTER defined but no-op (P3 will wire real impl)
        self._invoke_hooks(RegistryEvent(
            hook_type=HookType.ON_UNREGISTER,
            tool_name=name,
            timestamp=time.time(),
        ))

    # -- Hook system — spec tool-invocation-interface#12 --

    def add_hook(
        self,
        hook_type: HookType,
        fn: Callable[[RegistryEvent], None],
    ) -> None:
        """Register a hook. v1 only AUDIT is actually triggered; others accepted silently."""
        self._hooks.setdefault(hook_type, []).append(fn)

    def _invoke_hooks(self, event: RegistryEvent) -> None:
        """Fire all hooks registered for this event's hook_type.

        v1 implementation: only AUDIT is fired. Other hook types are
        accepted into _hooks (so add_hook works) but ignored (so P3
        implementation can enable them without signature change).
        """
        if event.hook_type != HookType.AUDIT:
            # v1: skip non-AUDIT hooks (P3 will enable)
            return
        for fn in self._hooks.get(event.hook_type, []):
            try:
                fn(event)
            except Exception:  # noqa: BLE001
                # hook errors MUST NOT break the registry
                pass

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    @staticmethod
    def _lookup_key(name: str) -> str:
        """Normalize names for suggestions only; never for execution."""
        return "".join(ch.lower() for ch in name if ch.isalnum())

    def _suggest_name(self, name: str) -> str | None:
        key = self._lookup_key(str(name or ""))
        if not key:
            return None
        matches = [
            registered
            for registered in self._tools
            if self._lookup_key(registered) == key
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    @staticmethod
    def _schema_name(schema: dict[str, Any]) -> str:
        """Extract a normalized tool name from either OpenAI or flat schemas."""
        fn = schema.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str):
                return name
        name = schema.get("name")
        return name if isinstance(name, str) else ""

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions with stable ordering for cache-friendly prompts.

        Built-in tools are sorted first as a stable prefix, then MCP tools are
        sorted and appended.  The result is cached until the next
        register/unregister call.
        """
        if self._cached_definitions is not None:
            return self._cached_definitions

        definitions = [tool.to_schema() for tool in self._tools.values()]
        builtins: list[dict[str, Any]] = []
        mcp_tools: list[dict[str, Any]] = []
        for schema in definitions:
            name = self._schema_name(schema)
            if name.startswith("mcp_"):
                mcp_tools.append(schema)
            else:
                builtins.append(schema)

        builtins.sort(key=self._schema_name)
        mcp_tools.sort(key=self._schema_name)
        self._cached_definitions = builtins + mcp_tools
        return self._cached_definitions

    def prepare_call(
        self,
        name: str,
        params: Any,
    ) -> tuple[Tool | None, Any, str | None]:
        """Resolve, cast, and validate one tool call."""
        tool = self._tools.get(name)
        if not tool:
            suggestion = self._suggest_name(str(name))
            hint = f" Did you mean '{suggestion}'? Tool names must match exactly." if suggestion else ""
            return None, params, (
                f"Error: Tool '{name}' not found.{hint} Available: {', '.join(self.tool_names)}"
            )

        params = self._coerce_params(tool, params)
        if not isinstance(params, dict):
            return tool, params, (
                f"Error: Tool '{name}' parameters must be a JSON object, got "
                f"{type(params).__name__}. Use named parameters like "
                'tool_name(param1="value1", param2="value2") matching the tool schema.'
            )

        cast_params = tool.cast_params(params)
        errors = tool.validate_params(cast_params)
        if errors:
            return tool, cast_params, (
                f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            )
        return tool, cast_params, None

    @classmethod
    def _coerce_argument_value(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        if not stripped:
            return {}

        if not stripped.startswith(("{", "[")):
            return value

        try:
            parsed = json.loads(stripped)
        except Exception:
            return value

        return parsed

    @classmethod
    def _coerce_params(cls, tool: Tool, params: Any) -> Any:
        params = cls._coerce_argument_value(params)
        return cls._unwrap_arguments_payload(tool, params)

    @classmethod
    def _unwrap_arguments_payload(cls, tool: Tool, params: Any) -> Any:
        if not isinstance(params, dict) or set(params) != {"arguments"}:
            return params
        properties = (tool.parameters or {}).get("properties", {})
        if isinstance(properties, dict) and "arguments" in properties:
            return params
        return cls._coerce_argument_value(params.get("arguments"))

    async def execute(self, name: str, params: Any) -> Any:
        """Execute a tool by name with given parameters."""
        import time as _t
        started = _t.monotonic()
        hint = "\n\n[Analyze the error above and try a different approach.]"
        tool, params, error = self.prepare_call(name, params)
        if error:
            self._invoke_hooks(RegistryEvent(
                hook_type=HookType.AUDIT,
                tool_name=name,
                timestamp=started,
                duration_ms=int((_t.monotonic() - started) * 1000),
                source="llm",
            ))
            return error + hint

        try:
            assert tool is not None  # guarded by prepare_call()
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                self._invoke_hooks(RegistryEvent(
                    hook_type=HookType.AUDIT,
                    tool_name=name,
                    timestamp=started,
                    duration_ms=int((_t.monotonic() - started) * 1000),
                    source="llm",
                ))
                return result + hint
            self._invoke_hooks(RegistryEvent(
                hook_type=HookType.AUDIT,
                tool_name=name,
                timestamp=started,
                duration_ms=int((_t.monotonic() - started) * 1000),
                source="llm",
            ))
            return result
        except Exception as e:
            self._invoke_hooks(RegistryEvent(
                hook_type=HookType.AUDIT,
                tool_name=name,
                timestamp=started,
                duration_ms=int((_t.monotonic() - started) * 1000),
                source="llm",
            ))
            return f"Error executing {name}: {str(e)}" + hint

    # -- Direct invocation (bypasses LLM) — spec tool-invocation-interface#2, #13, #14 --

    async def invoke(
        self,
        name: str,
        params: Any,
        *,
        timeout: float | None = None,
        ctx_overrides: dict | None = None,
    ) -> "ToolResult":
        """Thin wrapper around :func:`nanobot.agent.tools.invoker.invoke`."""
        from nanobot.agent.tools.invoker import invoke as _invoke
        return await _invoke(self, name, params, timeout=timeout, ctx_overrides=ctx_overrides)

    def list_capabilities(
        self,
        *,
        mode: str | None = None,
    ) -> list["CapabilityInfo"]:
        """List capabilities by invocation_mode filter.

        spec tool-invocation-interface#4
        """
        from nanobot.agent.tools.invoker import CapabilityInfo
        all_tools = list(self._tools.values())
        if mode is None:
            # default: capability + both (business-callable)
            selected = [
                t for t in all_tools
                if t.meta.invocation_mode in ("capability", "both")
            ]
        elif mode == "all":
            selected = all_tools
        elif mode in ("tool", "capability", "both", "internal"):
            selected = [t for t in all_tools if t.meta.invocation_mode == mode]
        else:
            selected = []
        return [
            CapabilityInfo(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
                meta=t.meta,
            )
            for t in selected
        ]

    def get_capability_metadata(self, name: str) -> "CapabilityMetadata | None":
        """Read-side metadata accessor. P3 will switch storage to DB; signature stable.

        spec tool-invocation-interface#13
        """
        from nanobot.agent.tools.invoker import CapabilityMetadata
        t = self._tools.get(name)
        if t is None:
            return None
        m = t.meta
        return CapabilityMetadata(
            name=t.name,
            invocation_mode=m.invocation_mode,
            is_long_running=m.is_long_running,
            is_idempotent=m.is_idempotent,
            requires_auth=m.requires_auth,
            timeout_s=m.timeout_s,
            side_effect_class=m.side_effect_class,
            owner=m.owner,
            version=m.version,
            status=m.status,
            tags=m.tags,
        )

    def list_capabilities_by_tag(self, tag: str) -> list["CapabilityInfo"]:
        """P3-compat: filter by meta.tags. v1 in-memory; P3 DB-backed.

        spec tool-invocation-interface#13
        """
        from nanobot.agent.tools.invoker import CapabilityInfo
        return [
            CapabilityInfo(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
                meta=t.meta,
            )
            for t in self._tools.values()
            if tag in t.meta.tags
        ]

    def list_capabilities_by_owner(self, owner: str) -> list["CapabilityInfo"]:
        """P3-compat: filter by meta.owner. v1 in-memory; P3 DB-backed.

        spec tool-invocation-interface#13
        """
        from nanobot.agent.tools.invoker import CapabilityInfo
        return [
            CapabilityInfo(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
                meta=t.meta,
            )
            for t in self._tools.values()
            if t.meta.owner == owner
        ]

    def check_auth(self, name: str, ctx: "AuthContext") -> "AuthResult":
        """P0 placeholder auth check. v1: requires_auth empty → allow; non-empty → deny.

        P3 will replace the deny-with-default-reason path with real policy evaluation.
        Signature MUST stay stable (spec tool-invocation-interface#14).
        """
        from nanobot.agent.tools.invoker import AuthResult
        meta = self.get_capability_metadata(name)
        if meta is None:
            return AuthResult(
                allowed=False,
                missing=frozenset(),
                reason=f"Tool '{name}' not found",
            )
        if not meta.requires_auth:
            return AuthResult(allowed=True, missing=frozenset(), reason=None)
        return AuthResult(
            allowed=False,
            missing=meta.requires_auth,
            reason="P0 阶段不实现鉴权语义（P3 实现）",
        )

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
