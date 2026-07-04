"""OpenAI-compatible HTTP API server for a fixed nanobot session.

Provides /v1/chat/completions and /v1/models endpoints.
All requests route to a single persistent API session.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from typing import Any

from aiohttp import web
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.config.paths import get_media_dir
from nanobot.utils.helpers import safe_filename
from nanobot.utils.media_decode import (
    MAX_FILE_SIZE,
)
from nanobot.utils.media_decode import (
    FileSizeExceeded as _FileSizeExceeded,
)
from nanobot.utils.media_decode import (
    save_base64_data_url as _save_base64_data_url,
)
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

__all__ = (
    "MAX_FILE_SIZE",
    "_FileSizeExceeded",
    "_save_base64_data_url",
    "create_app",
    "handle_chat_completions",
)


API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _error_json(status: int, message: str, err_type: str = "invalid_request_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": status}},
        status=status,
    )


def _chat_completion_response(
    content: str,
    model: str,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    prompt = (usage or {}).get("prompt_tokens", 0)
    completion = (usage or {}).get("completion_tokens", 0)
    total = (usage or {}).get("total_tokens", 0) or prompt + completion
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        },
    }


def _response_text(value: Any) -> str:
    """Normalize process_direct output to plain assistant text."""
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    return str(value)

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_chunk(delta: str, model: str, chunk_id: str, finish_reason: str | None = None) -> bytes:
    """Format a single OpenAI-compatible SSE chunk."""
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


_SSE_DONE = b"data: [DONE]\n\n"

# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def _parse_json_content(body: dict) -> tuple[str, list[str]]:
    """Parse JSON request body. Returns (text, media_paths)."""
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("Only a single user message is supported")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("Only a single user message is supported")

    user_content = message.get("content", "")
    media_dir = get_media_dir("api")
    media_paths: list[str] = []

    if isinstance(user_content, list):
        text_parts: list[str] = []
        for part in user_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    saved = _save_base64_data_url(url, media_dir)
                    if saved:
                        media_paths.append(saved)
                elif url:
                    raise ValueError(
                        "Remote image URLs are not supported. "
                        "Use base64 data URLs or upload files via multipart/form-data."
                    )
        text = " ".join(text_parts)
    elif isinstance(user_content, str):
        text = user_content
    else:
        raise ValueError("Invalid content format")

    return text, media_paths


async def _parse_multipart(request: web.Request) -> tuple[str, list[str], str | None, str | None]:
    """Parse multipart/form-data. Returns (text, media_paths, session_id, model)."""
    media_dir = get_media_dir("api")
    reader = await request.multipart()
    text = ""
    session_id = None
    model = None
    media_paths: list[str] = []

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "message":
            text = (await part.read()).decode("utf-8")
        elif part.name == "session_id":
            session_id = (await part.read()).decode("utf-8").strip()
        elif part.name == "model":
            model = (await part.read()).decode("utf-8").strip()
        elif part.name == "files":
            raw = await part.read()
            if len(raw) > MAX_FILE_SIZE:
                raise _FileSizeExceeded(
                    f"File '{part.filename}' exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit"
                )
            base = safe_filename(part.filename or "upload.bin")
            filename = f"{uuid.uuid4().hex[:12]}_{base}"
            dest = media_dir / filename
            dest.write_bytes(raw)
            media_paths.append(str(dest))

    if not text:
        text = "请分析上传的文件"

    return text, media_paths, session_id, model


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def handle_chat_completions(request: web.Request) -> web.Response:
    """POST /v1/chat/completions — supports JSON and multipart/form-data."""
    content_type = request.content_type or ""
    if not isinstance(content_type, str):
        content_type = ""

    agent_loop = request.app["agent_loop"]
    timeout_s: float = request.app.get("request_timeout", 120.0)
    model_name: str = request.app.get("model_name", "nanobot")

    stream = False
    try:
        if content_type.startswith("multipart/"):
            text, media_paths, session_id, requested_model = await _parse_multipart(request)
        else:
            try:
                body = await request.json()
            except Exception:
                return _error_json(400, "Invalid JSON body")
            stream = body.get("stream", False)
            requested_model = body.get("model")
            text, media_paths = _parse_json_content(body)
            session_id = body.get("session_id")
    except ValueError as e:
        return _error_json(400, str(e))
    except _FileSizeExceeded as e:
        return _error_json(413, str(e), err_type="invalid_request_error")
    except Exception:
        logger.exception("Error parsing upload")
        return _error_json(413, "File too large or invalid upload")

    if requested_model and requested_model != model_name:
        return _error_json(400, f"Only configured model '{model_name}' is available")

    session_key = f"api:{session_id}" if session_id else API_SESSION_KEY
    session_locks: dict[str, asyncio.Lock] = request.app["session_locks"]
    session_lock = session_locks.setdefault(session_key, asyncio.Lock())

    logger.info(
        "API request session_key={} media={} text={} stream={}",
        session_key, len(media_paths), text[:80], stream,
    )
    # -- streaming path --
    if stream:
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        await resp.prepare(request)

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        stream_failed = False
        emitted_content = False

        async def _on_stream(token: str) -> None:
            nonlocal emitted_content
            if token:
                emitted_content = True
            await queue.put(token)

        async def _on_stream_end(*_a: Any, **_kw: Any) -> None:
            # Agent stream-end callbacks mark generation segment boundaries.
            # Tool-backed requests may continue after a segment ends, so the
            # HTTP SSE stream is closed only when process_direct returns.
            return None

        async def _run() -> None:
            nonlocal stream_failed
            try:
                async with session_lock:
                    response = await asyncio.wait_for(
                        agent_loop.process_direct(
                            content=text,
                            media=media_paths if media_paths else None,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                            on_stream=_on_stream,
                            on_stream_end=_on_stream_end,
                        ),
                        timeout=timeout_s,
                    )
                    if not emitted_content:
                        response_text = _response_text(response)
                        if response_text.strip():
                            await queue.put(response_text)
            except Exception:
                stream_failed = True
                logger.exception("Streaming error for session {}", session_key)
            finally:
                await queue.put(None)

        task = asyncio.create_task(_run())
        try:
            while True:
                token = await queue.get()
                if token is None:
                    break
                await resp.write(_sse_chunk(token, model_name, chunk_id))
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        if not stream_failed:
            await resp.write(_sse_chunk("", model_name, chunk_id, finish_reason="stop"))
            await resp.write(_SSE_DONE)
        return resp

    # -- non-streaming path (original logic) --
    fallback = EMPTY_FINAL_RESPONSE_MESSAGE

    try:
        async with session_lock:
            try:
                response = await asyncio.wait_for(
                    agent_loop.process_direct(
                        content=text,
                        media=media_paths if media_paths else None,
                        session_key=session_key,
                        channel="api",
                        chat_id=API_CHAT_ID,
                    ),
                    timeout=timeout_s,
                )
                response_text = _response_text(response)

                if not response_text or not response_text.strip():
                    logger.warning("Empty response for session {}, retrying", session_key)
                    retry_response = await asyncio.wait_for(
                        agent_loop.process_direct(
                            content=text,
                            media=media_paths if media_paths else None,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                            persist_user_message=False,
                        ),
                        timeout=timeout_s,
                    )
                    response_text = _response_text(retry_response)
                    if not response_text or not response_text.strip():
                        logger.warning("Empty response after retry, using fallback")
                        response_text = fallback

            except asyncio.TimeoutError:
                return _error_json(504, f"Request timed out after {timeout_s}s")
            except Exception:
                logger.exception("Error processing request for session {}", session_key)
                return _error_json(500, "Internal server error", err_type="server_error")
    except Exception:
        logger.exception("Unexpected API lock error for session {}", session_key)
        return _error_json(500, "Internal server error", err_type="server_error")

    return web.json_response(
        _chat_completion_response(response_text, model_name, getattr(agent_loop, "_last_usage", None))
    )


async def handle_models(request: web.Request) -> web.Response:
    """GET /v1/models"""
    model_name = request.app.get("model_name", "nanobot")
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nanobot",
                }
            ],
        }
    )


async def handle_health(request: web.Request) -> web.Response:
    """GET /health"""
    return web.json_response({"status": "ok"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    agent_loop,
    model_name: str = "nanobot",
    request_timeout: float = 120.0,
    tool_registry=None,
    auth_token: str = "",
) -> web.Application:
    """Create the aiohttp application.

    Args:
        agent_loop: An initialized AgentLoop instance.
        model_name: Model name reported in responses.
        request_timeout: Per-request timeout in seconds.
        tool_registry: Optional ToolRegistry for /v1/tools/* routes.
        auth_token: Bearer token required for write ops on /v1/tools/*.
                    Empty string disables auth (dev mode).
    """
    app = web.Application(client_max_size=20 * 1024 * 1024)  # 20MB for base64 images
    app["agent_loop"] = agent_loop
    app["tool_registry"] = tool_registry
    app["auth_token"] = auth_token
    app["model_name"] = model_name
    app["request_timeout"] = request_timeout
    app["session_locks"] = {}  # per-user locks, keyed by session_key

    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)

    if tool_registry is not None:
        # /v1/tools/* — direct capability invocation (spec tool-invocation-interface#7, #8, #9)
        app.router.add_get("/v1/tools", handle_list_tools)
        app.router.add_get("/v1/tools/{name}", handle_describe_tool)
        app.router.add_post("/v1/tools/{name}/invoke", handle_invoke_tool)
        app.router.add_post("/v1/tools/{name}/register", handle_register_tool)
        app.router.add_delete("/v1/tools/{name}", handle_unregister_tool)
        # v1: register a 401 short-circuit middleware for write ops
        app.middlewares.append(_write_op_auth_middleware)

    return app


# -- Auth middleware (v1 minimal) — spec runtime-tool-registration#3, #4, #7 --


@web.middleware
async def _write_op_auth_middleware(request: web.Request, handler):
    """Authenticate write ops on /v1/tools/* (register/unregister/invoke).

    Read ops (GET /v1/tools, /v1/tools/{name}) MUST NOT be authenticated.
    aiohttp 3.14 new-style middleware via @web.middleware decorator.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return await handler(request)
    # Write op
    expected = request.app.get("auth_token", "")
    if not expected:
        # Dev mode: no token configured
        return await handler(request)
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[len("Bearer "):] != expected:
        return _error_json(401, "Unauthorized", "auth_error")
    return await handler(request)


# -- /v1/tools/* handlers — spec tool-invocation-interface#7, #8, #9 --


async def handle_list_tools(request: web.Request) -> web.Response:
    mode = request.query.get("mode")  # tag filter is v1 no-op (just ignored)
    registry = request.app["tool_registry"]
    items = registry.list_capabilities(mode=mode)
    return web.json_response(
        {
            "capabilities": [
                {
                    "name": i.name,
                    "description": i.description,
                    "parameters": i.parameters,
                    "meta": {
                        "invocation_mode": i.meta.invocation_mode,
                        "is_long_running": i.meta.is_long_running,
                        "is_idempotent": i.meta.is_idempotent,
                        "requires_auth": sorted(i.meta.requires_auth),
                        "timeout_s": i.meta.timeout_s,
                        "side_effect_class": i.meta.side_effect_class,
                        "owner": i.meta.owner,
                        "version": i.meta.version,
                        "status": i.meta.status,
                        "tags": sorted(i.meta.tags),
                    },
                }
                for i in items
            ]
        }
    )


async def handle_describe_tool(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    meta = request.app["tool_registry"].get_capability_metadata(name)
    if meta is None:
        return _error_json(404, f"Tool '{name}' not found", "not_found")
    return web.json_response(
        {
            "name": meta.name,
            "invocation_mode": meta.invocation_mode,
            "is_long_running": meta.is_long_running,
            "is_idempotent": meta.is_idempotent,
            "requires_auth": sorted(meta.requires_auth),
            "timeout_s": meta.timeout_s,
            "side_effect_class": meta.side_effect_class,
            "owner": meta.owner,
            "version": meta.version,
            "status": meta.status,
            "tags": sorted(meta.tags),
        }
    )


async def handle_invoke_tool(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return _error_json(400, "Invalid JSON body", "validation_error")
    params = body.get("params", {}) or {}
    timeout = body.get("timeout")
    if not isinstance(params, dict):
        return _error_json(400, "params must be a JSON object", "validation_error")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        return _error_json(400, "timeout must be a positive number", "validation_error")
    registry = request.app["tool_registry"]
    result = await registry.invoke(name, params, timeout=timeout)
    status = 200 if result.ok else 504 if "timeout" in (result.error or "") else 400 if "Invalid parameters" in (result.error or "") or "not found" in (result.error or "") else 500
    return web.json_response(
        {
            "ok": result.ok,
            "tool_name": result.tool_name,
            "content": result.content,
            "error": result.error,
            "duration_ms": result.duration_ms,
        },
        status=status,
    )


async def handle_register_tool(request: web.Request) -> web.Response:
    # spec runtime-tool-registration#3
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return _error_json(400, "Invalid JSON body", "validation_error")
    module = body.get("module")
    class_name = body.get("class_name")
    config = body.get("config", {})
    version = body.get("version", "0.1.0")
    if not module or not class_name:
        return _error_json(400, "Missing 'module' or 'class_name' field", "validation_error")
    try:
        mod = __import__(module, fromlist=[class_name])
        cls = getattr(mod, class_name)
    except ImportError as e:
        return _error_json(400, f"Cannot import module '{module}': {e}", "import_error")
    except AttributeError:
        return _error_json(400, f"Class '{class_name}' not found in module '{module}'", "import_error")
    if not (isinstance(cls, type) and issubclass(cls, Tool) and cls is not Tool):
        return _error_json(400, f"Class '{class_name}' is not a Tool subclass", "invalid_class")
    try:
        # Tools may require ToolContext; v1 minimal — try with config only
        tool = cls(**config) if config else cls()
    except TypeError as e:
        return _error_json(400, f"Failed to instantiate '{class_name}': {e}", "type_error")
    except Exception as e:
        return _error_json(400, f"Failed to instantiate '{class_name}': {e}", "instantiation_error")
    request.app["agent_loop"].tools.register(tool)
    return web.json_response(
        {"registered": tool.name, "version": version}, status=201
    )


async def handle_unregister_tool(request: web.Request) -> web.Response:
    # spec runtime-tool-registration#4
    name = request.match_info["name"]
    existed = request.app["agent_loop"].tools.has(name)
    if not existed:
        return _error_json(404, f"Tool '{name}' not registered", "not_found")
    request.app["agent_loop"].tools.unregister(name)
    return web.Response(status=204)
