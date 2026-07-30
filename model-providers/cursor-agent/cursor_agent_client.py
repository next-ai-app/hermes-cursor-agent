"""OpenAI-compatible shim that forwards Hermes requests to `cursor-agent`.

Each request spawns a short-lived headless Cursor Agent CLI process, sends the
formatted conversation as a single prompt, parses stream-json output, and
converts the result into the minimal shape Hermes expects from an OpenAI client.

This module lives in a user plugin under ``$HERMES_HOME/plugins/model-providers/``
so it survives ``hermes update`` (which never touches the user home).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_logger = logging.getLogger("hermes.plugins.cursor_agent.client")

try:
    from cursor_sdk import Agent as _SDKAgent, LocalAgentOptions as _SDKLocalOptions
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

CURSOR_MARKER_BASE_URL = "acp://cursor"


def _agent_ttl_seconds() -> float:
    """Idle TTL for cached SDK agents.

    A longer TTL keeps the cheap incremental path alive across natural chat
    pauses (every replay resends the full transcript), at the cost of idle
    cursor-sdk-bridge processes lingering a bit longer.
    """
    raw = os.getenv("HERMES_CURSOR_AGENT_TTL_SECONDS", "").strip()
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return 3600.0


_AGENT_TTL_SECONDS = _agent_ttl_seconds()

# After an SDK failure, route through the CLI for this long and then retry
# the SDK. Permanently disabling on first failure meant a single transient
# bridge hiccup downgraded the whole process to full-transcript-per-request
# CLI mode until restart.
_SDK_FAILURE_COOLDOWN_SECONDS = 300.0


def _default_timeout_seconds() -> float:
    raw = os.getenv("HERMES_CURSOR_TIMEOUT_SECONDS", "").strip()
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return 900.0


_DEFAULT_TIMEOUT_SECONDS = _default_timeout_seconds()

# `auto` is the only Cursor model guaranteed to be available on this account —
# the Composer tier ("composer-2.5", "composer-2.5-fast") is out of usage and
# returns nothing but an "out of usage. Switch to Auto" stderr status. Using it
# as the fallback caused empty/error replies (3x retries, ~30s wasted) and left
# the TUI streaming box half-rendered (garbled output). Default to `auto` and
# transparently recover to it when a model reports a usage limit.
_AUTO_MODEL = "auto"

_CURSOR_MODEL_LINE_RE = re.compile(r"^(\S+)\s+-\s+")
_USAGE_LIMIT_RE = re.compile(
    r"out of usage|switch to auto|increase your limit|usage limit", re.I
)


def _resolve_default_model() -> str:
    return os.getenv("HERMES_CURSOR_DEFAULT_MODEL", "").strip() or _AUTO_MODEL


def _is_usage_limit(text: str | None) -> bool:
    return bool(text and _USAGE_LIMIT_RE.search(text))


def _kill_process_group(proc: "subprocess.Popen[str]") -> None:
    """Kill the process AND its children.

    Subprocesses are spawned with ``start_new_session=True`` so the whole
    group can be signalled. A bare ``proc.kill()`` only kills the direct
    child; grandchildren (e.g. tools cursor-agent spawns) keep the stdout
    pipe open, so the reader stays blocked past the timeout.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_CURSOR_AGENT_COMMAND", "").strip()
        or os.getenv("CURSOR_AGENT_PATH", "").strip()
        or "cursor-agent"
    )


def _resolve_api_key(explicit: str | None = None) -> str:
    return (explicit or os.getenv("CURSOR_API_KEY", "") or "").strip()


def _resolve_home_dir() -> str:
    try:
        from hermes_constants import get_subprocess_home

        profile_home = get_subprocess_home()
        if profile_home:
            return profile_home
    except Exception:
        pass

    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok
        if resolved:
            return resolved
    except Exception:
        pass

    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = _resolve_home_dir()
    api_key = _resolve_api_key()
    if api_key:
        env["CURSOR_API_KEY"] = api_key
    return env


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
                if item_type == "image_url":
                    image_ref = item.get("image_url")
                    url = ""
                    if isinstance(image_ref, dict):
                        url = str(image_ref.get("url") or "").strip()
                    elif isinstance(image_ref, str):
                        url = image_ref.strip()
                    materialized = _materialize_image_url(url)
                    if materialized:
                        parts.append(
                            f"[User attached an image saved at: {materialized}. "
                            "Read and analyze this file to answer the request.]"
                        )
                    elif url:
                        parts.append(f"[User attached an image at: {url}]")
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _materialize_image_url(url: str) -> str | None:
    """Persist inline vision payloads so cursor-agent can read them from disk."""
    if not url:
        return None
    if url.startswith(("http://", "https://", "/")):
        return url
    if not url.startswith("data:"):
        return None

    header, _, payload = url.partition(",")
    if not payload:
        return None

    mime = "image/png"
    if ":" in header:
        mime = header.split(":", 1)[1].split(";", 1)[0].strip() or mime
    ext_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }
    suffix = ext_map.get(mime.lower(), ".png")
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception:
        return None
    if not raw:
        return None

    fd, path = tempfile.mkstemp(prefix="hermes-vision-", suffix=suffix)
    os.close(fd)
    try:
        Path(path).write_bytes(raw)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        return None
    return path


def _render_tool_calls_block(tool_calls: Any) -> str:
    """Render assistant tool calls so a replayed transcript keeps the link
    between each call (name + arguments) and its subsequent tool result."""
    if not isinstance(tool_calls, list):
        return ""
    parts: list[str] = []
    for call in tool_calls:
        if isinstance(call, dict):
            call_id = str(call.get("id") or "").strip()
            function = call.get("function")
            name = str(function.get("name") or "") if isinstance(function, dict) else ""
            arguments = function.get("arguments") if isinstance(function, dict) else None
        else:
            call_id = str(getattr(call, "id", "") or "").strip()
            function = getattr(call, "function", None)
            name = str(getattr(function, "name", "") or "")
            arguments = getattr(function, "arguments", None)
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments, ensure_ascii=True)
            except Exception:
                arguments = str(arguments)
        if not name and not arguments.strip():
            continue
        id_attr = f' id="{call_id}"' if call_id else ""
        parts.append(f'<tool_call{id_attr} name="{name}">\n{arguments}\n</tool_call>')
    return "\n".join(parts)


def _format_messages_as_prompt(messages: list[dict[str, Any]], *, model: str | None = None) -> str:
    """Format OpenAI-style messages into a structured prompt for cursor-agent.

    Uses XML-style role boundaries so the underlying model can clearly
    distinguish system instructions, user requests, and prior assistant turns
    — preserving the semantic structure that flat-text concatenation destroys.
    """
    system_parts: list[str] = []
    conversation: list[str] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        rendered = _render_message_content(message.get("content"))

        if role == "assistant":
            # An assistant turn may be tool calls only (content=None); it
            # must still appear in the transcript or the following tool
            # results lose their provenance.
            tool_calls_block = _render_tool_calls_block(message.get("tool_calls"))
            body = "\n".join(part for part in (rendered, tool_calls_block) if part)
            if body:
                conversation.append(f"<assistant>\n{body}\n</assistant>")
            continue

        if not rendered:
            continue

        if role == "system":
            system_parts.append(rendered)
        elif role == "user":
            conversation.append(f"<user>\n{rendered}\n</user>")
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "").strip()
            id_attr = f' call_id="{call_id}"' if call_id else ""
            conversation.append(f"<tool_result{id_attr}>\n{rendered}\n</tool_result>")
        else:
            conversation.append(f"<context>\n{rendered}\n</context>")

    sections: list[str] = [
        "<hermes_instructions>\n"
        "You are the active agent backend for Hermes.\n"
        "Complete the user's request using your available tools when needed.\n"
        "Respond with a clear, complete answer.\n"
        "</hermes_instructions>",
    ]

    if model:
        sections.append(f'<meta model="{model}" />')

    if system_parts:
        sections.append("<system>\n" + "\n\n".join(system_parts) + "\n</system>")

    if conversation:
        sections.append("<conversation>\n" + "\n\n".join(conversation) + "\n</conversation>")

    sections.append("Respond to the latest user message above.")
    return "\n\n".join(sections)


def _extract_assistant_content(event: dict[str, Any]) -> tuple[str, str]:
    """Extract text and reasoning content from an assistant stream event.

    Returns ``(text, reasoning)`` where either may be empty.  Handles
    ``thinking`` / ``reasoning`` content blocks emitted by thinking-capable
    models (Claude extended-thinking, o-series reasoning, etc.).
    """
    message = event.get("message")
    if not isinstance(message, dict):
        return "", ""
    content = message.get("content")
    if not isinstance(content, list):
        return "", ""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        elif item_type in ("thinking", "reasoning"):
            thinking = (
                item.get("thinking") or item.get("text") or item.get("content") or ""
            )
            if isinstance(thinking, str) and thinking:
                reasoning_parts.append(thinking)
    return "".join(text_parts), "".join(reasoning_parts)


def _extract_assistant_text(event: dict[str, Any]) -> str:
    text, _ = _extract_assistant_content(event)
    return text


def _make_stream_chunk(
    content: str | None,
    *,
    model: str,
    finish_reason: str | None = None,
    usage: Any | None = None,
    reasoning: str | None = None,
) -> Any:
    """Build an OpenAI ChatCompletionChunk-compatible object.

    The Hermes streaming consumer reads ``chunk.choices[0].delta.content``,
    ``chunk.choices[0].delta.tool_calls`` and ``chunk.choices[0].finish_reason``;
    ``chunk.usage`` / ``chunk.model`` are accessed defensively via ``hasattr``.
    ``reasoning`` / ``reasoning_content`` carry thinking-model chain-of-thought.
    """
    delta = SimpleNamespace(
        role=None,
        content=content,
        tool_calls=None,
        reasoning=reasoning,
        reasoning_content=reasoning,
        reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _sdk_run_error(result: Any) -> str | None:
    """Extract an error message from a finished SDK run result, if any.

    Without this check a failed run (e.g. out of usage) surfaced as a silent
    empty reply instead of triggering the CLI fallback.
    """
    if result is None:
        return None
    error = getattr(result, "error", None)
    subtype = str(getattr(result, "subtype", "") or "").strip().lower()
    is_error = bool(getattr(result, "is_error", False))
    if error or is_error or subtype in ("error", "failure", "failed"):
        detail = error or getattr(result, "result", None) or subtype or "unknown error"
        return str(detail)
    return None


def _usage_field(source: Any, *names: str) -> int:
    for name in names:
        value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _sdk_extract_usage(result: Any) -> Any | None:
    """Map SDK run usage onto the OpenAI usage shape Hermes reads.

    Real token counts feed Hermes' cost tracking and context-compression
    triggers; reporting zeros disabled both.
    """
    usage_data = getattr(result, "usage", None)
    if usage_data is None:
        return None
    input_tokens = _usage_field(usage_data, "input_tokens", "inputTokens", "prompt_tokens")
    output_tokens = _usage_field(
        usage_data, "output_tokens", "outputTokens", "completion_tokens"
    )
    if not input_tokens and not output_tokens:
        return None
    cached = _usage_field(usage_data, "cache_read_tokens", "cacheReadTokens", "cached_tokens")
    return SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def _zero_usage() -> Any:
    return SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )


def parse_cursor_stream_line(line: str) -> dict[str, Any] | None:
    stripped = (line or "").strip()
    if not stripped or not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_cursor_stream_events(lines: list[str]) -> tuple[str, str | None, str, Any]:
    """Parse NDJSON lines from cursor-agent stream-json output.

    Returns ``(response_text, error_message, reasoning_text, usage)``.
    Prefers the canonical ``result`` event; falls back to concatenated
    streaming assistant deltas.  Reasoning (thinking) tokens and real
    usage stats from the ``result`` event are captured and returned.
    """
    streaming_parts: list[str] = []
    reasoning_parts: list[str] = []
    final_result = ""
    error_message: str | None = None
    usage: Any = None

    for line in lines:
        event = parse_cursor_stream_line(line)
        if not event:
            continue

        event_type = str(event.get("type") or "").strip()
        if event_type == "assistant":
            has_ts = "timestamp_ms" in event
            has_model_call = "model_call_id" in event
            if has_ts and not has_model_call:
                chunk_text, chunk_reasoning = _extract_assistant_content(event)
                if chunk_text:
                    streaming_parts.append(chunk_text)
                if chunk_reasoning:
                    reasoning_parts.append(chunk_reasoning)
            elif not has_ts and not has_model_call:
                chunk_text, chunk_reasoning = _extract_assistant_content(event)
                if chunk_text and not streaming_parts:
                    final_result = chunk_text
                if chunk_reasoning:
                    reasoning_parts.append(chunk_reasoning)
        elif event_type == "result":
            subtype = str(event.get("subtype") or "").strip().lower()
            usage_data = event.get("usage")
            if isinstance(usage_data, dict):
                input_tokens = usage_data.get("inputTokens", 0)
                output_tokens = usage_data.get("outputTokens", 0)
                usage = SimpleNamespace(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    prompt_tokens_details=SimpleNamespace(
                        cached_tokens=usage_data.get("cacheReadTokens", 0),
                    ),
                )
            if subtype == "success":
                result_text = event.get("result")
                if isinstance(result_text, str) and result_text.strip():
                    final_result = result_text.strip()
            else:
                err = event.get("error") or event.get("message") or event.get("result")
                if isinstance(err, str) and err.strip():
                    error_message = err.strip()
                elif err is not None:
                    error_message = str(err)

    reasoning = "".join(reasoning_parts).strip()

    if final_result:
        return final_result, error_message, reasoning, usage

    streamed = "".join(streaming_parts).strip()
    return streamed, error_message, reasoning, usage


def parse_cursor_list_models_output(stdout: str) -> list[str]:
    models: list[str] = []
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("available models"):
            continue
        match = _CURSOR_MODEL_LINE_RE.match(stripped)
        if match:
            model_id = match.group(1).strip()
            if model_id and model_id not in models:
                models.append(model_id)
    return models


_MODELS_CACHE_TTL_SECONDS = 300.0
_models_cache: dict[str, Any] = {"at": 0.0, "ids": None}


def _fetch_cursor_models_sdk(*, api_key: str | None = None) -> list[str] | None:
    """Model ids from the Cursor SDK catalog (``Cursor.models.list``).

    The SDK path is what actually serves chat turns, so its catalog — not
    the CLI's variant-flavoured ``--list-models`` output — is the
    authoritative list for the /model picker.
    """
    if not _SDK_AVAILABLE:
        return None
    try:
        from cursor_sdk import Cursor

        rows = Cursor.models.list(api_key=api_key or _resolve_api_key() or None)
    except Exception as exc:
        _logger.debug("SDK model listing failed (%s); falling back to CLI", exc)
        return None
    ids: list[str] = []
    for row in rows or ():
        model_id = str(getattr(row, "id", "") or "").strip()
        if model_id == "default":
            # The send path accepts "auto" for the default model, and that is
            # the id users already have in config.yaml.
            model_id = "auto"
        if model_id and model_id not in ids:
            ids.append(model_id)
    return ids or None


def _fetch_cursor_models_cli(
    *,
    command: str | None = None,
    api_key: str | None = None,
    timeout: float = 15.0,
) -> list[str] | None:
    resolved_command = command or _resolve_command()
    resolved = shutil.which(resolved_command)
    if not resolved:
        return None

    env = _build_subprocess_env()
    if api_key:
        env["CURSOR_API_KEY"] = api_key

    try:
        proc = subprocess.run(
            [resolved, "--list-models"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    models = parse_cursor_list_models_output(proc.stdout or "")
    return models or None


def fetch_cursor_models(
    *,
    command: str | None = None,
    api_key: str | None = None,
    timeout: float = 15.0,
    force_refresh: bool = False,
) -> list[str] | None:
    """Fetch the Cursor model catalog: SDK first, CLI fallback, short cache.

    Both backends take seconds (SDK HTTP call / CLI subprocess), and the
    /model picker may query more than once per interaction, so successful
    results are cached briefly.
    """
    now = time.monotonic()
    cached = _models_cache.get("ids")
    if (
        not force_refresh
        and cached
        and now - float(_models_cache.get("at") or 0.0) < _MODELS_CACHE_TTL_SECONDS
    ):
        return list(cached)

    models = _fetch_cursor_models_sdk(api_key=api_key)
    if not models:
        models = _fetch_cursor_models_cli(
            command=command, api_key=api_key, timeout=timeout
        )
    if models:
        _models_cache["at"] = now
        _models_cache["ids"] = list(models)
    return models or None


def _sdk_normalize_model(model: str, *, api_key: str | None = None) -> str:
    """Map a CLI variant name to the base id the SDK catalog accepts.

    The CLI advertises ~193 variant ids (``claude-fable-5-thinking-high``,
    ``gpt-5.5-medium``…) but the SDK API only accepts ~33 base ids
    (``claude-fable-5``, ``gpt-5.5``…) and rejects variants with
    ``invalid_argument: Cannot use this model``. Normalizing here avoids a
    guaranteed SDK failure (and the pointless 300s CLI cooldown it triggers)
    on every turn that uses a variant name. Unknown names are returned
    unchanged so the real API error still surfaces.
    """
    if not model:
        return model
    catalog = fetch_cursor_models(api_key=api_key)
    if not catalog:
        return model
    if model == "auto" and "auto" not in catalog:
        # The send path's default-model id is "default"; the catalog display
        # maps it to "auto", so reverse-map here.
        return "default"
    if model in catalog:
        return model
    # Longest catalog id that is a prefix of the requested variant, e.g.
    # "gpt-5.4-mini-high" -> "gpt-5.4-mini" (not "gpt-5.4").
    candidates = [c for c in catalog if c != "auto" and model.startswith(c + "-")]
    if candidates:
        base = max(candidates, key=len)
        _logger.info(
            "SDK catalog has no %r; using its base model %r for the SDK path",
            model,
            base,
        )
        return base
    return model


class _CursorChatCompletions:
    def __init__(self, client: "CursorAgentClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        client = self._client
        is_stream = kwargs.get("stream")

        # Try SDK path first for proper multi-turn conversation support.
        if _SDK_AVAILABLE and client._sdk_active():
            if is_stream:
                # Generators run lazily: returning the raw SDK generator here
                # would move all SDK errors outside this try/except (they only
                # fire once Hermes iterates). The wrapper catches them at
                # iteration time and still falls back to the CLI.
                return client._sdk_stream_with_cli_fallback(**kwargs)
            try:
                return client._sdk_chat_completion(**kwargs)
            except Exception as exc:
                client._mark_sdk_failure("completion", exc)

        if is_stream:
            return client._create_chat_completion_stream(**kwargs)
        return client._create_chat_completion(**kwargs)


class _AsyncCursorChatCompletions:
    """Async wrapper so auxiliary vision paths can ``await`` cursor-agent."""

    def __init__(self, sync_completions: _CursorChatCompletions):
        self._sync = sync_completions

    async def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            raise NotImplementedError(
                "cursor-agent async streaming is not supported via auxiliary_client"
            )
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncCursorChatNamespace:
    def __init__(self, completions: _AsyncCursorChatCompletions):
        self.completions = completions


class AsyncCursorAgentClient:
    """Async-compatible facade matching ``AsyncOpenAI.chat.completions.create()``."""

    def __init__(self, sync_client: "CursorAgentClient"):
        self._sync_client = sync_client
        async_completions = _AsyncCursorChatCompletions(sync_client.chat.completions)
        self.chat = _AsyncCursorChatNamespace(async_completions)
        self.api_key = sync_client.api_key
        self.base_url = sync_client.base_url

    def close(self) -> None:
        self._sync_client.close()


class _CursorChatNamespace:
    def __init__(self, client: "CursorAgentClient"):
        self.completions = _CursorChatCompletions(client)


class CursorAgentClient:
    """Minimal OpenAI-client-compatible facade for cursor-agent."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        cursor_command: str | None = None,
        cursor_args: list[str] | None = None,
        cursor_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ):
        self.api_key = api_key or _resolve_api_key() or "cursor-agent"
        self.base_url = base_url or CURSOR_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._cursor_command = cursor_command or command or _resolve_command()
        self._cursor_args = list(cursor_args or args or [])
        self._cursor_cwd = str(Path(cursor_cwd or os.getcwd()).resolve())
        self.chat = _CursorChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        # SDK multi-turn agent cache
        self._sdk_agents: dict[str, Any] = {}
        self._sdk_activity: dict[str, float] = {}
        self._sdk_msg_counts: dict[str, int] = {}
        # Running tally of cheap incremental turns vs full-transcript
        # replays, logged on each replay so a chronically desyncing setup
        # (e.g. over-eager context compression) is visible in the logs.
        self._sdk_turn_stats: dict[str, int] = {"incremental": 0, "replay": 0}
        self._sdk_lock = threading.Lock()
        # monotonic deadline until which the SDK path is skipped (0 = active)
        self._sdk_disabled_until = 0.0

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        with self._sdk_lock:
            for agent in self._sdk_agents.values():
                try:
                    agent.close()
                except Exception:
                    pass
            self._sdk_agents.clear()
            self._sdk_activity.clear()
            self._sdk_msg_counts.clear()

    # ------------------------------------------------------------------
    # SDK multi-turn agent methods
    # ------------------------------------------------------------------

    def _sdk_active(self) -> bool:
        return time.monotonic() >= self._sdk_disabled_until

    def _mark_sdk_failure(self, where: str, exc: BaseException) -> None:
        self._sdk_disabled_until = time.monotonic() + _SDK_FAILURE_COOLDOWN_SECONDS
        _logger.warning(
            "SDK %s failed (%s); using CLI for the next %.0fs",
            where,
            exc,
            _SDK_FAILURE_COOLDOWN_SECONDS,
        )

    def _sdk_stream_with_cli_fallback(self, **kwargs: Any) -> Any:
        """Iterate the SDK stream, falling back to the CLI on failure.

        Errors raised before the first chunk reaches the consumer are fully
        recoverable, so the whole turn is retried via the CLI. Once chunks
        have been delivered a CLI replay would duplicate visible output, so
        the error is surfaced for this turn only (the cooldown makes the next
        turn take the CLI path).
        """
        yielded = False
        try:
            for chunk in self._sdk_chat_completion_stream(**kwargs):
                yielded = True
                yield chunk
            return
        except Exception as exc:
            self._mark_sdk_failure("stream", exc)
            if yielded:
                raise
        yield from self._create_chat_completion_stream(**kwargs)

    def _conv_key(self, messages: list[dict[str, Any]], model: str) -> str:
        """Fingerprint a conversation for the agent cache.

        Hashes the *full* first system and first user message — Hermes system
        prompts share identical openings across sessions, so truncated
        prefixes collided and let unrelated sessions steal each other's
        agents. The model is part of the key so a mid-session /model switch
        gets a fresh agent (with full replay) instead of silently reusing an
        agent created for the old model.
        """
        first_system = ""
        first_user = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system" and not first_system:
                first_system = _render_message_content(msg.get("content"))
            elif role == "user" and not first_user:
                first_user = _render_message_content(msg.get("content"))
            if first_system and first_user:
                break
        return hashlib.sha256(
            f"{first_system}\x00{first_user}\x00{model}".encode()
        ).hexdigest()[:16]

    def _sdk_evict_locked(self, conv_key: str) -> Any | None:
        """Drop a cached agent; caller holds ``_sdk_lock`` and closes it."""
        agent = self._sdk_agents.pop(conv_key, None)
        self._sdk_activity.pop(conv_key, None)
        self._sdk_msg_counts.pop(conv_key, None)
        return agent

    def _sdk_evict(self, conv_key: str, expected: Any | None = None) -> None:
        """Remove and close a cached agent (no-op if ``expected`` mismatches)."""
        with self._sdk_lock:
            if expected is not None and self._sdk_agents.get(conv_key) is not expected:
                return
            agent = self._sdk_evict_locked(conv_key)
        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass

    def _sdk_start_watchdog(
        self, conv_key: str, agent: Any, run: Any, timeout_seconds: float
    ) -> tuple[threading.Event, threading.Event]:
        """Abort a hung SDK run after ``timeout_seconds``.

        ``run.stream()`` reads block indefinitely; without a watchdog the
        whole Hermes turn froze forever (the CLI path already had one).
        """
        done = threading.Event()
        timed_out = threading.Event()

        def _watchdog() -> None:
            if done.wait(timeout_seconds):
                return
            timed_out.set()
            for target in (run, agent):
                for name in ("cancel", "kill", "close"):
                    method = getattr(target, name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception:
                            pass
                        break
            # The agent was force-closed; drop it so the next turn replays.
            self._sdk_evict(conv_key, expected=agent)

        threading.Thread(target=_watchdog, daemon=True).start()
        return done, timed_out

    def _cleanup_stale_agents(self) -> None:
        now = time.monotonic()
        stale = [k for k, t in self._sdk_activity.items() if now - t > _AGENT_TTL_SECONDS]
        for key in stale:
            agent = self._sdk_evict_locked(key)
            if agent is not None:
                try:
                    agent.close()
                except Exception:
                    pass

    def _sdk_get_or_create_agent(
        self, conv_key: str, model: str
    ) -> tuple[Any, bool]:
        """Return ``(agent, is_new_conversation)``."""
        with self._sdk_lock:
            self._cleanup_stale_agents()
            if conv_key in self._sdk_agents:
                self._sdk_activity[conv_key] = time.monotonic()
                return self._sdk_agents[conv_key], False

            api_key = _resolve_api_key(
                self.api_key if self.api_key != "cursor-agent" else None
            )
            agent = _SDKAgent.create(
                model=model,
                api_key=api_key or None,
                local=_SDKLocalOptions(cwd=self._cursor_cwd),
            )
            self._sdk_agents[conv_key] = agent
            self._sdk_activity[conv_key] = time.monotonic()
            self._sdk_msg_counts[conv_key] = 0
            return agent, True

    def _sdk_prepare_turn(
        self,
        conv_key: str,
        model: str,
        msgs: list[dict[str, Any]],
    ) -> tuple[Any, str]:
        """Return ``(agent, prompt)`` via a hybrid incremental/replay strategy.

        The cheap incremental path (send only the new user message) is taken
        only when the cached SDK agent is provably in sync with Hermes'
        message array: since the last successful turn, exactly our own
        assistant reply plus one new user message were appended. Anything
        else — context compression, /undo, session resume, process restart,
        tool results in the tail, TTL eviction — closes the stale agent and
        replays the full transcript via ``_format_messages_as_prompt`` so no
        history is ever silently dropped.
        """
        stale_agent: Any | None = None
        with self._sdk_lock:
            self._cleanup_stale_agents()
            agent = self._sdk_agents.get(conv_key)
            synced = self._sdk_msg_counts.get(conv_key)
            if (
                agent is not None
                and synced is not None
                and len(msgs) == synced + 2
                and isinstance(msgs[-2], dict)
                and msgs[-2].get("role") == "assistant"
                and isinstance(msgs[-1], dict)
                and msgs[-1].get("role") == "user"
            ):
                prompt = _render_message_content(msgs[-1].get("content"))
                if prompt:
                    self._sdk_activity[conv_key] = time.monotonic()
                    self._sdk_turn_stats["incremental"] += 1
                    return agent, prompt
            if agent is not None:
                _logger.info(
                    "SDK agent desynced (conv=%s synced=%s incoming=%d); "
                    "replaying full transcript",
                    conv_key,
                    synced,
                    len(msgs),
                )
                stale_agent = self._sdk_evict_locked(conv_key)
        if stale_agent is not None:
            try:
                stale_agent.close()
            except Exception:
                pass

        with self._sdk_lock:
            self._sdk_turn_stats["replay"] += 1
            inc = self._sdk_turn_stats["incremental"]
            rep = self._sdk_turn_stats["replay"]
        _logger.info(
            "SDK turn stats: %d incremental / %d replay (%.0f%% incremental)",
            inc,
            rep,
            100.0 * inc / (inc + rep),
        )

        agent, _ = self._sdk_get_or_create_agent(conv_key, model)
        return agent, _format_messages_as_prompt(msgs, model=model)

    def _sdk_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> Any:
        """Non-streaming completion via Cursor SDK (multi-turn)."""
        msgs = messages or []
        resolved_model = _sdk_normalize_model(model or _resolve_default_model())
        conv_key = self._conv_key(msgs, resolved_model)
        agent, prompt = self._sdk_prepare_turn(conv_key, resolved_model, msgs)
        if not prompt:
            raise RuntimeError("No user message found in messages")

        effective_timeout = self._resolve_timeout(timeout)
        run = agent.send(prompt)
        done, timed_out = self._sdk_start_watchdog(
            conv_key, agent, run, effective_timeout
        )

        response_text = ""
        reasoning_text = ""

        try:
            for msg in run.stream():
                msg_type = getattr(msg, "type", "")
                if msg_type == "assistant":
                    message_obj = getattr(msg, "message", None)
                    if message_obj:
                        for block in getattr(message_obj, "content", []):
                            if getattr(block, "type", "") == "text":
                                response_text += getattr(block, "text", "")
                elif msg_type == "thinking":
                    thinking = getattr(msg, "text", "")
                    if thinking:
                        reasoning_text += thinking

            result = run.wait()
        except Exception as exc:
            self._sdk_evict(conv_key, expected=agent)
            if timed_out.is_set():
                raise TimeoutError(
                    f"cursor-agent SDK turn timed out after {effective_timeout:.0f}s"
                ) from exc
            raise
        finally:
            done.set()

        if timed_out.is_set():
            self._sdk_evict(conv_key, expected=agent)
            raise TimeoutError(
                f"cursor-agent SDK turn timed out after {effective_timeout:.0f}s"
            )

        run_error = _sdk_run_error(result)
        if not run_error and getattr(result, "result", None):
            response_text = result.result
        if run_error and not response_text:
            # Failed run leaves the agent state unknown; force a clean replay.
            self._sdk_evict(conv_key, expected=agent)
            raise RuntimeError(f"cursor-agent SDK run failed: {run_error}")
        if run_error:
            _logger.warning("SDK run reported an error after partial output: %s", run_error)

        if not response_text and not reasoning_text:
            # A "successful" run with no text and no error means the SDK
            # transport is dead (observed in production: instant empty
            # replies on every turn, across brand-new sessions, until the
            # whole process was restarted). Returning "" here surfaces in
            # Hermes as a bare "Empty response" with no fallback — raise so
            # the caller routes this turn (and, via the failure cooldown,
            # the next few minutes of turns) to the CLI instead.
            self._sdk_evict(conv_key, expected=agent)
            _logger.warning(
                "SDK run returned empty content without an error "
                "(subtype=%r is_error=%r result=%r); treating as SDK failure",
                getattr(result, "subtype", None),
                getattr(result, "is_error", None),
                getattr(result, "result", None),
            )
            raise RuntimeError(
                "cursor-agent SDK returned an empty response without an "
                f"error (model={resolved_model}); the SDK transport is "
                "likely stuck — retrying this turn via the CLI path."
            )

        with self._sdk_lock:
            # Mark the agent as synced up to the full incoming array. Hermes
            # appends our assistant reply next, so the following turn is
            # incremental only when it arrives as exactly +assistant +user.
            self._sdk_msg_counts[conv_key] = len(msgs)

        usage = _sdk_extract_usage(result) or _zero_usage()
        assistant_message = SimpleNamespace(
            content=response_text,
            tool_calls=[],
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=resolved_model,
        )

    def _sdk_chat_completion_stream(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> Any:
        """Streaming completion via Cursor SDK — yields OpenAI chunks."""
        msgs = messages or []
        resolved_model = _sdk_normalize_model(model or _resolve_default_model())
        conv_key = self._conv_key(msgs, resolved_model)
        agent, prompt = self._sdk_prepare_turn(conv_key, resolved_model, msgs)
        if not prompt:
            raise RuntimeError("No user message found in messages")

        effective_timeout = self._resolve_timeout(timeout)
        run = agent.send(prompt)
        done, timed_out = self._sdk_start_watchdog(
            conv_key, agent, run, effective_timeout
        )

        yielded_text = False
        try:
            for msg in run.stream():
                msg_type = getattr(msg, "type", "")
                if msg_type == "assistant":
                    message_obj = getattr(msg, "message", None)
                    if message_obj:
                        for block in getattr(message_obj, "content", []):
                            if getattr(block, "type", "") == "text":
                                text = getattr(block, "text", "")
                                if text:
                                    yielded_text = True
                                    yield _make_stream_chunk(text, model=resolved_model)
                elif msg_type == "thinking":
                    thinking = getattr(msg, "text", "")
                    if thinking:
                        yield _make_stream_chunk(
                            None, model=resolved_model, reasoning=thinking
                        )

            result = run.wait()
        except Exception as exc:
            self._sdk_evict(conv_key, expected=agent)
            if timed_out.is_set():
                raise TimeoutError(
                    f"cursor-agent SDK turn timed out after {effective_timeout:.0f}s"
                ) from exc
            raise
        finally:
            done.set()

        if timed_out.is_set():
            self._sdk_evict(conv_key, expected=agent)
            raise TimeoutError(
                f"cursor-agent SDK turn timed out after {effective_timeout:.0f}s"
            )

        run_error = _sdk_run_error(result)
        if run_error and not yielded_text:
            # Nothing visible was emitted; evict and raise so the stream
            # wrapper can retry the whole turn via the CLI.
            self._sdk_evict(conv_key, expected=agent)
            raise RuntimeError(f"cursor-agent SDK run failed: {run_error}")
        if run_error:
            _logger.warning("SDK run reported an error after partial output: %s", run_error)

        if not yielded_text:
            # Some SDK runs emit no assistant deltas and put the whole reply
            # in the final result (the CLI path handles the same case via
            # its `result` event). Emit it as one chunk so the reply isn't
            # lost.
            final_text = str(getattr(result, "result", "") or "").strip()
            if final_text:
                yield _make_stream_chunk(final_text, model=resolved_model)
                yielded_text = True

        if not yielded_text and not run_error:
            # Same dead-transport case as the non-streaming path: a
            # "successful" run that emitted nothing. Raising before any
            # chunk lets _sdk_stream_with_cli_fallback retry via the CLI.
            self._sdk_evict(conv_key, expected=agent)
            _logger.warning(
                "SDK stream yielded no content and no error "
                "(subtype=%r is_error=%r); treating as SDK failure",
                getattr(result, "subtype", None),
                getattr(result, "is_error", None),
            )
            raise RuntimeError(
                "cursor-agent SDK streamed an empty response without an "
                f"error (model={resolved_model}); the SDK transport is "
                "likely stuck — retrying this turn via the CLI path."
            )

        with self._sdk_lock:
            # Mark the agent as synced up to the full incoming array. Hermes
            # appends our assistant reply next, so the following turn is
            # incremental only when it arrives as exactly +assistant +user.
            self._sdk_msg_counts[conv_key] = len(msgs)

        yield _make_stream_chunk(
            None,
            model=resolved_model,
            finish_reason="stop",
            usage=_sdk_extract_usage(result) or _zero_usage(),
        )

    # ------------------------------------------------------------------
    # CLI-based methods (fallback when SDK unavailable)
    # ------------------------------------------------------------------

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(messages or [], model=model)
        if timeout is None:
            effective_timeout = _default_timeout_seconds()
        elif isinstance(timeout, (int, float)):
            effective_timeout = float(timeout)
        else:
            candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
            effective_timeout = max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS

        response_text, reasoning_text, real_usage = self._run_prompt(
            prompt_text,
            model=model or _resolve_default_model(),
            timeout_seconds=effective_timeout,
        )

        usage = real_usage or SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=response_text,
            tool_calls=[],
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or _resolve_default_model(),
        )

    def _resolve_timeout(self, timeout: Any) -> float:
        if timeout is None:
            return _default_timeout_seconds()
        if isinstance(timeout, (int, float)):
            return float(timeout)
        candidates = [
            getattr(timeout, attr, None)
            for attr in ("read", "write", "connect", "pool", "timeout")
        ]
        numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
        return max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS

    def _create_chat_completion_stream(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> Any:
        """Yield OpenAI-style chunks as cursor-agent streams its reply.

        Returns a generator so Hermes' streaming consumer fires
        ``stream_delta_callback`` per fragment, which flows through the
        gateway SSE writer to the frontend in real time.
        """
        prompt_text = _format_messages_as_prompt(messages or [], model=model)
        effective_timeout = self._resolve_timeout(timeout)
        model_name = model or _resolve_default_model()
        return self._run_prompt_stream(
            prompt_text,
            model=model_name,
            timeout_seconds=effective_timeout,
        )

    def _run_prompt_stream(self, prompt_text: str, *, model: str, timeout_seconds: float):
        command = self._build_command(model=model, stream_partial=True)
        stderr_tail: deque[str] = deque(maxlen=40)

        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._cursor_cwd,
                env=_build_subprocess_env(),
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Cursor Agent command '{self._cursor_command}'. "
                "Install cursor-agent or set HERMES_CURSOR_AGENT_COMMAND/CURSOR_AGENT_PATH."
            ) from exc

        if proc.stdout is None or proc.stderr is None:
            proc.kill()
            raise RuntimeError("cursor-agent did not expose stdout/stderr pipes.")

        self._spawn_stdin_writer(proc, prompt_text)

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        err_thread.start()

        # Wall-clock watchdog: stream-json reads are line-driven, so kill a
        # hung process after the timeout instead of blocking the iterator
        # forever.
        done_event = threading.Event()

        def _watchdog() -> None:
            if done_event.wait(timeout_seconds):
                return
            if proc.poll() is None:
                _kill_process_group(proc)

        watch_thread = threading.Thread(target=_watchdog, daemon=True)
        watch_thread.start()

        streamed_any = False
        final_result = ""
        error_message: str | None = None
        accumulated_reasoning = ""
        real_usage: Any = None

        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                event = parse_cursor_stream_line(raw_line)
                if not event:
                    continue
                event_type = str(event.get("type") or "").strip()
                if event_type == "assistant":
                    has_ts = "timestamp_ms" in event
                    has_model_call = "model_call_id" in event
                    if has_ts and not has_model_call:
                        chunk_text, chunk_reasoning = _extract_assistant_content(event)
                        if chunk_text or chunk_reasoning:
                            streamed_any = True
                            yield _make_stream_chunk(
                                chunk_text or None,
                                model=model,
                                reasoning=chunk_reasoning or None,
                            )
                    elif not has_ts and not has_model_call:
                        chunk_text, chunk_reasoning = _extract_assistant_content(event)
                        if chunk_text and not final_result:
                            final_result = chunk_text
                        if chunk_reasoning:
                            accumulated_reasoning += chunk_reasoning
                elif event_type == "result":
                    subtype = str(event.get("subtype") or "").strip().lower()
                    usage_data = event.get("usage")
                    if isinstance(usage_data, dict):
                        input_tokens = usage_data.get("inputTokens", 0)
                        output_tokens = usage_data.get("outputTokens", 0)
                        real_usage = SimpleNamespace(
                            prompt_tokens=input_tokens,
                            completion_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens,
                            prompt_tokens_details=SimpleNamespace(
                                cached_tokens=usage_data.get("cacheReadTokens", 0),
                            ),
                        )
                    if subtype == "success":
                        result_text = event.get("result")
                        if isinstance(result_text, str) and result_text.strip():
                            final_result = result_text.strip()
                    else:
                        err = event.get("error") or event.get("message") or event.get("result")
                        if isinstance(err, str) and err.strip():
                            error_message = err.strip()
                        elif err is not None:
                            error_message = str(err)
        finally:
            done_event.set()
            err_thread.join(timeout=1)
            with self._active_process_lock:
                if self._active_process is proc:
                    self._active_process = None
            self.is_closed = True
            if proc.poll() is None:
                _kill_process_group(proc)

        stderr_text = "\n".join(stderr_tail).strip()

        # Transparent recovery: an out-of-usage model streams nothing but an
        # "out of usage. Switch to Auto" status. Nothing has been yielded yet,
        # so re-stream the same prompt with `auto` instead of erroring (and
        # leaving the TUI streaming box half-rendered).
        if (
            not streamed_any
            and not final_result
            and model != _AUTO_MODEL
            and (_is_usage_limit(stderr_text) or _is_usage_limit(error_message))
        ):
            yield from self._run_prompt_stream(
                prompt_text, model=_AUTO_MODEL, timeout_seconds=timeout_seconds
            )
            return

        # No incremental deltas reached the client (short replies, or a backend
        # that only emitted the canonical ``result`` event).  Emit the
        # aggregated text as a single chunk so the reply is never empty.
        if not streamed_any:
            if final_result:
                yield _make_stream_chunk(
                    final_result,
                    model=model,
                    reasoning=accumulated_reasoning or None,
                )
            elif error_message:
                raise RuntimeError(f"cursor-agent failed: {error_message}")
            elif proc.returncode not in (0, None):
                detail = stderr_text or f"exit code {proc.returncode}"
                raise RuntimeError(f"cursor-agent exited without a response: {detail}")
            elif stderr_text:
                raise RuntimeError(f"cursor-agent produced no response: {stderr_text}")
            else:
                # Exit 0 with no stdout AND no stderr: the CLI swallowed the
                # failure (seen with models pending a data-retention-policy
                # acknowledgement — the account-level ActionRequiredError only
                # appears on some paths). Hermes would otherwise report a
                # bare "Empty response"; give the user the actionable cause.
                raise RuntimeError(
                    f"cursor-agent returned an empty response (exit code 0, "
                    f"no output, model={model}). The model may be unavailable "
                    "on this account or require acknowledgement in Cursor "
                    "(e.g. a data-retention policy) — try /model to pick "
                    "another one."
                )

        final_usage = real_usage or SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        yield _make_stream_chunk(None, model=model, finish_reason="stop", usage=final_usage)

    def _build_command(self, *, model: str, stream_partial: bool = False) -> list[str]:
        cmd = [
            self._cursor_command,
            "-p",
            "--output-format",
            "stream-json",
            "--force",
            "--trust",
            "--workspace",
            self._cursor_cwd,
            "--model",
            model,
        ]
        # Emit incremental text deltas (one assistant event per fragment)
        # instead of a single consolidated message, so the reply streams to
        # the user token-by-token.
        if stream_partial:
            cmd.append("--stream-partial-output")
        api_key = _resolve_api_key(self.api_key if self.api_key != "cursor-agent" else None)
        if api_key:
            cmd.extend(["--api-key", api_key])
        cmd.extend(self._cursor_args)
        # NOTE: the prompt is deliberately NOT appended as a positional CLI arg.
        # A long conversation transcript can exceed Linux MAX_ARG_STRLEN (128 KiB
        # per single argv string) and raise OSError [Errno 7] Argument list too
        # long before the process even starts. The prompt is fed via stdin
        # instead (cursor-agent reads the prompt from stdin in --print mode).
        return cmd

    @staticmethod
    def _spawn_stdin_writer(proc: "subprocess.Popen[str]", prompt_text: str) -> threading.Thread:
        """Write the prompt to the child's stdin from a dedicated thread.

        Writing inline would deadlock when ``prompt_text`` exceeds the OS pipe
        buffer (~64 KiB): the child blocks writing stdout while we block writing
        stdin. A separate writer thread lets stdout drain concurrently.
        """

        def _writer() -> None:
            if proc.stdin is None:
                return
            try:
                proc.stdin.write(prompt_text)
                proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, ValueError, OSError):
                    pass

        thread = threading.Thread(target=_writer, daemon=True)
        thread.start()
        return thread

    def _run_prompt(self, prompt_text: str, *, model: str, timeout_seconds: float) -> tuple[str, str, Any]:
        command = self._build_command(model=model)
        stderr_tail: deque[str] = deque(maxlen=40)

        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._cursor_cwd,
                env=_build_subprocess_env(),
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Cursor Agent command '{self._cursor_command}'. "
                "Install cursor-agent or set HERMES_CURSOR_AGENT_COMMAND/CURSOR_AGENT_PATH."
            ) from exc

        if proc.stdout is None or proc.stderr is None:
            proc.kill()
            raise RuntimeError("cursor-agent did not expose stdout/stderr pipes.")

        self._spawn_stdin_writer(proc, prompt_text)

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        stdout_lines: list[str] = []

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        err_thread.start()

        # Wall-clock watchdog: `for line in proc.stdout` blocks until EOF, so
        # without this a hung cursor-agent blocked the turn forever — the
        # proc.wait(timeout=...) below is only reached after the read loop
        # ends. Kill only THIS process; never self.close(), which would wipe
        # every cached SDK agent on the client.
        done_event = threading.Event()
        timed_out = threading.Event()

        def _watchdog() -> None:
            if done_event.wait(timeout_seconds):
                return
            timed_out.set()
            if proc.poll() is None:
                _kill_process_group(proc)

        watch_thread = threading.Thread(target=_watchdog, daemon=True)
        watch_thread.start()

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                stdout_lines.append(line.rstrip("\n"))
            if timed_out.is_set():
                raise TimeoutError(
                    f"cursor-agent hung; killed after {timeout_seconds:.0f}s "
                    f"(model={model})."
                )
            proc.wait(timeout=10)
        finally:
            done_event.set()
            err_thread.join(timeout=1)
            with self._active_process_lock:
                if self._active_process is proc:
                    self._active_process = None
            self.is_closed = True

        response_text, stream_error, reasoning_text, real_usage = parse_cursor_stream_events(stdout_lines)
        stderr_text = "\n".join(stderr_tail).strip()

        # Transparent recovery: if the requested model is out of usage it emits
        # no response events (only an "out of usage. Switch to Auto" status).
        # Retry once with `auto` instead of surfacing an error to the user.
        if (
            not response_text
            and model != _AUTO_MODEL
            and (_is_usage_limit(stderr_text) or _is_usage_limit(stream_error))
        ):
            return self._run_prompt(
                prompt_text, model=_AUTO_MODEL, timeout_seconds=timeout_seconds
            )

        if stream_error and not response_text:
            raise RuntimeError(f"cursor-agent failed: {stream_error}")

        if response_text:
            return response_text, reasoning_text, real_usage

        if proc.returncode not in (0, None):
            detail = stderr_text or stream_error or f"exit code {proc.returncode}"
            raise RuntimeError(f"cursor-agent exited without a response: {detail}")

        if stderr_text:
            raise RuntimeError(f"cursor-agent produced no response: {stderr_text}")

        # Exit 0 with no stdout AND no stderr: the CLI swallowed the failure
        # (seen with models pending a data-retention-policy acknowledgement).
        # Never return "" silently — Hermes would surface a bare "Empty
        # response" with no clue about the cause.
        raise RuntimeError(
            f"cursor-agent returned an empty response (exit code 0, "
            f"no output, model={model}). The model may be unavailable "
            "on this account or require acknowledgement in Cursor "
            "(e.g. a data-retention policy) — try /model to pick "
            "another one."
        )
