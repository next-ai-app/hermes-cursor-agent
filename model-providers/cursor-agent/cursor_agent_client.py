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
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CURSOR_MARKER_BASE_URL = "acp://cursor"
_DEFAULT_TIMEOUT_SECONDS = 900.0

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


def _format_messages_as_prompt(messages: list[dict[str, Any]], *, model: str | None = None) -> str:
    sections: list[str] = [
        "You are being used as the active agent backend for Hermes.",
        "Complete the user's request using your own tools when needed.",
        "When you are done, reply with a clear final answer for the user.",
    ]
    if model:
        sections.append(f"Hermes requested model: {model}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        rendered = _render_message_content(message.get("content"))
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _extract_assistant_text(event: dict[str, Any]) -> str:
    message = event.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts)


def _make_stream_chunk(
    content: str | None,
    *,
    model: str,
    finish_reason: str | None = None,
    usage: Any | None = None,
) -> Any:
    """Build an OpenAI ChatCompletionChunk-compatible object.

    The Hermes streaming consumer reads ``chunk.choices[0].delta.content``,
    ``chunk.choices[0].delta.tool_calls`` and ``chunk.choices[0].finish_reason``;
    ``chunk.usage`` / ``chunk.model`` are accessed defensively via ``hasattr``.
    A SimpleNamespace mirroring that shape is all that's required.
    """
    delta = SimpleNamespace(
        role=None,
        content=content,
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def parse_cursor_stream_line(line: str) -> dict[str, Any] | None:
    stripped = (line or "").strip()
    if not stripped or not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_cursor_stream_events(lines: list[str]) -> tuple[str, str | None]:
    """Parse NDJSON lines from cursor-agent stream-json output.

    Returns (response_text, error_message). Prefers the canonical ``result`` event;
    falls back to concatenated streaming assistant deltas (``timestamp_ms`` present,
    ``model_call_id`` absent).
    """
    streaming_parts: list[str] = []
    final_result = ""
    error_message: str | None = None

    for line in lines:
        event = parse_cursor_stream_line(line)
        if not event:
            continue

        event_type = str(event.get("type") or "").strip()
        if event_type == "assistant":
            has_ts = "timestamp_ms" in event
            has_model_call = "model_call_id" in event
            if has_ts and not has_model_call:
                chunk = _extract_assistant_text(event)
                if chunk:
                    streaming_parts.append(chunk)
            elif not has_ts and not has_model_call:
                chunk = _extract_assistant_text(event)
                if chunk and not streaming_parts:
                    final_result = chunk
        elif event_type == "result":
            subtype = str(event.get("subtype") or "").strip().lower()
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

    if final_result:
        return final_result, error_message

    streamed = "".join(streaming_parts).strip()
    return streamed, error_message


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


def fetch_cursor_models(
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


class _CursorChatCompletions:
    def __init__(self, client: "CursorAgentClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        # When Hermes requests ``stream=True`` it iterates the return value as
        # an OpenAI-style chunk stream (``for chunk in stream`` reading
        # ``chunk.choices[0].delta.content``).  cursor-agent already emits
        # incremental ``assistant`` deltas over stream-json, so forward them
        # live instead of buffering the whole reply — this is what makes text
        # appear progressively in Open WebUI and cuts perceived latency.
        if kwargs.get("stream"):
            return self._client._create_chat_completion_stream(**kwargs)
        return self._client._create_chat_completion(**kwargs)


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

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

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
            effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            effective_timeout = float(timeout)
        else:
            candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
            effective_timeout = max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS

        response_text = self._run_prompt(
            prompt_text,
            model=model or _resolve_default_model(),
            timeout_seconds=effective_timeout,
        )

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=response_text,
            tool_calls=[],
            reasoning=None,
            reasoning_content=None,
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
            return _DEFAULT_TIMEOUT_SECONDS
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
        command = self._build_command(
            prompt_text=prompt_text, model=model, stream_partial=True
        )
        stderr_tail: deque[str] = deque(maxlen=40)

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._cursor_cwd,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Cursor Agent command '{self._cursor_command}'. "
                "Install cursor-agent or set HERMES_CURSOR_AGENT_COMMAND/CURSOR_AGENT_PATH."
            ) from exc

        if proc.stdout is None or proc.stderr is None:
            proc.kill()
            raise RuntimeError("cursor-agent did not expose stdout/stderr pipes.")

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
                try:
                    proc.kill()
                except Exception:
                    pass

        watch_thread = threading.Thread(target=_watchdog, daemon=True)
        watch_thread.start()

        streamed_any = False
        final_result = ""
        error_message: str | None = None

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
                        chunk_text = _extract_assistant_text(event)
                        if chunk_text:
                            streamed_any = True
                            yield _make_stream_chunk(chunk_text, model=model)
                    elif not has_ts and not has_model_call:
                        chunk_text = _extract_assistant_text(event)
                        if chunk_text and not final_result:
                            final_result = chunk_text
                elif event_type == "result":
                    subtype = str(event.get("subtype") or "").strip().lower()
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
                try:
                    proc.kill()
                except Exception:
                    pass

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
                yield _make_stream_chunk(final_result, model=model)
            elif error_message:
                raise RuntimeError(f"cursor-agent failed: {error_message}")
            elif proc.returncode not in (0, None):
                detail = stderr_text or f"exit code {proc.returncode}"
                raise RuntimeError(f"cursor-agent exited without a response: {detail}")
            elif stderr_text:
                raise RuntimeError(f"cursor-agent produced no response: {stderr_text}")

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        yield _make_stream_chunk(None, model=model, finish_reason="stop", usage=usage)

    def _build_command(
        self, *, prompt_text: str, model: str, stream_partial: bool = False
    ) -> list[str]:
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
        cmd.append(prompt_text)
        return cmd

    def _run_prompt(self, prompt_text: str, *, model: str, timeout_seconds: float) -> str:
        command = self._build_command(prompt_text=prompt_text, model=model)
        stderr_tail: deque[str] = deque(maxlen=40)

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._cursor_cwd,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Cursor Agent command '{self._cursor_command}'. "
                "Install cursor-agent or set HERMES_CURSOR_AGENT_COMMAND/CURSOR_AGENT_PATH."
            ) from exc

        if proc.stdout is None or proc.stderr is None:
            proc.kill()
            raise RuntimeError("cursor-agent did not expose stdout/stderr pipes.")

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

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                stdout_lines.append(line.rstrip("\n"))
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            self.close()
            raise TimeoutError(
                f"Timed out waiting for cursor-agent response after {timeout_seconds:.0f}s."
            ) from exc
        finally:
            err_thread.join(timeout=1)
            with self._active_process_lock:
                if self._active_process is proc:
                    self._active_process = None
            self.is_closed = True

        response_text, stream_error = parse_cursor_stream_events(stdout_lines)
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
            return response_text

        if proc.returncode not in (0, None):
            detail = stderr_text or stream_error or f"exit code {proc.returncode}"
            raise RuntimeError(f"cursor-agent exited without a response: {detail}")

        if stderr_text:
            raise RuntimeError(f"cursor-agent produced no response: {stderr_text}")

        return ""
