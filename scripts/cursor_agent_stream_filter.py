#!/usr/bin/env python3
"""Convert Cursor Agent stream-json output into human-readable live logs.

The raw stream includes a full `user` event containing the entire autocoder prompt,
which is too noisy for the public/live log. This filter keeps assistant/tool/result
progress while omitting prompt payloads and flushing every line for SSE tailing.
"""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Any

SECRET_PATTERNS = [
    (re.compile(r"(Authorization\s*[:=]\s*)(Bearer\s+)?[^\s'\"<>]+", re.I), r"\1\2[REDACTED]"),
    (re.compile(r"((?:LINEAR|OPENAI|OPENROUTER|ANTHROPIC|GOOGLE|GEMINI|CLAUDE|CURSOR|GITHUB|GH|NPM|DATABASE|MYSQL|POSTGRES|REDIS|JWT|SECRET|TOKEN|KEY)[A-Z0-9_]*\s*=\s*)[^\s'\"<>]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"\b(sk-or-v1-[A-Za-z0-9_-]{16,})\b"), "[REDACTED_OPENROUTER_KEY]"),
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{20,})\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9_]{20,})\b"), "[REDACTED_GITHUB_TOKEN]"),
]


def redact(text: str) -> str:
    for pattern, repl in SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def ts() -> str:
    return time.strftime("%F %T %z")


def content_text(message: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in message.get("content") or []:
        if not isinstance(item, dict):
            continue
        typ = item.get("type")
        if typ == "text" and item.get("text"):
            chunks.append(str(item["text"]))
        elif typ in {"tool_use", "tool_result"}:
            name = item.get("name") or item.get("tool_name") or item.get("id") or typ
            chunks.append(f"[{typ}: {name}]")
    return "".join(chunks).strip()


def emit(text: str = "") -> None:
    if text:
        print(redact(text), flush=True)
    else:
        print(flush=True)


emit(f"[cursor-stream] {ts()} stream-json filter started; user prompt payloads are omitted from live log")
for raw in sys.stdin:
    raw = raw.rstrip("\n")
    if not raw:
        continue
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        emit(f"[cursor-stderr] {raw}")
        continue

    typ = event.get("type")
    subtype = event.get("subtype")
    if typ == "user":
        emit(f"[cursor-user] {ts()} prompt delivered to agent (content omitted from log)")
        continue
    if typ == "system":
        model = event.get("model") or "unknown"
        sid = event.get("session_id") or "unknown"
        emit(f"[cursor-system] {ts()} init model={model} session={sid}")
        continue
    if typ == "assistant":
        text = content_text(event.get("message") or {})
        if text:
            emit(text)
        else:
            emit(f"[cursor-assistant] {ts()} assistant event")
        continue
    if typ == "result":
        usage = event.get("usage") or {}
        usage_bits = []
        for key in ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens"):
            if key in usage:
                usage_bits.append(f"{key}={usage[key]}")
        result = str(event.get("result") or "").strip()
        emit(f"[cursor-result] {ts()} subtype={subtype or 'unknown'} is_error={event.get('is_error')} duration_ms={event.get('duration_ms')} {' '.join(usage_bits)}")
        if result:
            emit(result)
        continue

    # Preserve unknown events compactly so future Cursor formats still show progress.
    compact = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    if len(compact) > 4000:
        compact = compact[:4000] + f"...[truncated {len(compact) - 4000} chars]"
    emit(f"[cursor-{typ or 'event'}] {compact}")

emit(f"[cursor-stream] {ts()} stream-json filter ended")
