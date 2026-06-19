"""Unit tests for cursor-agent stream parsing (no Hermes runtime required)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "model-providers" / "cursor-agent" / "cursor_agent_client.py"


def _load_client_module():
    spec = importlib.util.spec_from_file_location("cursor_agent_client_test", CLIENT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_client_module()


def test_parse_cursor_stream_line_ignores_garbage():
    assert mod.parse_cursor_stream_line("") is None
    assert mod.parse_cursor_stream_line("not json") is None


def test_parse_cursor_stream_events_prefers_result():
    lines = [
        '{"type":"assistant","message":{"content":[{"type":"text","text":"partial"}]},"timestamp_ms":1}',
        '{"type":"result","subtype":"success","result":"final answer"}',
    ]
    text, err = mod.parse_cursor_stream_events(lines)
    assert text == "final answer"
    assert err is None


def test_parse_cursor_stream_events_concatenates_streaming_deltas():
    lines = [
        '{"type":"assistant","message":{"content":[{"type":"text","text":"hel"}]},"timestamp_ms":1}',
        '{"type":"assistant","message":{"content":[{"type":"text","text":"lo"}]},"timestamp_ms":2}',
    ]
    text, err = mod.parse_cursor_stream_events(lines)
    assert text == "hello"
    assert err is None


def test_parse_cursor_list_models_output():
    stdout = """Available models:
composer-2.5-fast - fast composer
auto - automatic routing
gpt-5.3-codex - codex
"""
    models = mod.parse_cursor_list_models_output(stdout)
    assert models == ["composer-2.5-fast", "auto", "gpt-5.3-codex"]


def test_format_messages_as_prompt_includes_roles():
    prompt = mod._format_messages_as_prompt(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
        model="auto",
    )
    assert "Hermes requested model: auto" in prompt
    assert "System:\nBe concise." in prompt
    assert "User:\nHello" in prompt
    assert "Continue from the latest user request." in prompt
