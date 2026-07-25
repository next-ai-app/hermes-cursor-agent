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
        '{"type":"result","subtype":"success","result":"final answer","usage":{"inputTokens":50,"outputTokens":20}}',
    ]
    text, err, reasoning, usage = mod.parse_cursor_stream_events(lines)
    assert text == "final answer"
    assert err is None
    assert reasoning == ""
    assert usage is not None
    assert usage.prompt_tokens == 50
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 70


def test_parse_cursor_stream_events_concatenates_streaming_deltas():
    lines = [
        '{"type":"assistant","message":{"content":[{"type":"text","text":"hel"}]},"timestamp_ms":1}',
        '{"type":"assistant","message":{"content":[{"type":"text","text":"lo"}]},"timestamp_ms":2}',
    ]
    text, err, reasoning, usage = mod.parse_cursor_stream_events(lines)
    assert text == "hello"
    assert err is None
    assert reasoning == ""
    assert usage is None


def test_parse_cursor_stream_events_extracts_reasoning():
    lines = [
        '{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"step1"},{"type":"text","text":"ans"}]},"timestamp_ms":1}',
        '{"type":"result","subtype":"success","result":"final","usage":{"inputTokens":100,"outputTokens":50,"cacheReadTokens":10}}',
    ]
    text, err, reasoning, usage = mod.parse_cursor_stream_events(lines)
    assert text == "final"
    assert reasoning == "step1"
    assert usage.prompt_tokens == 100
    assert usage.prompt_tokens_details.cached_tokens == 10


def test_extract_assistant_content_thinking_blocks():
    event = {"message": {"content": [
        {"type": "thinking", "thinking": "Let me think..."},
        {"type": "text", "text": "The answer."},
    ]}}
    text, reasoning = mod._extract_assistant_content(event)
    assert text == "The answer."
    assert reasoning == "Let me think..."


def test_make_stream_chunk_forwards_reasoning():
    chunk = mod._make_stream_chunk("hello", model="test", reasoning="thinking...")
    assert chunk.choices[0].delta.content == "hello"
    assert chunk.choices[0].delta.reasoning == "thinking..."
    assert chunk.choices[0].delta.reasoning_content == "thinking..."


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
    assert '<meta model="auto" />' in prompt
    assert "<system>\nBe concise.\n</system>" in prompt
    assert "<user>\nHello\n</user>" in prompt
    assert "Respond to the latest user message above." in prompt
    assert "<hermes_instructions>" in prompt


# ---------------------------------------------------------------------------
# Multi-turn hybrid sync (incremental vs full replay)
# ---------------------------------------------------------------------------

import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402


def _m(role, text):
    return {"role": role, "content": text}


class _FakeRun:
    def __init__(self, result=None, stream_delay=0.0):
        self._result = (
            result if result is not None else SimpleNamespace(result="fake-reply")
        )
        self._delay = stream_delay

    def stream(self):
        if self._delay:
            time.sleep(self._delay)
        return iter(())

    def wait(self):
        return self._result


def _install_fake_sdk(monkeypatch, run_factory=None):
    """Patch the module-level SDK symbols with a recording fake."""
    instances = []
    make_run = run_factory or _FakeRun

    class FakeAgent:
        def __init__(self):
            self.sent = []
            self.closed = False

        def send(self, prompt):
            self.sent.append(prompt)
            return make_run()

        def close(self):
            self.closed = True

    class FakeSDKAgent:
        @staticmethod
        def create(**kwargs):
            agent = FakeAgent()
            agent.create_kwargs = kwargs
            instances.append(agent)
            return agent

    class FakeLocalOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # raising=False: names are undefined when cursor_sdk is not installed.
    monkeypatch.setattr(mod, "_SDKAgent", FakeSDKAgent, raising=False)
    monkeypatch.setattr(mod, "_SDKLocalOptions", FakeLocalOptions, raising=False)
    monkeypatch.setattr(mod, "_SDK_AVAILABLE", True)
    return instances


def _make_client():
    return mod.CursorAgentClient(api_key="test-key", cursor_cwd="/tmp")


def test_sdk_first_turn_sends_full_transcript(monkeypatch):
    instances = _install_fake_sdk(monkeypatch)
    client = _make_client()
    msgs = [_m("system", "Be concise."), _m("user", "Hello")]
    client._sdk_chat_completion(model="auto", messages=msgs)

    assert len(instances) == 1
    prompt = instances[0].sent[0]
    assert "<hermes_instructions>" in prompt
    assert "<system>\nBe concise.\n</system>" in prompt
    assert "<user>\nHello\n</user>" in prompt
    conv_key = client._conv_key(msgs, "auto")
    assert client._sdk_msg_counts[conv_key] == 2


def test_sdk_insync_followup_is_incremental(monkeypatch):
    instances = _install_fake_sdk(monkeypatch)
    client = _make_client()
    turn1 = [_m("system", "Be concise."), _m("user", "Hello")]
    client._sdk_chat_completion(model="auto", messages=turn1)

    turn2 = turn1 + [_m("assistant", "fake-reply"), _m("user", "And now?")]
    client._sdk_chat_completion(model="auto", messages=turn2)

    assert len(instances) == 1  # same agent reused
    assert instances[0].sent[1] == "And now?"  # only the new user message
    assert not instances[0].closed
    conv_key = client._conv_key(turn1, "auto")
    assert client._sdk_msg_counts[conv_key] == 4


def test_sdk_tool_results_trigger_full_replay(monkeypatch):
    instances = _install_fake_sdk(monkeypatch)
    client = _make_client()
    turn1 = [_m("system", "Be concise."), _m("user", "Hello")]
    client._sdk_chat_completion(model="auto", messages=turn1)

    # Hermes appended assistant + tool result + new user message (count skew).
    turn2 = turn1 + [
        _m("assistant", "fake-reply"),
        _m("tool", "file contents here"),
        _m("user", "What does it say?"),
    ]
    client._sdk_chat_completion(model="auto", messages=turn2)

    assert len(instances) == 2  # stale agent replaced
    assert instances[0].closed
    prompt = instances[1].sent[0]
    assert "<tool_result>\nfile contents here\n</tool_result>" in prompt
    assert "<user>\nHello\n</user>" in prompt  # earlier history preserved
    assert "<user>\nWhat does it say?\n</user>" in prompt


def test_sdk_undo_triggers_full_replay(monkeypatch):
    instances = _install_fake_sdk(monkeypatch)
    client = _make_client()
    turn1 = [_m("system", "Be concise."), _m("user", "Hello")]
    client._sdk_chat_completion(model="auto", messages=turn1)

    # /undo: history shrank back to the pre-reply state.
    client._sdk_chat_completion(model="auto", messages=list(turn1))

    assert len(instances) == 2
    assert instances[0].closed
    assert "<user>\nHello\n</user>" in instances[1].sent[0]


def test_sdk_fresh_client_replays_history(monkeypatch):
    """Process restart wipes the agent cache; history must still arrive."""
    instances = _install_fake_sdk(monkeypatch)
    client1 = _make_client()
    turn1 = [_m("system", "Be concise."), _m("user", "Hello")]
    client1._sdk_chat_completion(model="auto", messages=turn1)

    client2 = _make_client()  # e.g. after a gateway restart
    turn2 = turn1 + [_m("assistant", "fake-reply"), _m("user", "And now?")]
    client2._sdk_chat_completion(model="auto", messages=turn2)

    assert len(instances) == 2
    prompt = instances[1].sent[0]
    assert "<user>\nHello\n</user>" in prompt
    assert "<assistant>\nfake-reply\n</assistant>" in prompt
    assert "<user>\nAnd now?\n</user>" in prompt


def test_sdk_streaming_followup_is_incremental(monkeypatch):
    instances = _install_fake_sdk(monkeypatch)
    client = _make_client()
    turn1 = [_m("system", "Be concise."), _m("user", "Hello")]
    list(client._sdk_chat_completion_stream(model="auto", messages=turn1))

    turn2 = turn1 + [_m("assistant", "fake-reply"), _m("user", "And now?")]
    chunks = list(client._sdk_chat_completion_stream(model="auto", messages=turn2))

    assert len(instances) == 1
    assert instances[0].sent[1] == "And now?"
    assert chunks[-1].choices[0].finish_reason == "stop"


# ---------------------------------------------------------------------------
# Robustness fixes: fingerprint, model switch, tool calls, fallback, timeout
# ---------------------------------------------------------------------------


def test_conv_key_distinguishes_long_identical_prefixes():
    """Same 300-char opening (typical Hermes system prompt) must not collide."""
    client = _make_client()
    shared_prefix = "X" * 400
    a = [_m("system", shared_prefix + "A"), _m("user", "hello")]
    b = [_m("system", shared_prefix + "B"), _m("user", "hello")]
    assert client._conv_key(a, "auto") != client._conv_key(b, "auto")


def test_sdk_model_switch_gets_fresh_agent_with_replay(monkeypatch):
    instances = _install_fake_sdk(monkeypatch)
    client = _make_client()
    turn1 = [_m("system", "Be concise."), _m("user", "Hello")]
    client._sdk_chat_completion(model="model-a", messages=turn1)

    turn2 = turn1 + [_m("assistant", "fake-reply"), _m("user", "And now?")]
    client._sdk_chat_completion(model="model-b", messages=turn2)

    assert len(instances) == 2  # old-model agent is not silently reused
    prompt = instances[1].sent[0]
    assert "<user>\nHello\n</user>" in prompt  # new agent got full history


def test_format_messages_renders_tool_calls():
    prompt = mod._format_messages_as_prompt(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command": "ls"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt"},
        ]
    )
    assert '<tool_call id="call_1" name="terminal">' in prompt
    assert '{"command": "ls"}' in prompt
    assert '<tool_result call_id="call_1">\nfile1.txt\n</tool_result>' in prompt


def test_sdk_stream_error_falls_back_to_cli(monkeypatch):
    """Errors inside the lazily-executed stream must still reach the CLI path."""

    def _boom():
        raise RuntimeError("bridge crashed")

    _install_fake_sdk(monkeypatch, run_factory=_boom)
    client = _make_client()
    marker = mod._make_stream_chunk("cli-fallback", model="auto")
    monkeypatch.setattr(
        client, "_create_chat_completion_stream", lambda **kw: iter([marker])
    )

    chunks = list(
        client.chat.completions.create(
            stream=True, model="auto", messages=[_m("user", "hi")]
        )
    )

    assert chunks == [marker]
    assert not client._sdk_active()  # cooldown engaged, not permanent


def test_sdk_failure_cooldown_expires(monkeypatch):
    _install_fake_sdk(monkeypatch)
    client = _make_client()
    client._mark_sdk_failure("test", RuntimeError("transient"))
    assert not client._sdk_active()
    client._sdk_disabled_until = 0.0  # simulate cooldown elapsing
    assert client._sdk_active()


def test_sdk_run_error_raises_and_evicts_agent(monkeypatch):
    def _failed_run():
        return _FakeRun(result=SimpleNamespace(result=None, error="out of usage"))

    instances = _install_fake_sdk(monkeypatch, run_factory=_failed_run)
    client = _make_client()
    with pytest.raises(RuntimeError, match="out of usage"):
        client._sdk_chat_completion(
            model="auto", messages=[_m("user", "hi")]
        )
    assert instances[0].closed
    assert not client._sdk_agents  # evicted: next turn replays cleanly


def test_sdk_usage_extracted_from_result(monkeypatch):
    def _run_with_usage():
        return _FakeRun(
            result=SimpleNamespace(
                result="hi there",
                usage={"inputTokens": 10, "outputTokens": 5, "cacheReadTokens": 2},
            )
        )

    _install_fake_sdk(monkeypatch, run_factory=_run_with_usage)
    client = _make_client()
    response = client._sdk_chat_completion(model="auto", messages=[_m("user", "hi")])
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15
    assert response.usage.prompt_tokens_details.cached_tokens == 2


def test_sdk_timeout_watchdog_aborts_hung_run(monkeypatch):
    def _slow_run():
        return _FakeRun(stream_delay=0.6)

    instances = _install_fake_sdk(monkeypatch, run_factory=_slow_run)
    client = _make_client()
    with pytest.raises(TimeoutError):
        client._sdk_chat_completion(
            model="auto", messages=[_m("user", "hi")], timeout=0.1
        )
    assert instances[0].closed  # watchdog force-closed the agent
    assert not client._sdk_agents


# ---------------------------------------------------------------------------
# Model catalog: SDK-first with CLI fallback and short cache
# ---------------------------------------------------------------------------

import types  # noqa: E402


def _clear_models_cache():
    mod._models_cache["at"] = 0.0
    mod._models_cache["ids"] = None


def _install_fake_cursor_catalog(monkeypatch, ids, calls=None):
    fake_sdk = types.ModuleType("cursor_sdk")

    class _Models:
        @staticmethod
        def list(api_key=None):
            if calls is not None:
                calls.append(api_key)
            return [SimpleNamespace(id=i) for i in ids]

    class Cursor:
        models = _Models()

    fake_sdk.Cursor = Cursor
    monkeypatch.setitem(sys.modules, "cursor_sdk", fake_sdk)
    monkeypatch.setattr(mod, "_SDK_AVAILABLE", True)


def test_fetch_models_prefers_sdk_catalog(monkeypatch):
    _clear_models_cache()
    _install_fake_cursor_catalog(
        monkeypatch, ["default", "claude-fable-5", "composer-2.5"]
    )
    # CLI must not be needed at all.
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)

    out = mod.fetch_cursor_models(force_refresh=True)
    assert out is not None
    assert out[0] == "auto"  # SDK's "default" is exposed as the usable id
    assert "claude-fable-5" in out
    _clear_models_cache()


def test_fetch_models_caches_sdk_result(monkeypatch):
    _clear_models_cache()
    calls = []
    _install_fake_cursor_catalog(monkeypatch, ["default", "composer-2.5"], calls)

    first = mod.fetch_cursor_models(force_refresh=True)
    second = mod.fetch_cursor_models()
    assert first == second
    assert len(calls) == 1  # second call served from cache
    _clear_models_cache()


def test_fetch_models_falls_back_to_cli_when_sdk_fails(monkeypatch):
    _clear_models_cache()
    fake_sdk = types.ModuleType("cursor_sdk")

    class _Models:
        @staticmethod
        def list(api_key=None):
            raise RuntimeError("sdk down")

    class Cursor:
        models = _Models()

    fake_sdk.Cursor = Cursor
    monkeypatch.setitem(sys.modules, "cursor_sdk", fake_sdk)
    monkeypatch.setattr(mod, "_SDK_AVAILABLE", True)

    cli_called = []

    def _fake_cli(**kwargs):
        cli_called.append(kwargs)
        return ["auto", "composer-2.5"]

    monkeypatch.setattr(mod, "_fetch_cursor_models_cli", _fake_cli)
    out = mod.fetch_cursor_models(force_refresh=True)
    assert out == ["auto", "composer-2.5"]
    assert cli_called
    _clear_models_cache()


def test_sdk_turn_stats_track_incremental_vs_replay(monkeypatch):
    instances = _install_fake_sdk(monkeypatch)
    client = _make_client()
    turn1 = [_m("system", "Be concise."), _m("user", "Hello")]
    client._sdk_chat_completion(model="auto", messages=turn1)  # replay (new agent)

    turn2 = turn1 + [_m("assistant", "fake-reply"), _m("user", "And now?")]
    client._sdk_chat_completion(model="auto", messages=turn2)  # incremental

    turn3 = turn2 + [
        _m("assistant", "fake-reply"),
        _m("tool", "tool output"),
        _m("user", "next"),
    ]
    client._sdk_chat_completion(model="auto", messages=turn3)  # replay (desync)

    assert client._sdk_turn_stats == {"incremental": 1, "replay": 2}
    assert len(instances) == 2


# ---------------------------------------------------------------------------
# API key pool (CURSOR_API_KEYS rotation on usage limits)
# ---------------------------------------------------------------------------


def _fresh_pool(monkeypatch, *, single=None, pool=None):
    """Reset the module key-pool singleton and configure the key env vars."""
    if single is None:
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    else:
        monkeypatch.setenv("CURSOR_API_KEY", single)
    if pool is None:
        monkeypatch.delenv("CURSOR_API_KEYS", raising=False)
    else:
        monkeypatch.setenv("CURSOR_API_KEYS", pool)
    fresh = mod._ApiKeyPool()
    monkeypatch.setattr(mod, "_KEY_POOL", fresh)
    return fresh


def test_key_pool_merges_env_sources(monkeypatch):
    _fresh_pool(monkeypatch, single="key-a", pool="key-b, key-a;key-c\nkey-d")
    assert mod._ApiKeyPool.configured_keys() == ["key-a", "key-b", "key-c", "key-d"]


def test_key_pool_single_key_setup_unchanged(monkeypatch):
    pool = _fresh_pool(monkeypatch, single="key-a")
    assert pool.active() == "key-a"
    # Exhausted with no alternative: same key stays active (soonest recovery),
    # exactly the old single-key behaviour.
    assert pool.mark_limited("key-a") is None
    assert pool.active() == "key-a"


def test_key_pool_rotates_on_usage_limit(monkeypatch):
    pool = _fresh_pool(monkeypatch, pool="key-a,key-b,key-c")
    assert pool.active() == "key-a"
    assert pool.mark_limited("key-a") == "key-b"
    assert pool.active() == "key-b"
    assert pool.mark_limited("key-b") == "key-c"
    assert pool.mark_limited("key-c") is None  # everything cooling down
    assert pool.active() in {"key-a", "key-b", "key-c"}  # soonest recovery


def test_resolve_api_key_explicit_and_pool(monkeypatch):
    pool = _fresh_pool(monkeypatch, pool="key-a,key-b")
    # A key outside the pool (constructor/test override) wins as-is.
    assert mod._resolve_api_key("other-key") == "other-key"
    # A pool member defers to the pool, so rotation applies to it too.
    pool.mark_limited("key-a")
    assert mod._resolve_api_key("key-a") == "key-b"


def test_usage_limit_retry_prefers_next_key_then_auto(monkeypatch):
    _fresh_pool(monkeypatch, pool="key-a,key-b")
    # Another healthy key exists: retry the same model on it.
    assert mod._usage_limit_retry_model("composer-2.5", "key-a") == "composer-2.5"
    # Last key exhausted: downgrade to auto.
    assert mod._usage_limit_retry_model("composer-2.5", "key-b") == "auto"
    # Already on auto with nothing left: give up.
    assert mod._usage_limit_retry_model("auto", "key-b") is None


def test_sdk_usage_limit_rotates_pool_key(monkeypatch):
    def _failed_run():
        return _FakeRun(result=SimpleNamespace(result=None, error="out of usage"))

    instances = _install_fake_sdk(monkeypatch, run_factory=_failed_run)
    pool = _fresh_pool(monkeypatch, pool="key-a,key-b")
    client = mod.CursorAgentClient(cursor_cwd="/tmp")

    with pytest.raises(RuntimeError, match="out of usage"):
        client._sdk_chat_completion(model="auto", messages=[_m("user", "hi")])

    assert instances[0].create_kwargs["api_key"] == "key-a"
    assert pool.active() == "key-b"  # exhausted key cooled down

    with pytest.raises(RuntimeError, match="out of usage"):
        client._sdk_chat_completion(model="auto", messages=[_m("user", "hi")])
    assert instances[1].create_kwargs["api_key"] == "key-b"


def test_usage_limit_does_not_disable_sdk(monkeypatch):
    _fresh_pool(monkeypatch, pool="key-a,key-b")
    client = _make_client()
    client._mark_sdk_failure("stream", RuntimeError("out of usage. Switch to Auto"))
    assert client._sdk_active()  # key rotated; SDK path stays preferred
    client._mark_sdk_failure("stream", RuntimeError("bridge crashed"))
    assert not client._sdk_active()  # real failures still trigger the cooldown
