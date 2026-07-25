"""Cursor Agent provider — user plugin (survives ``hermes update``).

This single plugin wires ``cursor-agent`` into Hermes WITHOUT editing any core
repo file. It does two things at import time:

  1. Registers a ``ProviderProfile`` so the provider is discoverable.
  2. Monkeypatches a small set of core functions (credential resolution,
     runtime resolution, client construction, async wrapping, model catalog,
     diagnostics) so the external-process provider behaves like the built-in
     ``copilot-acp`` shim.

Every patch is wrapped in defensive error handling: if a target function moved
or changed shape in a future hermes release, that single patch is skipped (with
a warning) and the rest keep working. Because everything lives under
``$HERMES_HOME/plugins/model-providers/cursor-agent/``, ``hermes update`` never
touches it.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("hermes.plugins.cursor_agent")

_HERE = Path(__file__).resolve().parent
CURSOR_MARKER_BASE_URL = "acp://cursor"
CURSOR_ALIASES = frozenset({"cursor-agent", "cursor", "cursor-cli"})

_FALLBACK_MODELS = (
    "auto",
    "claude-fable-5",
    "composer-2.5-fast",
    "composer-2.5",
    "gpt-5.5-medium",
    "gpt-5.3-codex",
    "claude-4.6-sonnet-medium-thinking",
    "claude-opus-4-8-thinking-high",
)


# ---------------------------------------------------------------------------
# Load the sibling shim module with a stable, unique name so both this file and
# the patches below can import the client class regardless of how the plugin
# loader named us.
# ---------------------------------------------------------------------------
def _load_sibling(mod_attr: str, filename: str):
    full_name = f"hermes_cursor_agent_plugin.{mod_attr}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, _HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


_client_mod = _load_sibling("client", "cursor_agent_client.py")
CursorAgentClient = _client_mod.CursorAgentClient
AsyncCursorAgentClient = _client_mod.AsyncCursorAgentClient
fetch_cursor_models = _client_mod.fetch_cursor_models


# ---------------------------------------------------------------------------
# 1. Provider profile registration
# ---------------------------------------------------------------------------
from providers import register_provider  # noqa: E402
from providers.base import ProviderProfile  # noqa: E402


class CursorAgentProfile(ProviderProfile):
    def fetch_models(self, *, api_key: str | None = None, timeout: float = 8.0):
        try:
            return fetch_cursor_models(api_key=api_key, timeout=timeout)
        except Exception:
            return None


_profile = CursorAgentProfile(
    name="cursor-agent",
    aliases=("cursor", "cursor-cli"),
    display_name="Cursor Agent",
    description="Cursor Agent CLI (headless cursor-agent via CURSOR_API_KEY)",
    signup_url="https://cursor.com/",
    auth_type="external_process",
    base_url=CURSOR_MARKER_BASE_URL,
    env_vars=("CURSOR_API_KEY", "CURSOR_AGENT_BASE_URL"),
    supports_health_check=False,
    fallback_models=_FALLBACK_MODELS,
    default_aux_model="composer-2.5-fast",
)

register_provider(_profile)


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------
_applied: list[str] = []
_skipped: list[str] = []


def _safe_patch(label: str):
    """Decorator: wrap a patch closure so calling it records success/failure.

    The wrapped function is NOT executed at decoration time — it is executed
    once, explicitly, from the apply block at the bottom of this module. This
    keeps the apply order obvious and avoids double-wrapping core functions.
    """

    def _runner(fn):
        def _safe():
            try:
                fn()
                _applied.append(label)
            except Exception as exc:  # pragma: no cover - defensive
                _skipped.append(f"{label}: {exc}")
                logger.warning(
                    "cursor-agent plugin: skipped patch %r (%s)", label, exc
                )

        _safe.__name__ = getattr(fn, "__name__", "patch")
        return _safe

    return _runner


def _is_cursor_provider(value) -> bool:
    return str(value or "").strip().lower() in CURSOR_ALIASES


def _is_cursor_base_url(value) -> bool:
    return str(value or "").strip().lower().startswith("acp://cursor")


def _config_provider_is_cursor() -> bool:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        prov = str((cfg.get("model") or {}).get("provider") or "").strip().lower()
        if prov in CURSOR_ALIASES:
            return True
    except Exception:
        pass
    try:
        from hermes_cli.auth import get_active_provider

        if str(get_active_provider() or "").strip().lower() in CURSOR_ALIASES:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# 2. auth.PROVIDER_REGISTRY injection
# ---------------------------------------------------------------------------
@_safe_patch("auth.PROVIDER_REGISTRY")
def _patch_registry():
    from hermes_cli import auth

    if "cursor-agent" in auth.PROVIDER_REGISTRY:
        return
    cfg = auth.ProviderConfig(
        id="cursor-agent",
        name="Cursor Agent",
        auth_type="external_process",
        inference_base_url=CURSOR_MARKER_BASE_URL,
        base_url_env_var="CURSOR_AGENT_BASE_URL",
        api_key_env_vars=("CURSOR_API_KEY",),
    )
    auth.PROVIDER_REGISTRY["cursor-agent"] = cfg
    auth.PROVIDER_REGISTRY.setdefault("cursor", cfg)
    auth.PROVIDER_REGISTRY.setdefault("cursor-cli", cfg)


# ---------------------------------------------------------------------------
# 3. auth.resolve_external_process_provider_credentials
# ---------------------------------------------------------------------------
@_safe_patch("auth.resolve_external_process_provider_credentials")
def _patch_creds():
    from hermes_cli import auth

    _orig = auth.resolve_external_process_provider_credentials

    def _wrapped(provider_id: str):
        if str(provider_id).strip().lower() in CURSOR_ALIASES:
            return _resolve_cursor_credentials(auth, provider_id)
        return _orig(provider_id)

    auth.resolve_external_process_provider_credentials = _wrapped


def _resolve_cursor_credentials(auth, provider_id: str):
    import os
    import shlex
    import shutil

    pconfig = auth.PROVIDER_REGISTRY.get("cursor-agent")
    base_url = ""
    if pconfig and pconfig.base_url_env_var:
        base_url = os.getenv(pconfig.base_url_env_var, "").strip()
    if not base_url:
        base_url = pconfig.inference_base_url if pconfig else CURSOR_MARKER_BASE_URL

    command = (
        os.getenv("HERMES_CURSOR_AGENT_COMMAND", "").strip()
        or os.getenv("CURSOR_AGENT_PATH", "").strip()
        or "cursor-agent"
    )
    raw_args = os.getenv("HERMES_CURSOR_AGENT_ARGS", "").strip()
    args = shlex.split(raw_args) if raw_args else []
    resolved_command = shutil.which(command) if command else None
    api_key = os.getenv("CURSOR_API_KEY", "").strip()

    if not resolved_command:
        raise auth.AuthError(
            f"Could not find the Cursor Agent command '{command}'. "
            "Install cursor-agent or set HERMES_CURSOR_AGENT_COMMAND/CURSOR_AGENT_PATH.",
            provider="cursor-agent",
            code="missing_cursor_cli",
        )
    if not api_key:
        raise auth.AuthError(
            "Cursor Agent requires CURSOR_API_KEY. "
            "Set it in ~/.hermes/.env or export CURSOR_API_KEY.",
            provider="cursor-agent",
            code="missing_cursor_api_key",
        )
    return {
        "provider": "cursor-agent",
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "command": resolved_command or command,
        "args": args,
        "source": "process",
    }


# ---------------------------------------------------------------------------
# 4. auth.get_external_process_provider_status + get_auth_status (doctor / status)
# ---------------------------------------------------------------------------
@_safe_patch("auth.get_external_process_provider_status")
def _patch_status():
    from hermes_cli import auth

    _orig = auth.get_external_process_provider_status

    def _wrapped(provider_id: str):
        if str(provider_id).strip().lower() in CURSOR_ALIASES:
            return _cursor_status(auth)
        return _orig(provider_id)

    auth.get_external_process_provider_status = _wrapped


def _cursor_status(auth):
    import os
    import shutil

    pconfig = auth.PROVIDER_REGISTRY.get("cursor-agent")
    if not pconfig:
        return {"configured": False}
    command = (
        os.getenv("HERMES_CURSOR_AGENT_COMMAND", "").strip()
        or os.getenv("CURSOR_AGENT_PATH", "").strip()
        or "cursor-agent"
    )
    api_key = os.getenv("CURSOR_API_KEY", "").strip()
    base_url = ""
    if pconfig.base_url_env_var:
        base_url = os.getenv(pconfig.base_url_env_var, "").strip()
    if not base_url:
        base_url = pconfig.inference_base_url
    resolved_command = shutil.which(command) if command else None
    return {
        "configured": bool(resolved_command and api_key),
        "provider": "cursor-agent",
        "name": pconfig.name,
        "command": command,
        "args": [],
        "resolved_command": resolved_command,
        "base_url": base_url,
        "logged_in": bool(resolved_command and api_key),
        "api_key_configured": bool(api_key),
    }


@_safe_patch("auth.get_auth_status")
def _patch_auth_status():
    from hermes_cli import auth

    _orig = auth.get_auth_status

    def _wrapped(provider_id=None):
        target = str(provider_id or "").strip().lower()
        if target in CURSOR_ALIASES:
            return auth.get_external_process_provider_status("cursor-agent")
        return _orig(provider_id)

    auth.get_auth_status = _wrapped


# ---------------------------------------------------------------------------
# 5. runtime_provider.resolve_runtime_provider
# ---------------------------------------------------------------------------
@_safe_patch("runtime_provider.resolve_runtime_provider")
def _patch_runtime():
    from hermes_cli import runtime_provider

    _orig = runtime_provider.resolve_runtime_provider

    def _wrapped(
        *,
        requested=None,
        explicit_api_key=None,
        explicit_base_url=None,
        target_model=None,
    ):
        is_cursor = (
            _is_cursor_provider(requested)
            or _is_cursor_base_url(explicit_base_url)
            or (
                str(requested or "").strip().lower() in ("", "auto", "none", "default")
                and _config_provider_is_cursor()
            )
        )
        if is_cursor:
            from hermes_cli.auth import resolve_external_process_provider_credentials

            creds = resolve_external_process_provider_credentials("cursor-agent")
            return {
                "provider": "cursor-agent",
                "api_mode": "chat_completions",
                "base_url": creds.get("base_url", CURSOR_MARKER_BASE_URL).rstrip("/"),
                "api_key": creds.get("api_key", ""),
                "command": creds.get("command", ""),
                "args": list(creds.get("args") or []),
                "source": creds.get("source", "process"),
                "requested_provider": "cursor-agent",
            }
        return _orig(
            requested=requested,
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
            target_model=target_model,
        )

    runtime_provider.resolve_runtime_provider = _wrapped


# ---------------------------------------------------------------------------
# 6. agent_runtime_helpers.create_openai_client (build CursorAgentClient)
# ---------------------------------------------------------------------------
@_safe_patch("agent_runtime_helpers.create_openai_client")
def _patch_create_client():
    from agent import agent_runtime_helpers

    _orig = agent_runtime_helpers.create_openai_client

    def _wrapped(agent, client_kwargs, *, reason, shared):
        kwargs = dict(client_kwargs or {})
        base_url = str(kwargs.get("base_url") or "")
        if _is_cursor_provider(getattr(agent, "provider", None)) or _is_cursor_base_url(base_url):
            client = CursorAgentClient(
                api_key=kwargs.get("api_key"),
                base_url=base_url or CURSOR_MARKER_BASE_URL,
                default_headers=kwargs.get("default_headers"),
                command=getattr(agent, "acp_command", None) or kwargs.get("command"),
                args=getattr(agent, "acp_args", None) or kwargs.get("args"),
            )
            # CursorAgentClient streams cursor-agent's incremental stream-json
            # deltas as OpenAI-style chunks (see _create_chat_completion_stream),
            # so leave streaming enabled — disabling it buffers the whole reply
            # and forces the user to wait for the full latency before seeing any
            # text.  Explicitly clear any stale disable flag from a prior run.
            try:
                agent._disable_streaming = False
            except Exception:
                pass
            return client
        return _orig(agent, client_kwargs, reason=reason, shared=shared)

    agent_runtime_helpers.create_openai_client = _wrapped


# ---------------------------------------------------------------------------
# 7. auxiliary_client._to_async_client (do not wrap the shim in AsyncOpenAI)
# ---------------------------------------------------------------------------
@_safe_patch("auxiliary_client._to_async_client")
def _patch_to_async():
    from agent import auxiliary_client

    _orig = auxiliary_client._to_async_client

    def _wrapped(sync_client, model, is_vision=False):
        if isinstance(sync_client, CursorAgentClient):
            return AsyncCursorAgentClient(sync_client), model
        return _orig(sync_client, model, is_vision)

    auxiliary_client._to_async_client = _wrapped


# ---------------------------------------------------------------------------
# 7b. auxiliary_client.resolve_provider_client — aux tasks using cursor-agent
# ---------------------------------------------------------------------------
@_safe_patch("auxiliary_client.resolve_provider_client")
def _patch_resolve_provider_client():
    from agent import auxiliary_client

    _orig = auxiliary_client.resolve_provider_client

    def _wrapped(
        provider,
        model=None,
        async_mode=False,
        raw_codex=False,
        explicit_base_url=None,
        explicit_api_key=None,
        api_mode=None,
        main_runtime=None,
        is_vision=False,
        task=None,
    ):
        if _is_cursor_provider(provider):
            from hermes_cli.auth import resolve_external_process_provider_credentials

            creds = resolve_external_process_provider_credentials("cursor-agent")
            final_model = auxiliary_client._normalize_resolved_model(
                model
                or (main_runtime.get("model") if main_runtime else None)
                or auxiliary_client._read_main_model(),
                "cursor-agent",
            )
            api_key = str(creds.get("api_key", "")).strip()
            base_url = str(creds.get("base_url", "")).strip()
            command = str(creds.get("command", "")).strip() or None
            args = list(creds.get("args") or [])
            if not final_model:
                auxiliary_client.logger.warning(
                    "resolve_provider_client: cursor-agent requested but no model "
                    "was provided or configured"
                )
                return None, None
            if not api_key or not base_url:
                auxiliary_client.logger.warning(
                    "resolve_provider_client: cursor-agent requested but external "
                    "process credentials are incomplete"
                )
                return None, None
            client = CursorAgentClient(
                api_key=api_key,
                base_url=base_url,
                command=command,
                args=args,
            )
            auxiliary_client.logger.debug(
                "resolve_provider_client: %s (%s)", "cursor-agent", final_model
            )
            return (
                auxiliary_client._to_async_client(
                    client, final_model, is_vision=is_vision
                )
                if async_mode
                else (client, final_model)
            )
        return _orig(
            provider,
            model=model,
            async_mode=async_mode,
            raw_codex=raw_codex,
            explicit_base_url=explicit_base_url,
            explicit_api_key=explicit_api_key,
            api_mode=api_mode,
            main_runtime=main_runtime,
            is_vision=is_vision,
            task=task,
        )

    auxiliary_client.resolve_provider_client = _wrapped


# ---------------------------------------------------------------------------
# 7c. auxiliary_client._PROVIDER_ALIASES — cursor / cursor-cli → cursor-agent
# ---------------------------------------------------------------------------
@_safe_patch("auxiliary_client._PROVIDER_ALIASES")
def _patch_aux_aliases():
    from agent import auxiliary_client

    aliases = getattr(auxiliary_client, "_PROVIDER_ALIASES", None)
    if isinstance(aliases, dict):
        aliases.setdefault("cursor", "cursor-agent")
        aliases.setdefault("cursor-cli", "cursor-agent")


# ---------------------------------------------------------------------------
# 8. AIAgent._provider_model_requires_responses_api guard (cursor → False)
#    Prevents cursor's gpt-5.x model names from being upgraded to the Codex
#    Responses transport in code paths that pass api_mode=None.
# ---------------------------------------------------------------------------
@_safe_patch("AIAgent._provider_model_requires_responses_api")
def _patch_responses_guard():
    import run_agent

    _orig = run_agent.AIAgent._provider_model_requires_responses_api

    def _wrapped(self, model, *args, provider=None, **kwargs):
        prov = provider if provider is not None else getattr(self, "provider", "")
        if _is_cursor_provider(prov) or _is_cursor_base_url(getattr(self, "base_url", "")):
            return False
        return _orig(self, model, *args, provider=provider, **kwargs)

    run_agent.AIAgent._provider_model_requires_responses_api = _wrapped


# ---------------------------------------------------------------------------
# 9. models catalog + picker entries
# ---------------------------------------------------------------------------
@_safe_patch("models.catalog")
def _patch_models():
    from hermes_cli import models

    models._PROVIDER_MODELS.setdefault("cursor-agent", list(_FALLBACK_MODELS))

    # Add to the canonical picker list (avoid dup on reload).
    if not any(getattr(e, "slug", None) == "cursor-agent" for e in models.CANONICAL_PROVIDERS):
        models.CANONICAL_PROVIDERS.append(
            models.ProviderEntry(
                "cursor-agent",
                "Cursor Agent",
                "Cursor Agent CLI (headless cursor-agent via CURSOR_API_KEY)",
            )
        )

    _orig = models.provider_model_ids

    def _wrapped(provider, *, force_refresh=False):
        if _is_cursor_provider(provider):
            try:
                live = fetch_cursor_models()
                if live:
                    return live
            except Exception:
                pass
            return list(models._PROVIDER_MODELS.get("cursor-agent", _FALLBACK_MODELS))
        return _orig(provider, force_refresh=force_refresh)

    models.provider_model_ids = _wrapped


# ---------------------------------------------------------------------------
# 10. model_metadata provider-prefix awareness
# ---------------------------------------------------------------------------
@_safe_patch("model_metadata._PROVIDER_PREFIXES")
def _patch_metadata_prefixes():
    from agent import model_metadata

    prefixes = getattr(model_metadata, "_PROVIDER_PREFIXES", None)
    if prefixes is not None and "cursor-agent" not in prefixes:
        model_metadata._PROVIDER_PREFIXES = frozenset(set(prefixes) | {"cursor-agent"})


# ---------------------------------------------------------------------------
# 10b. providers.HERMES_OVERLAYS — so resolve_provider_full()/get_provider()
#      recognise cursor-agent (used by `hermes doctor` config validation and
#      --provider resolution).
# ---------------------------------------------------------------------------
@_safe_patch("providers.HERMES_OVERLAYS")
def _patch_providers_overlay():
    from hermes_cli import providers as P

    if "cursor-agent" not in P.HERMES_OVERLAYS:
        P.HERMES_OVERLAYS["cursor-agent"] = P.HermesOverlay(
            transport="openai_chat",
            auth_type="external_process",
            extra_env_vars=("CURSOR_API_KEY",),
            base_url_override=CURSOR_MARKER_BASE_URL,
            base_url_env_var="CURSOR_AGENT_BASE_URL",
        )
    aliases = getattr(P, "ALIASES", None)
    if isinstance(aliases, dict):
        aliases.setdefault("cursor", "cursor-agent")
        aliases.setdefault("cursor-cli", "cursor-agent")


# ---------------------------------------------------------------------------
# 11. `hermes model` interactive setup flow for cursor
#     The dispatcher in hermes_cli/main.py falls through to
#     _model_flow_api_key_provider when _is_profile_api_key_provider() is True.
#     We patch both (in the main namespace) so selecting Cursor in the picker
#     writes a correct external-process config block.
# ---------------------------------------------------------------------------
@_safe_patch("main.cursor_setup_flow")
def _patch_setup_flow():
    from hermes_cli import main as _main

    _orig_is_api = getattr(_main, "_is_profile_api_key_provider", None)
    _orig_flow = getattr(_main, "_model_flow_api_key_provider", None)
    if _orig_is_api is None or _orig_flow is None:
        raise RuntimeError("setup-flow hooks not found")

    def _is_api(provider):
        if _is_cursor_provider(provider):
            return True
        return _orig_is_api(provider)

    def _flow(config, selected_provider, current_model=""):
        if _is_cursor_provider(selected_provider):
            return _cursor_model_flow(config, current_model)
        return _orig_flow(config, selected_provider, current_model)

    _main._is_profile_api_key_provider = _is_api
    _main._model_flow_api_key_provider = _flow


def _cursor_model_flow(config, current_model=""):
    from hermes_cli.config import save_config
    from hermes_cli.models import provider_model_ids

    models = provider_model_ids("cursor-agent") or list(_FALLBACK_MODELS)
    chosen = current_model if current_model in models else models[0]

    # Try the interactive numbered picker that hermes uses elsewhere; fall back
    # to the first model if no TTY/picker is available.
    try:
        from hermes_cli.main import _prompt_provider_choice

        idx = _prompt_provider_choice(
            list(models), default=models.index(chosen), title="Select Cursor model:"
        )
        if idx is not None:
            chosen = models[idx]
    except Exception:
        pass

    model_cfg = config.setdefault("model", {})
    model_cfg["provider"] = "cursor-agent"
    model_cfg["base_url"] = CURSOR_MARKER_BASE_URL
    model_cfg["default"] = chosen
    model_cfg["api_mode"] = "chat_completions"
    save_config(config)
    print(f"✓ Cursor Agent configured (model: {chosen}).")
    print("  Requires: cursor-agent CLI on PATH + CURSOR_API_KEY in ~/.hermes/.env")


# ---------------------------------------------------------------------------
# 11b. doctor — show Cursor Agent row in Auth Providers section
# ---------------------------------------------------------------------------
@_safe_patch("doctor.cursor_auth_row")
def _patch_doctor():
    from hermes_cli import doctor

    _orig_section = doctor._section
    _cursor_row_printed = [False]

    def _print_cursor_auth_row() -> None:
        from hermes_cli.auth import get_external_process_provider_status

        st = get_external_process_provider_status("cursor-agent")
        if st.get("configured"):
            cmd = st.get("resolved_command") or st.get("command") or "cursor-agent"
            doctor.check_ok("Cursor Agent", f"(CLI: {cmd}, CURSOR_API_KEY set)")
        else:
            missing = []
            if not st.get("resolved_command"):
                missing.append("cursor-agent CLI")
            if not st.get("api_key_configured"):
                missing.append("CURSOR_API_KEY")
            detail = ", ".join(missing) if missing else "not configured"
            doctor.check_warn("Cursor Agent", f"({detail})")

    def _wrapped_section(title: str) -> None:
        if title == "Directory Structure" and not _cursor_row_printed[0]:
            _cursor_row_printed[0] = True
            try:
                _print_cursor_auth_row()
            except Exception as exc:  # pragma: no cover
                doctor.check_warn("Cursor Agent", f"(could not check: {exc})")
        _orig_section(title)

    doctor._section = _wrapped_section


# ---------------------------------------------------------------------------
# 11c. conversation_loop — enable SSE streaming for cursor-agent subprocess
#
# CursorAgentClient now streams cursor-agent's incremental stream-json deltas
# as OpenAI-style chunks (see CursorAgentClient._create_chat_completion_stream
# + the --stream-partial-output flag), so the SSE/token streaming path works.
# Force ``_disable_streaming`` OFF at the start of every cursor conversation —
# this wrapper runs after client creation, so it must enable (not disable)
# streaming for the change to take effect.
# ---------------------------------------------------------------------------
@_safe_patch("conversation_loop.streaming")
def _patch_conversation_streaming():
    import inspect
    from agent import conversation_loop

    source = inspect.getsource(conversation_loop.run_conversation)
    if "acp://cursor" in source:
        return  # upstream already handles cursor

    _orig = conversation_loop.run_conversation

    def _wrapped(agent, *args, **kwargs):
        if _is_cursor_provider(getattr(agent, "provider", None)) or _is_cursor_base_url(
            getattr(agent, "base_url", "")
        ):
            try:
                agent._disable_streaming = False
            except Exception:
                pass
        return _orig(agent, *args, **kwargs)

    conversation_loop.run_conversation = _wrapped


# ---------------------------------------------------------------------------
# 11d. gateway.run — deliver restart/startup notifications reliably
#
# Core-gateway race (not cursor-specific; patched here because this plugin is
# the maintained user plugin on this install): right after a (re)start the
# Telegram adapter marks its send path degraded until the first getUpdates
# long-poll completes (~10s), but the gateway fires its "restarted" / "online"
# notification about 1s after connect and never retries. The restart itself
# succeeds, yet the confirmation is always dropped with `send_path_degraded`,
# so from chat it looks like /restart did nothing. Wait (bounded) for the
# send path to become healthy before sending.
# ---------------------------------------------------------------------------
@_safe_patch("gateway.run.notification_send_path_wait")
def _patch_gateway_notifications():
    import asyncio
    import time as _time

    from gateway import run as gateway_run

    runner_cls = gateway_run.GatewayRunner

    async def _wait_send_path_ready(runner, timeout: float = 60.0) -> None:
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            adapters = list(getattr(runner, "adapters", {}).values() or ())
            if adapters and not any(
                getattr(adapter, "_send_path_degraded", False)
                for adapter in adapters
            ):
                return
            await asyncio.sleep(0.5)

    _orig_restart = runner_cls._send_restart_notification
    if not getattr(_orig_restart, "_cursor_plugin_wrapped", False):

        async def _restart_wrapped(self):
            await _wait_send_path_ready(self)
            return await _orig_restart(self)

        _restart_wrapped._cursor_plugin_wrapped = True  # type: ignore[attr-defined]
        runner_cls._send_restart_notification = _restart_wrapped

    _orig_home = runner_cls._send_home_channel_startup_notifications
    if not getattr(_orig_home, "_cursor_plugin_wrapped", False):

        async def _home_wrapped(self, **kwargs):
            await _wait_send_path_ready(self)
            return await _orig_home(self, **kwargs)

        _home_wrapped._cursor_plugin_wrapped = True  # type: ignore[attr-defined]
        runner_cls._send_home_channel_startup_notifications = _home_wrapped


# ---------------------------------------------------------------------------
# 11e. model_normalize — strip vendor-only prefix for cursor-agent models
# ---------------------------------------------------------------------------
@_safe_patch("model_normalize._STRIP_VENDOR_ONLY_PROVIDERS")
def _patch_model_normalize():
    from hermes_cli import model_normalize

    providers = getattr(model_normalize, "_STRIP_VENDOR_ONLY_PROVIDERS", None)
    if providers is not None and "cursor-agent" not in providers:
        model_normalize._STRIP_VENDOR_ONLY_PROVIDERS = frozenset(
            set(providers) | {"cursor-agent"}
        )


# ---------------------------------------------------------------------------
# Deferred patch application.
#
# This plugin is imported from INSIDE hermes_cli/auth.py's module body (auth
# eagerly discovers provider plugins). At that point auth — and every core
# module that imports auth — is only partially initialized, so patching now
# both fails AND poisons the import graph (tools/other plugins break on the
# circular import). Instead we register a MetaPathFinder that applies each
# patch only AFTER its target module finishes loading.
#
# auth's own patches run as soon as ANY module that imports auth finishes
# loading: by then auth is guaranteed complete (the importer blocked on it).
# ---------------------------------------------------------------------------
import importlib.abc  # noqa: E402

_AUTH_PATCHES = (_patch_registry, _patch_creds, _patch_status, _patch_auth_status)
_MODULE_PATCHES = {
    "hermes_cli.runtime_provider": (_patch_runtime,),
    "agent.agent_runtime_helpers": (_patch_create_client,),
    "agent.auxiliary_client": (
        _patch_to_async,
        _patch_resolve_provider_client,
        _patch_aux_aliases,
    ),
    "run_agent": (_patch_responses_guard,),
    "hermes_cli.models": (_patch_models,),
    "agent.model_metadata": (_patch_metadata_prefixes,),
    "hermes_cli.providers": (_patch_providers_overlay,),
    "hermes_cli.main": (_patch_setup_flow,),
    "hermes_cli.doctor": (_patch_doctor,),
    "agent.conversation_loop": (_patch_conversation_streaming,),
    "hermes_cli.model_normalize": (_patch_model_normalize,),
    "gateway.run": (_patch_gateway_notifications,),
}

_auth_done = [False]
_done_modules: set[str] = set()


def _module_ready(name: str) -> bool:
    mod = sys.modules.get(name)
    if mod is None:
        return False
    spec = getattr(mod, "__spec__", None)
    return not getattr(spec, "_initializing", False)


def _ensure_auth_patched() -> None:
    if _auth_done[0] or not _module_ready("hermes_cli.auth"):
        return
    _auth_done[0] = True
    for fn in _AUTH_PATCHES:
        fn()


def _run_module_patches(name: str) -> None:
    _ensure_auth_patched()
    if name in _done_modules:
        return
    _done_modules.add(name)
    for fn in _MODULE_PATCHES.get(name, ()):  # type: ignore[arg-type]
        fn()


class _CursorPatchFinder(importlib.abc.MetaPathFinder):
    """Wrap target modules' loaders so we patch them post-load."""

    def find_spec(self, fullname, path=None, target=None):
        # auth's patches must land as early as possible (the doctor / picker
        # query PROVIDER_REGISTRY before any module in _MODULE_PATCHES loads).
        # Once auth is fully initialized, the very next import — whatever it is
        # — triggers them. The check is a cheap bool short-circuit afterwards.
        if not _auth_done[0]:
            _ensure_auth_patched()
        if fullname not in _MODULE_PATCHES:
            return None
        try:
            idx = sys.meta_path.index(self)
        except ValueError:
            return None
        spec = None
        for finder in sys.meta_path[idx + 1:]:
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        if getattr(loader, "_cursor_wrapped", False):
            return spec
        _orig_exec = loader.exec_module

        def _exec(module, __orig=_orig_exec, __name=fullname):
            __orig(module)
            try:
                _run_module_patches(__name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("cursor-agent post-load patch %s failed: %s", __name, exc)

        try:
            loader.exec_module = _exec  # type: ignore[method-assign]
            loader._cursor_wrapped = True  # type: ignore[attr-defined]
        except Exception:
            return spec
        return spec


_draining = [False]


def _drain_pending() -> None:
    """Apply every patch whose target module is now fully loaded.

    Re-entrancy guarded: applying a patch imports its target, which re-enters
    our __import__ hook — the guard makes that a no-op.
    """
    if _draining[0]:
        return
    _draining[0] = True
    try:
        _ensure_auth_patched()
        for name in list(_MODULE_PATCHES):
            if name not in _done_modules and _module_ready(name):
                _run_module_patches(name)
    finally:
        _draining[0] = False


# The finder catches modules first-loaded AFTER us; the __import__ hook is the
# reliable backstop that drains pending patches after EVERY import statement
# (including ``from x import y`` on already-loaded modules), which is how the
# patches land for modules that were mid-load when this plugin was imported.
if not any(isinstance(f, _CursorPatchFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _CursorPatchFinder())

import builtins  # noqa: E402

_orig_import = builtins.__import__


def _cursor_import(name, *args, **kwargs):
    module = _orig_import(name, *args, **kwargs)
    if not _draining[0] and (
        not _auth_done[0] or len(_done_modules) < len(_MODULE_PATCHES)
    ):
        _drain_pending()
    return module


if getattr(builtins, "__import__", None) is not _cursor_import:
    builtins.__import__ = _cursor_import

# Catch up on anything already fully loaded before us.
_drain_pending()

logger.info(
    "cursor-agent plugin armed (finder + import hook installed; applied so far=%d)",
    len(_applied),
)
