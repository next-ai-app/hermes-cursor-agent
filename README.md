# hermes-cursor-agent

[English](#english) | [繁體中文](#繁體中文)

Hermes Agent **model provider plugin** that routes chat requests to the local [Cursor Agent CLI](https://cursor.com/docs/cli/overview) (`cursor-agent`).

Hermes speaks OpenAI-style chat completions; this plugin spawns a short-lived `cursor-agent` process per request, parses `stream-json` output, and streams incremental assistant text back to Hermes (CLI, gateway, Open WebUI, Telegram, etc.).

## Architecture

```text
Hermes (any platform)
  -> CursorAgentClient (OpenAI-compatible shim)
    -> cursor-agent -p --output-format stream-json --stream-partial-output ...
      -> Cursor cloud models + local tools
```

The plugin lives entirely under `$HERMES_HOME/plugins/model-providers/cursor-agent/`, so `hermes update` never overwrites it.

### Components

| Path | Role |
|------|------|
| `model-providers/cursor-agent/cursor_agent_client.py` | Subprocess runner, prompt formatting, stream-json parser, OpenAI chunk adapter |
| `model-providers/cursor-agent/__init__.py` | Hermes integration: provider profile, auth, runtime resolution, streaming, doctor, model picker |
| `scripts/cursor_agent_stream_filter.py` | Optional NDJSON filter for live logs (redacts secrets, omits full user prompts) |

## Requirements

- [Hermes Agent](https://hermes-agent.nousresearch.com/docs) installed
- [Cursor Agent CLI](https://cursor.com/docs/cli/overview) on `PATH` (`cursor-agent` or `agent`)
- `CURSOR_API_KEY` from your Cursor account

## Install

```bash
git clone https://github.com/StrawCoding/hermes-cursor-agent.git
cd hermes-cursor-agent
chmod +x install.sh
./install.sh
```

Manual install (equivalent):

```bash
mkdir -p ~/.hermes/plugins/model-providers
cp -a model-providers/cursor-agent ~/.hermes/plugins/model-providers/
```

## Configure

1. Put your API key in `~/.hermes/.env`:

   ```bash
   CURSOR_API_KEY=your_key_here
   ```

2. Select the provider:

   ```bash
   hermes model
   # choose Cursor Agent / cursor-agent
   ```

   Or edit `~/.hermes/config.yaml`:

   ```yaml
   model:
     provider: cursor-agent
     base_url: acp://cursor
     default: auto
     api_mode: chat_completions
   ```

3. Verify:

   ```bash
   hermes doctor
   cursor-agent --list-models
   ```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CURSOR_API_KEY` | Cursor API key (required unless `CURSOR_API_KEYS` is set) |
| `CURSOR_API_KEYS` | Optional key pool: comma/semicolon/whitespace-separated keys |
| `HERMES_CURSOR_KEY_COOLDOWN_SECONDS` | Cooldown for a usage-limited pool key (default: `3600`) |
| `HERMES_CURSOR_AGENT_COMMAND` | Override CLI binary (default: `cursor-agent`) |
| `CURSOR_AGENT_PATH` | Alias for command override |
| `HERMES_CURSOR_AGENT_ARGS` | Extra CLI args (shell-split) |
| `HERMES_CURSOR_DEFAULT_MODEL` | Default model when none specified (default: `auto`) |
| `CURSOR_AGENT_BASE_URL` | Marker base URL override (default: `acp://cursor`) |

## API key pool

Set `CURSOR_API_KEYS` to rotate across multiple Cursor accounts:

```bash
# ~/.hermes/.env
CURSOR_API_KEYS=key_one,key_two,key_three
```

`CURSOR_API_KEY` (if also set) stays first in the pool, so existing setups
are unaffected. When the active key reports a usage limit, it is put on a
cooldown (default 1 h) and the next healthy key takes over transparently —
the request is retried with the same model before any downgrade to `auto`.
The env is re-read on every request, so keys can be added without a restart.

## Streaming

When Hermes requests `stream=True`, the client forwards `cursor-agent` incremental `assistant` deltas as OpenAI-style chunks so UIs show text progressively.

If a model reports a usage limit, the client transparently retries on the next pool key (same model), then falls back to `auto`.

## Live log filter (optional)

Pipe raw `stream-json` through the filter for public or tail-friendly logs:

```bash
cursor-agent -p --output-format stream-json --stream-partial-output ... \
  | python3 scripts/cursor_agent_stream_filter.py
```

## Development

```bash
python3 -m pytest tests/ -q
```

## Related projects

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — host framework
- [hjcenry/hermes-cursor-agent-plugin](https://github.com/hjcenry/hermes-cursor-agent-plugin) — tool-style Cursor CLI bridge (different design)
- [Cosmic-Construct/hermes-cursor-harness](https://github.com/Cosmic-Construct/hermes-cursor-harness) — SDK/ACP harness plugin

## License

MIT — see [LICENSE](LICENSE).

---

<a id="繁體中文"></a>

# hermes-cursor-agent（繁體中文）

將 [Hermes Agent](https://hermes-agent.nousresearch.com/docs) 的對話請求轉發到本機 [Cursor Agent CLI](https://cursor.com/docs/cli/overview) 的 **model provider 外掛**。

Hermes 使用 OpenAI 相容的 chat completions 介面；此外掛每次請求會啟動 `cursor-agent` 子行程，解析 `stream-json`，並把增量文字串流回 Hermes（CLI、閘道、Open WebUI、Telegram 等）。

## 安裝

```bash
git clone https://github.com/StrawCoding/hermes-cursor-agent.git
cd hermes-cursor-agent
chmod +x install.sh
./install.sh
```

## 設定

1. 在 `~/.hermes/.env` 設定 `CURSOR_API_KEY`
2. 執行 `hermes model` 選擇 Cursor Agent
3. `hermes doctor` 確認 CLI 與金鑰就緒

## 元件說明

- `cursor_agent_client.py`：子行程、prompt 格式化、stream-json 解析、OpenAI chunk 轉換
- `__init__.py`：註冊 provider、認證、runtime、串流、doctor、模型選擇等 Hermes 整合
- `cursor_agent_stream_filter.py`：可選的即時 log 過濾器（遮蔽密鑰、省略完整 user prompt）

## 授權

MIT
