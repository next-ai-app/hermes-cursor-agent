# hermes-cursor-agent

[English](#english) | [繁體中文](#繁體中文)

Hermes Agent **model provider plugin** that routes chat requests to [Cursor Agent](https://cursor.com/docs/cli/overview) — via the `cursor_sdk` ACP bridge when available, with automatic fallback to the `cursor-agent` CLI.

Hermes speaks OpenAI-style chat completions; this plugin returns OpenAI-shaped responses/chunks while Cursor's cloud models + local tools do the work underneath.

## Architecture

```text
Hermes (any platform)
  -> CursorAgentClient (OpenAI-compatible shim)
    -> 1. cursor_sdk ACP agent (persistent, per-conversation, multi-turn)
       2. fallback: cursor-agent -p --output-format stream-json (short-lived subprocess)
         -> Cursor cloud models + local tools
```

- **SDK path (preferred):** a persistent agent per conversation keeps server-side multi-turn context, so follow-ups are incremental and cheap.
- **CLI path (fallback):** spawns a fresh `cursor-agent` process per request and replays the full transcript as a single prompt. Used automatically when the SDK is unavailable or unhealthy (300s cooldown after a failure, then retried).

The plugin lives entirely under `$HERMES_HOME/plugins/model-providers/cursor-agent/`, so `hermes update` never overwrites it.

### Components

| Path | Role |
|------|------|
| `model-providers/cursor-agent/cursor_agent_client.py` | SDK agent management, subprocess runner, prompt formatting, stream-json parser, OpenAI chunk adapter, liveness/watchdog logic |
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

## Model names: CLI vs SDK

The CLI accepts ~193 model identifiers (including variants like `claude-fable-5-thinking-high`, `gpt-5.5-medium`, `composer-2.5-fast`). The SDK accepts only the ~33 base names (`claude-fable-5`, `gpt-5.5`, `composer-2.5`, …) plus `default`.

The plugin normalizes automatically on the SDK path: variant names map to their base model, and `auto` maps to `default`. Unknown names pass through unchanged. You can keep using CLI-style names everywhere.

Some models (e.g. Fable 5) require acknowledging a data-retention policy in your Cursor account settings (cursor.com → Settings) before first use; otherwise every call fails with `ActionRequiredError: Review Data Policy`.

## Reliability behavior

The plugin is designed to **surface the real error** instead of a bare "Empty response", and to recover without restarts:

| Situation | Behavior |
|-----------|----------|
| SDK run fails | Falls back to the CLI for that turn; SDK retried after a 300s cooldown |
| SDK "succeeds" with empty content (dead transport) | Agent evicted, treated as SDK failure, turn retried via CLI |
| Model out of usage on a named model | Transparent one-time retry with `auto` |
| CLI turn hangs (blocking read) | Wall-clock watchdog kills the **whole process group** (children included) and raises `TimeoutError` |
| CLI exits 0 with no output | Raises with the model name + likely causes (e.g. unacknowledged data policy) |
| CLI exits 0 with only stderr | The real stderr (e.g. `ActionRequiredError: …`) is included in the exception |
| Long agentic turn (tool calls, no reply text yet) | `tool_call` events surface as reasoning chunks (`🔧 tool name`); a keep-alive heartbeat every 30s prevents Hermes' stale-stream kill |
| Truly stalled stream (no output at all) | Killed after 180s with an explicit "connection likely stalled" error |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CURSOR_API_KEY` | Required. Cursor API key |
| `HERMES_CURSOR_AGENT_COMMAND` | Override CLI binary (default: `cursor-agent`) |
| `CURSOR_AGENT_PATH` | Alias for command override |
| `HERMES_CURSOR_AGENT_ARGS` | Extra CLI args (shell-split) |
| `HERMES_CURSOR_DEFAULT_MODEL` | Default model when none specified (default: `auto`) |
| `CURSOR_AGENT_BASE_URL` | Marker base URL override (default: `acp://cursor`) |
| `HERMES_CURSOR_TIMEOUT_SECONDS` | Per-turn wall-clock timeout for CLI turns |
| `HERMES_CURSOR_AGENT_TTL_SECONDS` | Model-catalog cache TTL |
| `HERMES_CURSOR_STREAM_HEARTBEAT_S` | Keep-alive interval for busy turns (default: `30`) |
| `HERMES_CURSOR_STREAM_IDLE_KILL_S` | Kill streams with zero output after this long (default: `180`) |

## Streaming

When Hermes requests `stream=True`, incremental `assistant` deltas stream back as OpenAI-style chunks. Tool-call progress and heartbeats arrive as `reasoning` content on chunks, so Hermes' stale-stream detector stays satisfied during long agentic turns and users see live progress.

Short replies (or backends that only emit the final `result` event) are emitted as a single chunk, so replies are never lost.

## Troubleshooting

- **`ActionRequiredError: Review Data Policy`** — acknowledge the model's data-retention policy in cursor.com → Settings, or switch to another model (`/model auto`).
- **Repeated empty replies that only a restart fixes** — was the dead-SDK-transport symptom; current versions route around it via the CLI automatically. Check logs for `SDK run returned empty content without an error` diagnostics.
- **"Busy" for a long time with no reply** — check whether the turn is doing legitimate agentic tool work (you should see `🔧`/`⏳` progress). If it dies with "no output … likely stalled", the network to Cursor's API is the bottleneck; tune `HERMES_CURSOR_STREAM_IDLE_KILL_S` if your link is merely slow.
- Gateway logs: `journalctl --user -u hermes-gateway.service` (add your profile suffix, e.g. `hermes-gateway-diver.service`).

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

<a id=\"繁體中文\"></a>

# hermes-cursor-agent（繁體中文）

將 [Hermes Agent](https://hermes-agent.nousresearch.com/docs) 的對話請求轉發到 [Cursor Agent](https://cursor.com/docs/cli/overview) 的 **model provider 外掛** — 優先使用 `cursor_sdk` ACP bridge，失效時自動 fallback 到 `cursor-agent` CLI。

Hermes 使用 OpenAI 相容的 chat completions 介面；此外掛負責轉換，底層由 Cursor 雲端模型 + 本機工具執行。

## 架構

- **SDK path（優先）**：每個對話一個持久 agent，保留 server-side multi-turn context，follow-up 只需增量傳送。
- **CLI path（fallback）**：每次請求啟動新 `cursor-agent` 子行程，重放完整 transcript。SDK 不可用或不健康時自動使用（失敗後冷卻 300 秒再重試）。

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

## 模型名稱：CLI vs SDK

CLI 接受約 193 個模型名（包括 `claude-fable-5-thinking-high`、`gpt-5.5-medium` 等 variant）；SDK 只接受約 33 個 base 名加 `default`。外掛會自動 normalize：variant 名映射到 base 名，`auto` 映射到 `default`，未知名稱原樣保留。

部分模型（例如 Fable 5）需要先在 cursor.com → Settings 確認 data retention policy，否則所有呼叫都會失敗並回報 `ActionRequiredError: Review Data Policy`。

## 可靠性行為

設計原則：**報出真實錯誤**，唔會淨係得個 "Empty response"，而且唔使 restart 都會自我修復：

| 情況 | 行為 |
|------|------|
| SDK run 失敗 | 當前 turn 轉用 CLI；SDK 冷卻 300 秒後重試 |
| SDK「成功」但回空（transport 已死） | 剔除 agent、視為 SDK 失敗、經 CLI 重試 |
| 指定模型用量耗盡 | 自動用 `auto` 重試一次 |
| CLI turn 掛起（blocking read） | wall-clock watchdog 殺**整個 process group**（包括子行程）並 raise `TimeoutError` |
| CLI exit 0 但無輸出 | raise 時附模型名 + 可能原因（例如未確認 data policy） |
| CLI exit 0 只有 stderr | 真實 stderr（例如 `ActionRequiredError: …`）會放入 exception |
| 長時間 agentic turn（做緊 tool 工作、未有回覆文字） | `tool_call` event 以 reasoning chunk 顯示（`🔧 工具名`）；每 30 秒 keep-alive 防止 Hermes 誤殺 |
| Stream 完全無輸出（真正 stall） | 180 秒後殺掉並報明確原因 |

## 環境變數

除咗英文版列出嘅基本設定外，仲有：

| Variable | 用途 |
|----------|------|
| `HERMES_CURSOR_TIMEOUT_SECONDS` | 每個 CLI turn 嘅 wall-clock 上限 |
| `HERMES_CURSOR_AGENT_TTL_SECONDS` | 模型列表 cache TTL |
| `HERMES_CURSOR_STREAM_HEARTBEAT_S` | 忙碌 turn 嘅 keep-alive 間隔（預設 `30`） |
| `HERMES_CURSOR_STREAM_IDLE_KILL_S` | 完全無輸出幾耐之後殺 stream（預設 `180`） |

## 疑難排解

- **`ActionRequiredError: Review Data Policy`** — 去 cursor.com → Settings 確認該模型嘅 data retention policy，或轉用其他模型（`/model auto`）。
- **反覆回空、要 restart 先好** — 舊版嘅 SDK transport 死亡症狀；新版會自動繞道 CLI。Log 關鍵字：`SDK run returned empty content without an error`。
- **長時間「忙碌」無回覆** — 睇吓係咪做緊正常嘅 agentic tool 工作（應該會見到 `🔧`/`⏳` 進度）。如果报 "no output … likely stalled"，即係去 Cursor API 嘅網絡有問題；網速慢可以調大 `HERMES_CURSOR_STREAM_IDLE_KILL_S`。
- Gateway log：`journalctl --user -u hermes-gateway.service`（按 profile 加後綴）。

## 元件說明

- `cursor_agent_client.py`：SDK agent 管理、子行程、prompt 格式化、stream-json 解析、OpenAI chunk 轉換、liveness/watchdog
- `__init__.py`：註冊 provider、認證、runtime、串流、doctor、模型選擇等 Hermes 整合
- `cursor_agent_stream_filter.py`：可選的即時 log 過濾器（遮蔽密鑰、省略完整 user prompt）

## 授權

MIT
