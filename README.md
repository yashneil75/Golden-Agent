# Golden Agent

**Your local AI agent. No cloud. No API keys. No excuses.**

A terminal coding agent that runs entirely on your machine — auto-installs its own
GPU-accelerated inference stack, auto-downloads models, and gets to work on basically
any post 2000 hardware.
---

## Why Golden Agent?

Every AI coding assistant wants your code uploaded to someone else's server.
Golden Agent flips that: the model, the tools, and your data never leave your machine.

- **100% local** — inference happens on-device via llama.cpp. Nothing is sent anywhere (except when *you* ask it to search or fetch the web).
- **Zero-config setup** — one `pip install`, one command. Golden Agent detects your GPU and downloads the matching prebuilt llama.cpp backend automatically (Vulkan on Windows/Linux, Metal on macOS, CUDA where available, CPU otherwise), then pulls the model weights from Hugging Face with resumable, retry-hardened downloads.
- **Runs on basically any hardware** — Vulkan means one code path for NVIDIA, AMD, and Intel GPUs. Full offload, flash attention, and Q5_0 KV-cache quantization squeeze maximum performance out of modest VRAM. Small models start at ~1.5 GB — a laptop iGPU can run it.
- **A real agent, not a chatbot** — reads and edits files, runs shell commands, searches the web, fetches pages, and loops on results until the job is done. Writes outside your project ask first.
- **Models for everyone** — three sizes from 1.5 GB to 8.2 GB, switchable with a numbered picker on startup.

## Quickstart

```bash
pip install golden-agent
golden-agent setup
golden-agent
```

That's it. First launch:
1. Golden Agent sets up the llama.cpp wheel for your platform.
1. Pick a model
3. The model downloads on first run (~1.5–9 GB, resumable if interrupted).
4. You get a prompt: `❯` — start asking it to do things.

```text
❯ fix the failing test in tests/test_agent.py
❯ search for how uv handles lockfiles and summarize
❯ write a script that renames all photos by EXIF date
```

## Models

| Key | Model | Size | Context | Good for |
|---|---|---|---|---|
| `tiny` | [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) | ~1.5 GB | 128K | Low-end hardware, laptops, iGPUs |
| `lite` *(default)* | [Ornith-1.5-9B](https://huggingface.co/AtomicChat/Ornith-1.5-9B-GGUF) | ~5.2 GB | 153K | Best balance of speed and capability |
| `pro` | [Qwen3.8-27B](https://huggingface.co/sdkyuan/qwen3.8-27b-qat-q2_0-gguf) | ~8.2 GB | 153K | Heavy reasoning when you can afford it |

## Built-in tools

Golden Agent ships with six tools the model uses autonomously:

| Tool | What it does |
|---|---|
| `read_file` | Reads files with automatic paging for large ones |
| `write_file` | Creates/overwrites files |
| `edit_file` | Exact-match string replacement (with `replace_all`) |
| `bash` | Runs real shell commands — actual bash even on Windows |
| `web_search` | DuckDuckGo search |
| `web_fetch` | Fetches a URL and extracts readable text |

**Guardrails:** file writes outside the directory you launched from trigger an
interactive permission prompt (`once` / `always this session` / `deny`). Tool output
is capped so runaway commands can't flood the context.

## What you'll see

- **Live streaming markdown** as the model writes.
- **Visible reasoning** — `<think>` blocks stream inline, dimmed, so you can watch it think.
- **Tool activity panels** — file diffs and command output render with syntax highlighting; reads and searches show as quiet status lines.
- **Automatic context compaction** — past 140K tokens, older turns are summarized by the model itself so long sessions keep going.

## Slash commands

```
/tools            list available tools
/clear            wipe conversation history
/help             show help
/exit             quit
```

## Requirements

- Python 3.10+
- A GPU (NVIDIA, AMD, Intel) — or it still works on CPU, just slower.
- Internet once, for the initial setup + model download. After that: fully offline.

Golden Agent downloads a prebuilt llama.cpp binary for your detected backend, so no
compilation is required. If you want to build a custom backend yourself, the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home) is only needed for that manual path.

## Under the hood

- **Inference:** the prebuilt `llama-server` with full GPU offload (`-ngl 99`), flash attention, and Q5_0-quantized KV cache.
- **Downloads:** direct-from-HuggingFace GGUF fetching with HTTP range resume, byte-exact validation, and exponential-backoff retries.
- **Tool calling:** hand-rolled parsers for both Qwen-style XML calls and LFM2 pythonic call syntax — robust to streaming chunk boundaries, unclosed tags, and parallel calls.
- **Model cache:** `%LOCALAPPDATA%\golden-agent\models` (Windows), `~/Library/Caches/golden-agent/models` (macOS), `$XDG_CACHE_HOME/golden-agent/models` (Linux).

### Debugging

Set `GOLDEN_AGENT_RAW_LOG` to a file path to capture every raw model response — invaluable
when a tool call goes sideways:

```bash
GOLDEN_AGENT_RAW_LOG=./raw.log golden-agent
```

## Development

```bash
git clone <repo> && cd golden-agent
pip install -e .[dev]
pytest
ruff check src tests
```

Run without installing:

```bash
pip install -e .
python -m golden_agent
```

## License

MIT — see [LICENSE](LICENSE).
