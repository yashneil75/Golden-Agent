import json
from collections.abc import Callable, Generator
from urllib.parse import urljoin

import httpx

from golden_agent.backend import server_url, start_server
from golden_agent.config import DEFAULT_MODEL_KEY, MODELS, Backend, ModelSpec
from golden_agent.download import (
    ensure_draft_model_downloaded,
    ensure_model_downloaded,
    model_cached,
)


def selected_model() -> ModelSpec:
    return MODELS[_selected_key]


_selected_key: str = DEFAULT_MODEL_KEY
_server_ready: bool = False


def set_model(key: str) -> ModelSpec:
    global _selected_key
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; available: {', '.join(MODELS)}")
    _selected_key = key
    return MODELS[key]


def count_tokens(text: str) -> int:
    """Rough token count estimate."""
    return max(1, len(text) // 4)


def bootstrap_server(
    *,
    backend: Backend = Backend.CPU,
    download_progress: Callable[[int, int], None] | None = None,
    draft_progress: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Download model(s) and start llama-server."""
    global _server_ready
    if _server_ready:
        return

    spec = selected_model()

    if model_cached(filename=spec.filename, expected_size=spec.size_bytes):
        log(f"{spec.label} weights cached")
    else:
        log(f"repairing {spec.label} weights ...")
    target_path = str(ensure_model_downloaded(
        progress=download_progress,
        repo=spec.repo,
        filename=spec.filename,
        expected_size=spec.size_bytes,
    ))

    draft_path = None
    if spec.draft_filename and spec.draft_repo and spec.draft_size_bytes:
        if model_cached(filename=spec.draft_filename, expected_size=spec.draft_size_bytes):
            log(f"{spec.draft_filename} cached")
        else:
            log(f"repairing {spec.draft_filename} ...")
        draft_path = str(ensure_draft_model_downloaded(
            progress=draft_progress,
            repo=spec.draft_repo,
            filename=spec.draft_filename,
            expected_size=spec.draft_size_bytes,
        ))

    start_server(
        spec,
        target_path,
        draft_path=draft_path,
        backend=backend,
        log=log,
    )
    _server_ready = True



def stream_chat(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
) -> Generator[dict, None, None]:
    """Stream chat completion chunks from llama-server via SSE."""
    url = urljoin(server_url(), "/v1/chat/completions")

    payload: dict = {
        "model": "local",
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
    }

    spec = selected_model()
    if spec.sampling_overrides:
        for key, value in spec.sampling_overrides.items():
            payload[key] = value

    if spec.reasoning_effort:
        payload["reasoning_effort"] = spec.reasoning_effort

    if tools:
        payload["tools"] = tools

    with httpx.Client(timeout=httpx.Timeout(connect=10, read=300, write=10, pool=10)) as client:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        continue


def call_model_stream(
    prompt: str,
    *,
    system: str = "",
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    for chunk in stream_chat(messages):
        choices = chunk.get("choices") or [{}]
        delta = choices[0].get("delta", {})
        piece = delta.get("content")
        if piece:
            yield piece
