import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

SPEC_TYPE_NONE = "none"
SPEC_TYPE_DRAFT_DFLASH = "draft-dflash"


class Backend(str, Enum):
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    CPU = "cpu"


BACKEND_LABELS = {
    Backend.CUDA: "NVIDIA CUDA",
    Backend.METAL: "Apple Metal",
    Backend.VULKAN: "Vulkan",
    Backend.CPU: "CPU only",
}


def detect_backend() -> Backend:
    import shutil
    import sys

    if sys.platform == "darwin":
        return Backend.METAL

    if shutil.which("nvidia-smi"):
        return Backend.CUDA

    if shutil.which("vulkaninfo"):
        return Backend.VULKAN

    return Backend.CPU


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo: str
    filename: str
    size_bytes: int
    n_ctx: int | None = None
    sampling_overrides: dict[str, float] | None = None
    tool_format: str = "qwen3_xml"
    reasoning: str = "auto"
    reasoning_effort: str | None = None
    draft_repo: str | None = None
    draft_filename: str | None = None
    draft_size_bytes: int | None = None
    spec_type: str = SPEC_TYPE_NONE


MODELS = {
    "tiny": ModelSpec(
        key="tiny",
        label="LFM2.5-2.6B",
        repo="LiquidAI/LFM2.5-2.6B-GGUF",
        filename="LFM2.5-2.6B-QAD-Q4_0.gguf",
        size_bytes=1593894944,
        n_ctx=128000,
        sampling_overrides={"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1},
        tool_format="lfm2",
        reasoning="on",
    ),
    "lite": ModelSpec(
        key="lite",
        label="Ornith-1.5-9B",
        repo="AtomicChat/Ornith-1.5-9B-GGUF",
        filename="Ornith-1.5-9B-AD-Q4_K-IQ4_XS.gguf",
        size_bytes=5611873024,
        sampling_overrides={"temperature": 0.7, "top_k": 20, "repeat_penalty": 1.05},
        draft_repo="audreyt/Ornith-1.5-9B-DFlash-GGUF",
        draft_filename="ornith1.5-9b-dflash-bf16-projection-Q4_K_M.gguf",
        draft_size_bytes=765959552,
        spec_type=SPEC_TYPE_DRAFT_DFLASH,
        reasoning="on",
    ),
    "pro": ModelSpec(
        key="pro",
        label="Qwen3.8-27B",
        repo="sdkyuan/qwen3.8-27B-qat-q2_0-gguf",
        filename="qwen38-27b-qat-q2_0.gguf",
        size_bytes=8759266208,
        sampling_overrides={"temperature": 0.7, "top_k": 20, "repeat_penalty": 1.05},
        draft_repo="incoai/Qwen3.8-27B-DFlash2-GGUF",
        draft_filename="Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
        draft_size_bytes=1143006816,
        spec_type=SPEC_TYPE_DRAFT_DFLASH,
        reasoning="on",
        reasoning_effort="medium",
    ),
}

DEFAULT_MODEL_KEY = "lite"
LLAMA_SERVER_VERSION = "b10621"
LLAMA_SERVER_BASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download"
DEFAULT_PORT = 2011
DEFAULT_HOST = "127.0.0.1"


# Official llama.cpp prebuilt assets for each (platform, backend) pair.
# macOS binaries ship with Metal enabled; there is no separate Metal asset.
_WINDOWS_ASSETS = {
    Backend.CUDA: f"llama-{LLAMA_SERVER_VERSION}-bin-win-cuda-12.4-x64.zip",
    Backend.VULKAN: f"llama-{LLAMA_SERVER_VERSION}-bin-win-vulkan-x64.zip",
}
_LINUX_ASSETS = {
    Backend.VULKAN: f"llama-{LLAMA_SERVER_VERSION}-bin-ubuntu-vulkan-x64.tar.gz",
}


def _server_download_url(
    platform: str | None = None,
    backend: "Backend | None" = None,
) -> str:
    import platform as _platform
    import sys

    platform = platform or sys.platform

    if platform == "darwin":
        # Metal is compiled into the official macOS binaries.
        arch = "arm64" if _platform.machine() == "arm64" else "x64"
        asset = f"llama-{LLAMA_SERVER_VERSION}-bin-macos-{arch}.tar.gz"
        return f"{LLAMA_SERVER_BASE_URL}/{LLAMA_SERVER_VERSION}/{asset}"

    if platform == "win32":
        asset = _WINDOWS_ASSETS.get(backend, f"llama-{LLAMA_SERVER_VERSION}-bin-win-cpu-x64.zip")
        return f"{LLAMA_SERVER_BASE_URL}/{LLAMA_SERVER_VERSION}/{asset}"

    # Linux: Vulkan has a prebuilt; CUDA does not, so it falls back to CPU.
    asset = _LINUX_ASSETS.get(backend, f"llama-{LLAMA_SERVER_VERSION}-bin-ubuntu-x64.tar.gz")
    return f"{LLAMA_SERVER_BASE_URL}/{LLAMA_SERVER_VERSION}/{asset}"




CONFIG_DIR = Path.home() / ".golden-agent"
CONFIG_FILE = CONFIG_DIR / "config.json"
INFERENCE_FILE = CONFIG_DIR / "inference.json"


def load_config() -> dict:
    if CONFIG_FILE.is_file():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_config(data: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def default_inference() -> dict:
    """Per-model llama-server argument overrides.

    ``spec_draft_n_max`` caps the number of draft tokens per speculative step
    (DFlash2 recommends 7). ``reasoning_effort`` overrides the chat-template
    reasoning budget. ``extra_args`` is appended verbatim to the llama-server
    command line for full control.
    """
    out: dict = {}
    for key, spec in MODELS.items():
        out[key] = {
            "spec_draft_n_max": 7 if spec.draft_filename else None,
            "reasoning_effort": spec.reasoning_effort,
            "extra_args": [],
        }
    return out


def load_inference() -> dict:
    """Return the inference overrides, creating ``inference.json`` once if absent.

    The file is never overwritten once it exists, so user edits survive across
    golden-agent runs/setups.
    """
    if INFERENCE_FILE.is_file():
        try:
            return json.loads(INFERENCE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    data = default_inference()
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        INFERENCE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass
    return data
