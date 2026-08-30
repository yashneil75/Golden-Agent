import http.client
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

from golden_agent.config import DEFAULT_MODEL_KEY, MODELS

DEFAULT_MODEL = MODELS[DEFAULT_MODEL_KEY]
MODEL_REPO = DEFAULT_MODEL.repo
MODEL_FILENAME = DEFAULT_MODEL.filename
MODEL_SIZE_BYTES = DEFAULT_MODEL.size_bytes
CHUNK_SIZE = 1024 * 1024
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 30.0

OpenFn = Callable[[str, dict[str, str]], object]
ProgressFn = Callable[[int, int], None]


class ModelDownloadError(RuntimeError):
    pass


def model_url(repo: str = MODEL_REPO, filename: str = MODEL_FILENAME) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def cache_models_dir(
    platform: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    platform = platform or sys.platform
    env = env or os.environ

    if platform == "win32":
        base = Path(env.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    elif platform == "darwin":
        base = Path(env.get("HOME") or Path.home()) / "Library" / "Caches"
    else:
        base = Path(env.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))

    return base / "golden-agent" / "models"


def _http_opener(url: str, headers: dict[str, str]):
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request)


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or error.code >= 500
    return isinstance(error, (OSError, http.client.HTTPException))


def _stream_attempt(
    opener: OpenFn,
    url: str,
    part_path: Path,
    dest_dir: Path,
    expected_size: int,
    progress: ProgressFn | None,
    chunk_size: int,
) -> int:
    offset = part_path.stat().st_size if part_path.exists() else 0
    if offset > expected_size:
        offset = 0

    headers: dict[str, str] = {"Range": f"bytes={offset}-"} if offset else {}
    dest_dir.mkdir(parents=True, exist_ok=True)
    with opener(url, headers) as response:
        if getattr(response, "status", 200) != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        written = offset
        with open(part_path, mode) as sink:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                sink.write(chunk)
                written += len(chunk)
                if progress:
                    progress(written, expected_size)
    if progress:
        progress(written, expected_size)

    if written != expected_size:
        part_path.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"size mismatch for {part_path.name}: downloaded {written} bytes, "
            f"expected {expected_size}"
        )
    return written


def model_cached(
    dest_dir: Path | str | None = None,
    *,
    filename: str = MODEL_FILENAME,
    expected_size: int = MODEL_SIZE_BYTES,
) -> bool:
    dest = Path(dest_dir) if dest_dir is not None else cache_models_dir()
    final_path = dest / filename
    return final_path.is_file() and final_path.stat().st_size == expected_size


def ensure_model_downloaded(
    dest_dir: Path | str | None = None,
    *,
    opener: OpenFn | None = None,
    repo: str = MODEL_REPO,
    filename: str = MODEL_FILENAME,
    expected_size: int = MODEL_SIZE_BYTES,
    progress: ProgressFn | None = None,
    chunk_size: int = CHUNK_SIZE,
    max_attempts: int = MAX_ATTEMPTS,
    sleeper: Callable[[float], None] = time.sleep,
) -> Path:
    dest = Path(dest_dir) if dest_dir is not None else cache_models_dir()
    final_path = dest / filename
    part_path = dest / f"{filename}.part"

    if model_cached(dest, filename=filename, expected_size=expected_size):
        return dest / filename

    opener = opener or _http_opener
    url = model_url(repo=repo, filename=filename)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            _stream_attempt(opener, url, part_path, dest, expected_size, progress, chunk_size)
            break
        except Exception as err:
            if not _is_retryable(err):
                raise
            last_error = err
            if attempt < max_attempts:
                delay = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
                sleeper(delay)
    else:
        raise ModelDownloadError(
            f"download of {filename} failed after {max_attempts} attempts; partial data kept "
            f"at {part_path} for resume ({last_error})"
        ) from last_error

    os.replace(part_path, final_path)
    return final_path


def ensure_draft_model_downloaded(
    dest_dir: Path | str | None = None,
    *,
    repo: str,
    filename: str,
    expected_size: int,
    progress: ProgressFn | None = None,
) -> Path:
    """Download a draft model GGUF if not already cached."""
    return ensure_model_downloaded(
        dest_dir,
        repo=repo,
        filename=filename,
        expected_size=expected_size,
        progress=progress,
    )



