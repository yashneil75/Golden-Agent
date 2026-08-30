from pathlib import Path


def make_sparse_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.seek(size - 1)
        handle.write(b"\x00")


class FakeResponse:
    def __init__(self, chunks: list[bytes], status: int = 206):
        self._chunks = list(chunks)
        self.status = status

    def read(self, size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class FakeOpener:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str]) -> FakeResponse:
        self.calls.append((url, dict(headers)))
        if not self.responses:
            raise AssertionError("unexpected extra network call")
        return self.responses.pop(0)


def chunked_response(payload: bytes) -> tuple[FakeOpener, int]:
    response = FakeResponse([payload[i : i + 4096] for i in range(0, len(payload), 4096)])
    return FakeOpener([response]), len(payload)


def test_model_url_targets_huggingface_resolve() -> None:
    from golden_agent.download import MODEL_FILENAME, MODEL_REPO, model_url

    url = model_url()
    assert url == f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILENAME}"
    assert MODEL_FILENAME == "Ornith-1.5-9B-AD-Q4_K-IQ4_XS.gguf"


def test_cache_dir_windows_uses_localappdata() -> None:
    from golden_agent.download import cache_models_dir

    result = cache_models_dir(
        platform="win32", env={"LOCALAPPDATA": "C:/Users/someone/AppData/Local"}
    )
    assert result == Path("C:/Users/someone/AppData/Local/golden-agent/models")


def test_cache_dir_linux_uses_xdg_cache_home() -> None:
    from golden_agent.download import cache_models_dir

    result = cache_models_dir(platform="linux", env={"XDG_CACHE_HOME": "/home/someone/cache"})
    assert result == Path("/home/someone/cache/golden-agent/models")


def test_cache_dir_darwin_uses_library_caches(tmp_path: Path) -> None:
    from golden_agent.download import cache_models_dir

    result = cache_models_dir(platform="darwin", env={"HOME": str(tmp_path)})
    assert result == tmp_path / "Library" / "Caches" / "golden-agent" / "models"


def test_skips_download_when_cached_file_has_expected_size(tmp_path: Path) -> None:
    from golden_agent.download import MODEL_FILENAME, MODEL_SIZE_BYTES, ensure_model_downloaded

    cached = tmp_path / MODEL_FILENAME
    make_sparse_file(cached, MODEL_SIZE_BYTES)

    def boom(url: str, headers: dict[str, str]):
        raise AssertionError("network must not be touched")

    result = ensure_model_downloaded(tmp_path, opener=boom)
    assert result == cached


def test_model_cached_true_when_file_has_expected_size(tmp_path: Path) -> None:
    from golden_agent.download import MODEL_FILENAME, MODEL_SIZE_BYTES, model_cached

    make_sparse_file(tmp_path / MODEL_FILENAME, MODEL_SIZE_BYTES)
    assert model_cached(tmp_path) is True


def test_model_cached_false_when_missing_or_wrong_size(tmp_path: Path) -> None:
    from golden_agent.download import MODEL_FILENAME, MODEL_SIZE_BYTES, model_cached

    assert model_cached(tmp_path) is False

    make_sparse_file(tmp_path / MODEL_FILENAME, MODEL_SIZE_BYTES - 1)
    assert model_cached(tmp_path) is False


def test_redownloads_when_cached_size_mismatched(tmp_path: Path) -> None:
    from golden_agent.download import MODEL_FILENAME, ensure_model_downloaded

    stale_size = 1024
    make_sparse_file(tmp_path / MODEL_FILENAME, stale_size)
    opener, payload_len = chunked_response(b"z" * 2048)

    result = ensure_model_downloaded(tmp_path, opener=opener, expected_size=payload_len)

    assert result.stat().st_size == payload_len
    assert len(opener.calls) == 1
    url, headers = opener.calls[0]
    assert url.endswith(MODEL_FILENAME)
    assert "Range" not in headers


def test_resumes_partial_download_with_range_request(tmp_path: Path) -> None:
    from golden_agent.download import MODEL_FILENAME, ensure_model_downloaded

    prefix, remainder = b"abcde", b"fghij"
    (tmp_path / f"{MODEL_FILENAME}.part").write_bytes(prefix)

    opener = FakeOpener([FakeResponse([remainder], status=206)])

    result = ensure_model_downloaded(
        tmp_path,
        opener=opener,
        expected_size=len(prefix) + len(remainder),
    )

    _, headers = opener.calls[0]
    assert headers["Range"] == f"bytes={len(prefix)}-"
    assert result.read_bytes() == prefix + remainder
    assert not (tmp_path / f"{MODEL_FILENAME}.part").exists()


def test_restarts_from_scratch_when_server_ignores_range(tmp_path: Path) -> None:
    from golden_agent.download import ensure_model_downloaded

    (tmp_path / "fake.gguf.part").write_bytes(b"stale")
    opener = FakeOpener([FakeResponse([b"fresh-body"], status=200)])

    result = ensure_model_downloaded(
        tmp_path,
        opener=opener,
        filename="fake.gguf",
        expected_size=len(b"fresh-body"),
    )

    assert result.read_bytes() == b"fresh-body"
    _, headers = opener.calls[0]
    assert headers["Range"] == "bytes=5-"


def test_raises_and_cleans_part_when_final_size_is_wrong(tmp_path: Path) -> None:
    import pytest

    from golden_agent.download import ensure_model_downloaded

    opener = FakeOpener([FakeResponse([b"too-short"], status=200)])

    with pytest.raises(RuntimeError, match="size"):
        ensure_model_downloaded(
            tmp_path,
            opener=opener,
            filename="fake.gguf",
            expected_size=999,
        )
    assert not (tmp_path / "fake.gguf.part").exists()


def test_progress_callback_reports_cumulative_bytes(tmp_path: Path) -> None:
    from golden_agent.download import ensure_model_downloaded

    opener = FakeOpener([FakeResponse([b"a" * 10, b"b" * 5], status=206)])
    observed: list[tuple[int, int]] = []

    ensure_model_downloaded(
        tmp_path,
        opener=opener,
        filename="fake.gguf",
        expected_size=15,
        progress=lambda done, total: observed.append((done, total)),
    )

    assert observed[-1] == (15, 15)


def test_none_dest_defaults_to_cache_dir(monkeypatch, tmp_path: Path) -> None:
    from golden_agent import download as dl

    cache = tmp_path / "cache-models"
    make_sparse_file(cache / dl.MODEL_FILENAME, dl.MODEL_SIZE_BYTES)
    monkeypatch.setattr(dl, "cache_models_dir", lambda *a, **k: cache)

    def boom(url: str, headers: dict[str, str]):
        raise AssertionError("network must not be touched")

    result = dl.ensure_model_downloaded(None, opener=boom)
    assert result == cache / dl.MODEL_FILENAME


class MidStreamFailResponse(FakeResponse):
    def read(self, size: int = -1) -> bytes:
        if not self._chunks:
            raise ConnectionResetError("connection reset by peer")
        return self._chunks.pop(0)


class ResumableServerFake:
    def __init__(self, payload: bytes, fail_first_attempt_after: int | None = None):
        self.payload = payload
        self.fail_budget = fail_first_attempt_after
        self.failed_once = False
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str]) -> FakeResponse:
        self.calls.append((url, dict(headers)))
        start = int(headers["Range"].split("=")[1].rstrip("-")) if "Range" in headers else 0
        remaining = self.payload[start:]
        chunks = [remaining[i : i + 16] for i in range(0, len(remaining), 16)]
        if self.fail_budget is not None and not self.failed_once:
            self.failed_once = True
            budget_chunks: list[bytes] = []
            taken = 0
            for chunk in chunks:
                if taken >= self.fail_budget:
                    break
                piece = chunk[: self.fail_budget - taken]
                budget_chunks.append(piece)
                taken += len(piece)
            return MidStreamFailResponse(budget_chunks)
        return FakeResponse(chunks, status=206 if start else 200)


def test_retries_mid_stream_failure_and_resumes_from_offset(tmp_path: Path) -> None:
    from golden_agent.download import MODEL_FILENAME, ensure_model_downloaded

    payload = b"0123456789" * 10
    server = ResumableServerFake(payload, fail_first_attempt_after=40)

    result = ensure_model_downloaded(
        tmp_path,
        opener=server,
        filename=MODEL_FILENAME,
        expected_size=len(payload),
        sleeper=lambda seconds: None,
    )

    assert result.read_bytes() == payload
    assert len(server.calls) == 2
    assert server.calls[1][1]["Range"] == "bytes=40-"


def test_backs_off_exponentially_between_retries(tmp_path: Path) -> None:
    import pytest

    from golden_agent.download import ensure_model_downloaded

    def always_reset(url: str, headers: dict[str, str]):
        raise ConnectionResetError("down")

    sleeps: list[float] = []

    with pytest.raises(RuntimeError, match="attempts"):
        ensure_model_downloaded(
            tmp_path,
            opener=always_reset,
            filename="fake.gguf",
            expected_size=100,
            max_attempts=4,
            sleeper=sleeps.append,
        )

    assert sleeps == [1.0, 2.0, 4.0]


def test_keeps_partial_file_when_retries_are_exhausted(tmp_path: Path) -> None:
    import pytest

    from golden_agent.download import ensure_model_downloaded

    partial = tmp_path / "fake.gguf.part"
    partial.write_bytes(b"keep-me")

    def always_reset(url: str, headers: dict[str, str]):
        raise ConnectionResetError("down")

    with pytest.raises(RuntimeError, match="attempts"):
        ensure_model_downloaded(
            tmp_path,
            opener=always_reset,
            filename="fake.gguf",
            expected_size=100,
            max_attempts=2,
            sleeper=lambda seconds: None,
        )

    assert partial.read_bytes() == b"keep-me"


def test_client_errors_fail_fast_without_retry(tmp_path: Path) -> None:
    import urllib.error

    import pytest

    from golden_agent.download import ensure_model_downloaded

    forbidden = urllib.error.HTTPError("url", 403, "forbidden", None, None)

    def denied(url: str, headers: dict[str, str]):
        raise forbidden

    sleeps: list[float] = []
    calls: list[int] = []

    def counting_opener(url: str, headers: dict[str, str]):
        calls.append(1)
        return denied(url, headers)

    with pytest.raises(urllib.error.HTTPError):
        ensure_model_downloaded(
            tmp_path,
            opener=counting_opener,
            filename="fake.gguf",
            expected_size=100,
            max_attempts=5,
            sleeper=sleeps.append,
        )

    assert len(calls) == 1
    assert sleeps == []


def test_stale_part_larger_than_expected_restarts_from_scratch(tmp_path: Path) -> None:
    from golden_agent.download import MODEL_FILENAME, ensure_model_downloaded

    payload = b"fresh-data"
    (tmp_path / f"{MODEL_FILENAME}.part").write_bytes(b"x" * 999)
    server = ResumableServerFake(payload)

    result = ensure_model_downloaded(
        tmp_path,
        opener=server,
        filename=MODEL_FILENAME,
        expected_size=len(payload),
        sleeper=lambda seconds: None,
    )

    _, headers = server.calls[0]
    assert "Range" not in headers
    assert result.read_bytes() == payload
