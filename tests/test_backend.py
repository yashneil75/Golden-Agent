from unittest.mock import MagicMock, patch

from golden_agent.backend import (
    _build_server_args,
    _wait_for_server,
    stop_server,
)
from golden_agent.config import Backend


def test_build_server_args_cpu_no_ngl() -> None:
    from golden_agent.config import ModelSpec

    spec = ModelSpec(
        key="test",
        label="Test",
        repo="repo/model",
        filename="model.gguf",
        size_bytes=1000,
    )
    args = _build_server_args(spec, "/path/to/model.gguf", backend=Backend.CPU)

    assert "-ngl" not in args
    assert "-fa" in args


def test_build_server_args_with_ctx() -> None:
    from golden_agent.config import ModelSpec

    spec = ModelSpec(
        key="test",
        label="Test",
        repo="repo/model",
        filename="model.gguf",
        size_bytes=1000,
        n_ctx=128000,
    )
    args = _build_server_args(spec, "/path/to/model.gguf")

    assert "-c" in args
    assert "128000" in args


def test_build_server_args_with_draft() -> None:
    from golden_agent.config import ModelSpec

    spec = ModelSpec(
        key="test",
        label="Test",
        repo="repo/model",
        filename="model.gguf",
        size_bytes=1000,
        spec_type="draft-dflash",
    )
    args = _build_server_args(spec, "/path/to/model.gguf", draft_path="/path/to/draft.gguf")

    assert "-md" in args
    assert "/path/to/draft.gguf" in args
    assert "--spec-type" in args
    assert "draft-dflash" in args


def test_build_server_args_no_draft_when_none() -> None:
    from golden_agent.config import ModelSpec

    spec = ModelSpec(
        key="test",
        label="Test",
        repo="repo/model",
        filename="model.gguf",
        size_bytes=1000,
        spec_type="none",
    )
    args = _build_server_args(spec, "/path/to/model.gguf")

    assert "-md" not in args
    assert "--spec-type" not in args


def test_build_server_args_passes_reasoning_on() -> None:
    from golden_agent.config import ModelSpec

    spec = ModelSpec(
        key="test",
        label="Test",
        repo="repo/model",
        filename="model.gguf",
        size_bytes=1000,
        reasoning="on",
    )
    args = _build_server_args(spec, "/path/to/model.gguf")

    assert "--reasoning" in args
    assert args[args.index("--reasoning") + 1] == "on"


def test_build_server_args_reasoning_requires_jinja_and_inline_format() -> None:
    from golden_agent.config import ModelSpec

    spec = ModelSpec(
        key="test",
        label="Test",
        repo="repo/model",
        filename="model.gguf",
        size_bytes=1000,
        reasoning="on",
    )
    args = _build_server_args(spec, "/path/to/model.gguf")

    # llama-server only honors reasoning flags under the Jinja template,
    # and format "none" keeps think tags inline for client-side rendering.
    assert "--jinja" in args
    assert "--reasoning-format" in args
    assert args[args.index("--reasoning-format") + 1] == "none"


def test_build_server_args_omits_reasoning_when_auto() -> None:
    from golden_agent.config import ModelSpec

    spec = ModelSpec(
        key="test",
        label="Test",
        repo="repo/model",
        filename="model.gguf",
        size_bytes=1000,
    )
    args = _build_server_args(spec, "/path/to/model.gguf")

    assert "--reasoning" not in args
    assert "--reasoning-effort" not in args
    assert "--jinja" not in args


def test_build_server_args_passes_reasoning_effort() -> None:
    from golden_agent.config import ModelSpec

    spec = ModelSpec(
        key="test",
        label="Test",
        repo="repo/model",
        filename="model.gguf",
        size_bytes=1000,
        reasoning="on",
        reasoning_effort="medium",
    )
    args = _build_server_args(spec, "/path/to/model.gguf")

    assert "--reasoning" in args
    assert "--reasoning-effort" in args
    assert args[args.index("--reasoning-effort") + 1] == "medium"


def test_build_server_args_omits_reasoning_effort_when_none() -> None:
    from golden_agent.config import ModelSpec

    spec = ModelSpec(
        key="test",
        label="Test",
        repo="repo/model",
        filename="model.gguf",
        size_bytes=1000,
        reasoning="on",
    )
    args = _build_server_args(spec, "/path/to/model.gguf")

    assert "--reasoning" in args
    assert "--reasoning-effort" not in args


def test_build_server_args_with_sampling_overrides() -> None:
    from golden_agent.config import ModelSpec

    spec = ModelSpec(
        key="test",
        label="Test",
        repo="repo/model",
        filename="model.gguf",
        size_bytes=1000,
        sampling_overrides={"temperature": 0.5, "top_p": 0.9},
    )
    args = _build_server_args(spec, "/path/to/model.gguf")

    assert "--temp" in args
    assert "0.5" in args
    assert "--top-p" in args
    assert "0.9" in args


def test_build_server_args_custom_port() -> None:
    from golden_agent.config import ModelSpec

    spec = ModelSpec(
        key="test",
        label="Test",
        repo="repo/model",
        filename="model.gguf",
        size_bytes=1000,
    )
    args = _build_server_args(spec, "/path/to/model.gguf", port=9090, host="0.0.0.0")

    assert "--port" in args
    assert "9090" in args
    assert "--host" in args
    assert "0.0.0.0" in args


def test_wait_for_server_returns_true_on_200() -> None:
    mock_conn = MagicMock()
    mock_response = MagicMock()
    mock_response.status = 200
    mock_conn.getresponse.return_value = mock_response

    with patch("http.client.HTTPConnection", return_value=mock_conn):
        assert _wait_for_server("127.0.0.1", 8080, timeout=1.0) is True


def test_wait_for_server_returns_false_on_timeout() -> None:
    with patch("http.client.HTTPConnection", side_effect=OSError("refused")):
        assert _wait_for_server("127.0.0.1", 8080, timeout=0.1) is False


def test_stop_server_kills_process() -> None:
    import golden_agent.backend as be

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    be._hookpoint_process = mock_proc
    be._server_pid = 12345
    be._server_port = 2011

    stop_server()

    mock_proc.terminate.assert_called_once()
    assert be._hookpoint_process is None
    assert be._server_pid is None
    assert be._server_port is None


def test_stop_server_handles_already_stopped() -> None:
    import golden_agent.backend as be

    be._hookpoint_process = None
    be._server_pid = None
    be._server_port = None
    stop_server()
