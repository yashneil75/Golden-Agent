from unittest.mock import MagicMock, patch


def test_sampling_params_exposed_for_chat_completion() -> None:
    from golden_agent.config import MODELS

    spec = MODELS["lite"]
    assert spec.tool_format == "qwen3_xml"
    assert spec.spec_type == "draft-dflash"
    assert spec.draft_repo == "audreyt/Ornith-1.5-9B-DFlash-GGUF"


def test_tiny_model_registered_with_lfm25_config() -> None:
    from golden_agent.config import MODELS

    tiny = MODELS["tiny"]
    assert tiny.label == "LFM2.5-2.6B"
    assert tiny.repo == "LiquidAI/LFM2.5-2.6B-GGUF"
    assert tiny.filename == "LFM2.5-2.6B-QAD-Q4_0.gguf"
    assert tiny.size_bytes == 1593894944
    assert tiny.n_ctx == 128000
    assert tiny.tool_format == "lfm2"


def test_default_tool_format_is_qwen3_xml() -> None:
    from golden_agent.config import MODELS

    assert MODELS["lite"].tool_format == "qwen3_xml"
    assert MODELS["pro"].tool_format == "qwen3_xml"


def test_count_tokens_estimates_correctly() -> None:
    from golden_agent.model import count_tokens

    assert count_tokens("") == 1
    assert count_tokens("hi") == 1
    assert count_tokens("hello world test") == 4


def test_set_model_switches_selected() -> None:
    from golden_agent.model import selected_model, set_model

    set_model("lite")
    assert selected_model().key == "lite"

    set_model("pro")
    assert selected_model().key == "pro"

    set_model("lite")
    assert selected_model().key == "lite"


def test_set_model_raises_on_unknown_key() -> None:
    import pytest

    from golden_agent.model import set_model

    with pytest.raises(KeyError, match="unknown model"):
        set_model("nonexistent")


class FakeHTTPResponse:
    def __init__(self, chunks: list[str]):
        self._lines = list(chunks)
        self.status_code = 200

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return self._lines


class FakeHTTPStream:
    def __init__(self, lines: list[str]):
        self._lines = lines
        self.response = FakeHTTPResponse(lines)

    def __enter__(self):
        return self.response

    def __exit__(self, *args):
        return False


def test_stream_chat_yields_parsed_chunks() -> None:
    from golden_agent.model import stream_chat

    chunks = [
        'data: {"choices":[{"delta":{"content":"hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        "data: [DONE]",
    ]

    mock_client = MagicMock()
    mock_client.stream.return_value = FakeHTTPStream(chunks)
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("golden_agent.model.httpx.Client", return_value=mock_client):
        with patch("golden_agent.model.server_url", return_value="http://127.0.0.1:8080"):
            from golden_agent.model import set_model

            set_model("lite")
            result = list(stream_chat([{"role": "user", "content": "hi"}]))

    assert len(result) == 2
    assert result[0]["choices"][0]["delta"]["content"] == "hel"
    assert result[1]["choices"][0]["delta"]["content"] == "lo"


def test_call_model_stream_yields_content_pieces() -> None:
    from golden_agent.model import call_model_stream

    chunks = [
        'data: {"choices":[{"delta":{"content":"hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        "data: [DONE]",
    ]

    mock_client = MagicMock()
    mock_client.stream.return_value = FakeHTTPStream(chunks)
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("golden_agent.model.httpx.Client", return_value=mock_client):
        with patch("golden_agent.model.server_url", return_value="http://127.0.0.1:8080"):
            from golden_agent.model import set_model

            set_model("lite")
            out = list(call_model_stream("hi", system="sys"))

    assert out == ["hel", "lo"]
