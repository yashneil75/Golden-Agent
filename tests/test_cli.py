from io import StringIO

from rich.console import Console

from golden_agent import cli
from golden_agent.render import StreamingReply, split_thinking


class FakeReply:
    def __init__(self):
        self.events: list[tuple[bool, str]] = []
        self.finished = False

    def add(self, in_thinking: bool, text: str) -> None:
        self.events.append((in_thinking, text))

    def finish(self) -> None:
        self.finished = True


def make_sink_console(force_terminal: bool = False, height: int = 25):
    sink = StringIO()
    console = Console(
        file=sink,
        width=80,
        height=height,
        force_terminal=force_terminal,
        color_system="truecolor" if force_terminal else "auto",
    )
    return sink, console


def merge_events(events):
    merged = []
    for in_thinking, text in events:
        if merged and merged[-1][0] == in_thinking:
            merged[-1] = (in_thinking, merged[-1][1] + text)
        else:
            merged.append((in_thinking, text))
    return merged


def test_thinking_and_answer_routed_by_split():
    reply = FakeReply()

    for in_thinking, text in split_thinking(["Let me think.", "</think>", "\n\nThe answer is 4."]):
        reply.add(in_thinking, text)
    reply.finish()

    assert merge_events(reply.events) == [
        (True, "Let me think."),
        (False, "The answer is 4."),
    ]
    assert reply.finished is True


def test_explicit_open_tag_routes_to_thinking():
    reply = FakeReply()

    tokens = ["<think>", "pondering...", "</think>", "\n\nAnswer text."]
    for in_thinking, text in split_thinking(tokens):
        reply.add(in_thinking, text)
    reply.finish()

    assert merge_events(reply.events) == [
        (True, "pondering..."),
        (False, "Answer text."),
    ]


def test_tag_split_across_chunk_boundaries():
    reply = FakeReply()

    for in_thinking, text in split_thinking(["reasoning text</th", "ink>\n\nFinal: 42"]):
        reply.add(in_thinking, text)
    reply.finish()

    thinking = "".join(text for flag, text in reply.events if flag)
    answer = "".join(text for flag, text in reply.events if not flag)
    assert thinking == "reasoning text"
    assert "</th" not in answer
    assert "Final: 42" in answer


def test_unclosed_stream_re_renders_as_answer():
    sink, console = make_sink_console()
    reply = StreamingReply(console)

    for in_thinking, text in split_thinking(["Never closed the ", "think block."]):
        reply.add(in_thinking, text)
    reply.finish()

    assert reply.text == "Never closed the think block."
    assert reply.closed_think_block is False


def test_closed_stream_keeps_answer_separate():
    sink, console = make_sink_console()
    reply = StreamingReply(console)

    for in_thinking, text in split_thinking(["hmm", "</think>", "\n\nvisible"]):
        reply.add(in_thinking, text)
    reply.finish()

    assert reply.text == "visible"
    assert reply.closed_think_block is True


def test_answer_chunks_arrive_progressively():
    reply = FakeReply()

    tokens = ["thinking", "</think>", "\n\nHello ", "big ", "world"]
    for in_thinking, text in split_thinking(tokens):
        reply.add(in_thinking, text)
    reply.finish()

    answer_events = [(flag, text) for flag, text in reply.events if not flag]
    assert "".join(text for _, text in answer_events) == "Hello big world"
    assert len(answer_events) >= 2
    assert reply.finished is True


def test_turn_separator_prints_blank_line():
    sink, console = make_sink_console(force_terminal=True)

    cli.turn_separator(console)

    assert sink.getvalue() == "\n"


def test_llama_params_configure_server():
    from golden_agent.config import MODELS

    spec = MODELS["lite"]
    assert spec.n_ctx is None
    assert spec.tool_format == "qwen3_xml"


def test_split_thinking_separates_reasoning_from_answer():
    events = list(split_thinking(["Let me think.", "</think>", "\n\nThe ", "answer."]))

    assert merge_events(events) == [
        (True, "Let me think."),
        (False, "The answer."),
    ]


def test_split_thinking_strips_leading_newlines_after_close():
    events = list(split_thinking(["reason", "</think>", "\n\n\nAnswer"]))

    assert merge_events(events) == [(True, "reason"), (False, "Answer")]


def test_split_thinking_holds_partial_tag_across_tokens():
    events = list(split_thinking(["reason</th", "ink>", "Answer"]))

    assert merge_events(events) == [(True, "reason"), (False, "Answer")]


def test_live_renderer_streams_markdown_answer():
    sink, console = make_sink_console()
    reply = StreamingReply(console)

    reply.add(False, "# Heading")
    reply.add(False, "\n\nBody text")
    reply.finish()

    output = sink.getvalue()
    assert "Heading" in output
    assert "Body text" in output


def test_reply_streams_thinking_greyed_then_answer():
    sink, console = make_sink_console(force_terminal=True)
    reply = StreamingReply(console)

    reply.add(True, "quiet reasoning")
    reply.add(False, "\n\nanswer")
    reply.finish()

    output = sink.getvalue()
    assert "quiet reasoning" in output
    assert "thought for" not in output
    assert "answer" in output


def test_long_answer_prints_each_line_exactly_once():
    sink, console = make_sink_console(height=10)
    reply = StreamingReply(console)

    for i in range(40):
        reply.add(False, f"line {i} of a long answer\n")
        if i % 5 == 4:
            reply.add(False, "\n")
    reply.finish()

    output = sink.getvalue()
    for i in (0, 5, 20, 39):
        assert output.count(f"line {i}") == 1, f"line {i} printed {output.count(f'line {i}')} times"


def test_reply_keeps_full_answer_when_thinking_never_closed():
    sink, console = make_sink_console(force_terminal=True)
    reply = StreamingReply(console)

    reply.add(True, "this is actually the answer")
    reply.finish()

    output = sink.getvalue()
    assert "this is actually the answer" in output


def test_reply_ignores_empty_chunks():
    sink, console = make_sink_console()
    reply = StreamingReply(console)

    reply.add(False, "")
    reply.finish()

    assert sink.getvalue() == ""


def test_streaming_tokens_render_incrementally():
    sink, console = make_sink_console()
    reply = StreamingReply(console)

    for i in range(50):
        reply.add(False, f"chunk {i}\n")
    reply.finish()

    output = sink.getvalue()
    assert "chunk 0" in output
    assert "chunk 49" in output
    assert output.count("chunk 25") == 1
