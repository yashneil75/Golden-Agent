import json
from io import StringIO

from rich.console import Console

from golden_agent.agent import extract_tool_calls, render_tool_block, suppress_tool_blocks


def make_sink_console():
    sink = StringIO()
    return sink, Console(file=sink, width=80, force_terminal=True, highlight=False)


SINGLE_CALL = """Let me search for that.

<tool_call>
<function=web_search>
<parameter=query>
qwen3 xml tool call parser
</parameter>
<parameter=max_results>
6
</parameter>
</function>
</tool_call>"""

TWO_CALLS = """<tool_call>
<function=web_search>
<parameter=query>first query</parameter>
</function>
</tool_call>
middle text
<tool_call>
<function=web_fetch>
<parameter=url>
https://example.com
</parameter>
</function>
</tool_call>"""

LFM_TAGGED = (
    "Checking now.\n<|tool_call_start|>"
    '[web_search(query="lfm pythonic calls", max_results=6)]'
    "<|tool_call_end|>\nDone."
)
LFM_PARALLEL = (
    '<|tool_call_start|>[web_search(query="first"), web_search(query="second")]<|tool_call_end|>'
)
LFM_BARE = '[read_file(path="src/x.py")]'


def test_extract_returns_name_and_arguments():
    _, calls = extract_tool_calls(SINGLE_CALL)

    assert len(calls) == 1
    assert calls[0]["name"] == "web_search"
    assert json.loads(calls[0]["arguments"]) == {
        "query": "qwen3 xml tool call parser",
        "max_results": "6",
    }


def test_extract_strips_blocks_from_visible_text():
    visible, calls = extract_tool_calls(SINGLE_CALL)

    assert visible == "Let me search for that."
    assert "tool_call" not in visible


def test_extract_handles_multiple_calls_with_text_between():
    visible, calls = extract_tool_calls(TWO_CALLS)

    assert [call["name"] for call in calls] == ["web_search", "web_fetch"]
    assert json.loads(calls[1]["arguments"]) == {"url": "https://example.com"}
    assert visible == "middle text"
    assert "function=" not in visible


def test_extract_returns_empty_for_plain_text():
    visible, calls = extract_tool_calls("Just a normal answer, no tools.")

    assert calls == []
    assert visible == "Just a normal answer, no tools."


def test_extract_lfm2_tagged_pythonic_calls():
    visible, calls = extract_tool_calls(LFM_TAGGED, "lfm2")

    assert len(calls) == 1
    assert calls[0]["name"] == "web_search"
    assert json.loads(calls[0]["arguments"]) == {
        "query": "lfm pythonic calls",
        "max_results": 6,
    }
    assert visible == "Checking now.\n\nDone."
    assert "tool_call_start" not in visible


def test_extract_lfm2_parallel_calls_in_one_list():
    _, calls = extract_tool_calls(LFM_PARALLEL, "lfm2")

    assert [call["name"] for call in calls] == ["web_search", "web_search"]
    assert json.loads(calls[0]["arguments"])["query"] == "first"
    assert json.loads(calls[1]["arguments"])["query"] == "second"


def test_extract_lfm2_bare_list_without_special_tokens():
    visible, calls = extract_tool_calls(LFM_BARE, "lfm2")

    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert json.loads(calls[0]["arguments"]) == {"path": "src/x.py"}
    assert visible == ""


def test_extract_lfm2_ignores_plain_text():
    visible, calls = extract_tool_calls("Just a normal answer, no tools.", "lfm2")

    assert calls == []
    assert visible == "Just a normal answer, no tools."


def test_extract_lfm2_bare_list_inside_thinking_and_prose():
    raw = (
        "<think>need to search first</think>\n\n"
        "Let me look that up.\n\n"
        '[web_search(query="qwen 3.8"), web_search(query="perf")]\n'
    )

    visible, calls = extract_tool_calls(raw, "lfm2")

    assert [call["name"] for call in calls] == ["web_search", "web_search"]
    assert json.loads(calls[0]["arguments"]) == {"query": "qwen 3.8"}
    assert "web_search" not in visible


def test_extract_lfm2_keeps_non_call_bracket_lines():
    visible, calls = extract_tool_calls("See reference [1] for details.", "lfm2")

    assert calls == []
    assert visible == "See reference [1] for details."


def test_extract_lfm2_call_embedded_mid_line_with_trailing_prose():
    raw = 'Sure thing. [web_search(query="qwen 3.8")] Let me check.'

    visible, calls = extract_tool_calls(raw, "lfm2")

    assert len(calls) == 1
    assert calls[0]["name"] == "web_search"
    assert "web_search" not in visible


def test_extract_lfm2_list_wrapped_across_lines():
    raw = '[web_search(query="a"),\n  web_search(query="b")]'

    _, calls = extract_tool_calls(raw, "lfm2")

    assert [json.loads(c["arguments"])["query"] for c in calls] == ["a", "b"]


def test_extract_lfm2_handles_smart_quotes():
    raw = "[web_search(query=\u2018qwen 3.8\u2019)]"

    _, calls = extract_tool_calls(raw, "lfm2")

    assert json.loads(calls[0]["arguments"]) == {"query": "qwen 3.8"}


def test_extract_lfm2_keeps_unknown_function_brackets():
    raw = "The expression [foo(bar=1)] is a python call."

    visible, calls = extract_tool_calls(raw, "lfm2")

    assert calls == []
    assert visible == raw


def test_extract_lfm2_treats_unknown_tool_name_on_own_line_as_attempt():
    raw = 'Let me grab that page.\n[search_web(query="qwen 3.8")]\n'

    visible, calls = extract_tool_calls(raw, "lfm2")

    assert len(calls) == 1
    assert calls[0]["name"] == "search_web"
    assert json.loads(calls[0]["arguments"]) == {"query": "qwen 3.8"}
    assert "search_web(" not in visible


def test_extract_lfm2_mixed_known_and_unknown_names_keeps_both():
    raw = '[web_search(query="a"), search_web(query="b")]'

    _, calls = extract_tool_calls(raw, "lfm2")

    assert [call["name"] for call in calls] == ["web_search", "search_web"]


def test_extract_lfm2_span_with_nested_brackets_in_argument():
    content = "<ul>[Local, Local, Global]</ul> it\\'s fine"
    raw = f"[write_file(path='r.html', content='{content}')]"

    visible, calls = extract_tool_calls(raw, "lfm2")

    assert [call["name"] for call in calls] == ["write_file"]
    expected = "<ul>[Local, Local, Global]</ul> it's fine"
    assert json.loads(calls[0]["arguments"])["content"] == expected
    assert visible == ""


def test_suppress_hides_bare_span_with_nested_brackets():
    pieces = ["ok.\n", "[write_file(path='r.html', content='[a, b] it\\'s')]", "\nend"]

    output = "".join(suppress_tool_blocks(pieces, "lfm2"))

    assert "write_file" not in output
    assert output == "ok.\n\nend"


def test_matching_bracket_ignores_brackets_in_strings():
    from golden_agent.agent import _matching_bracket

    text = "[f(a='x]y', b=\"[z]\")] tail"

    assert _matching_bracket(text, 0) == 20
    assert text[_matching_bracket(text, 0)] == "]"


def test_suppress_hides_complete_blocks_from_stream():
    output = "".join(suppress_tool_blocks([SINGLE_CALL]))

    assert "web_search" not in output
    assert "parameter" not in output
    assert "Let me search for that." in output


def test_suppress_handles_tags_split_across_chunks():
    pieces = [
        "thinking <tool_",
        "call><function=web_search>",
        "<parameter=query>a</param",
        "eter></function></tool_call>",
        "\n\nFinal answer",
    ]

    output = "".join(suppress_tool_blocks(pieces))

    assert output == "thinking \n\nFinal answer"


def test_suppress_keeps_text_after_unterminated_block_hidden():
    pieces = ["before <tool_call><function=x>", " never closed"]

    output = "".join(suppress_tool_blocks(pieces))

    assert output == "before "


def test_suppress_passes_plain_text_through_unchanged():
    pieces = ["hello ", "world", "!"]

    assert "".join(suppress_tool_blocks(pieces)) == "hello world!"


def test_suppress_hides_lfm2_blocks_from_stream():
    output = "".join(suppress_tool_blocks([LFM_TAGGED], "lfm2"))

    assert "web_search" not in output
    assert "tool_call_start" not in output
    assert "Checking now." in output
    assert "Done." in output


def test_suppress_handles_lfm2_tags_split_across_chunks():
    pieces = [
        "before <|tool_",
        'call_start|>[bash(command="ls")]<|tool_c',
        "all_end|>",
        " after",
    ]

    output = "".join(suppress_tool_blocks(pieces, "lfm2"))

    assert output == "before  after"


def test_suppress_hides_bare_lfm2_span_from_stream():
    pieces = ["Let me look.\n", '[web_search(query="a")]', "\nDone."]

    output = "".join(suppress_tool_blocks(pieces, "lfm2"))

    assert output == "Let me look.\n\nDone."


def test_suppress_hides_bare_span_split_across_chunks():
    pieces = ["x\n[web_se", 'arch(query="a")', "]\ny"]

    output = "".join(suppress_tool_blocks(pieces, "lfm2"))

    assert output == "x\n\ny"


def test_suppress_hides_bracket_and_name_in_separate_chunks():
    seen: list[str] = []
    pieces = ["Hello\n", "[", "bash(command=", "'ls -la'", ", timeout=30)", "]"]

    output = "".join(suppress_tool_blocks(pieces, "lfm2", on_call=seen.append))

    assert output == "Hello\n"
    assert seen == ["bash"]


def test_suppress_hides_midline_span_with_known_tool():
    pieces = ["Here: [read_file(path='a.py')]\n"]
    seen: list[str] = []

    output = "".join(suppress_tool_blocks(pieces, "lfm2", on_call=seen.append))

    assert output == "Here: \n"
    assert seen == ["read_file"]


def test_suppress_keeps_prose_bracket_across_chunks():
    pieces = ["see footnote [", "1] and more text here"]

    output = "".join(suppress_tool_blocks(pieces, "lfm2"))

    assert output == "see footnote [1] and more text here"


def test_suppress_reports_detected_calls_to_on_call():
    seen: list[str] = []

    "".join(suppress_tool_blocks([LFM_TAGGED, "\n", LFM_BARE], "lfm2", on_call=seen.append))

    assert seen == ["web_search", "read_file"]


def test_suppress_announces_qwen3_xml_calls():
    seen: list[str] = []

    "".join(suppress_tool_blocks([SINGLE_CALL], on_call=seen.append))

    assert seen == ["web_search"]


def test_suppress_keeps_non_call_bracket_text_visible():
    pieces = ["see reference ", "[1] ", "for details"]

    output = "".join(suppress_tool_blocks(pieces, "lfm2"))
    seen: list[str] = []
    output2 = "".join(suppress_tool_blocks(["see [1] ok"], "lfm2", on_call=seen.append))

    assert output == "see reference [1] for details"
    assert output2 == "see [1] ok"
    assert seen == []


def test_suppress_hides_unknown_name_own_line_attempt():
    seen: list[str] = []

    output = "".join(
        suppress_tool_blocks(['hmm.\n[search_web(query="q")]\nend'], "lfm2", on_call=seen.append)
    )

    assert "search_web" not in output
    assert seen == ["search_web"]


def test_panel_tool_renders_one_block_with_result_status():
    sink, console = make_sink_console()

    render_tool_block(console, "bash", {"command": "ls"}, "a\nb")

    output = sink.getvalue()
    assert "bash" in output
    assert "✓" in output
    assert "2 lines" in output


def test_error_result_is_marked_failed():
    sink, console = make_sink_console()

    render_tool_block(
        console, "write_file", {"path": "x.py", "content": "hi"}, "error: no permission"
    )

    output = sink.getvalue()
    assert "✗" in output
    assert "no permission" in output


def test_line_tool_error_uses_same_failure_marker():
    sink, console = make_sink_console()

    render_tool_block(console, "read_file", {"path": "missing.py"}, "error: not found")

    output = sink.getvalue()
    assert "✗" in output
    assert "not found" in output
    assert "╭" not in output


def test_line_tool_renders_status_line_without_panel_border():
    sink, console = make_sink_console()

    render_tool_block(console, "read_file", {"path": "src/x.py"}, "l1\nl2\nl3")

    output = sink.getvalue()
    assert "src/x.py" in output
    assert "3 lines" in output
    assert "╭" not in output


def test_web_search_reports_result_count():
    sink, console = make_sink_console()
    result = "title one\nsnippet one\n\ntitle two\nsnippet two"

    render_tool_block(console, "web_search", {"query": "qwen tools"}, result)

    output = sink.getvalue()
    assert "2 results" in output
