import ast
import json
import os
import re

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from golden_agent.model import selected_model, stream_chat
from golden_agent.pause import SteeringError, pausable
from golden_agent.render import StreamingReply, console, split_thinking
from golden_agent.tools import TOOL_SCHEMAS, execute_tool

MAX_TOOL_ROUNDS = 130
PREVIEW_LINES = 40

KNOWN_TOOL_NAMES = {schema["function"]["name"] for schema in TOOL_SCHEMAS}

TOOL_MARKERS = {
    "qwen3_xml": ("<tool_call>", "</tool_call>"),
    "lfm2": ("<|tool_call_start|>", "<|tool_call_end|>"),
}

CALL_ANNOUNCEMENTS = {
    "read_file": "Reading file",
    "write_file": "Writing file",
    "edit_file": "Editing file",
    "bash": "Running command",
    "web_search": "Searching the web",
    "web_fetch": "Fetching page",
}


def tool_label(name: str) -> str:
    """Friendly one-liner shown while a tool executes."""
    return CALL_ANNOUNCEMENTS.get(name, "Calling tool")


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>]*)>(?P<body>.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(r"<parameter=([^>]*)>(.*?)</parameter>", re.DOTALL)

_LFM_TAGGED_RE = re.compile(
    r"<\|tool_call_start\|>\s*(?P<body>.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)
_LFM_UNCLOSED_RE = re.compile(r"<\|tool_call_start\|>\s*\[.*\]\s*$", re.DOTALL)
_SMART_QUOTES = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}


def _parse_pythonic_calls(body: str, start_index: int) -> list[dict]:
    """Parse a pythonic call list like [f(a=1), g(b="x")] into tool calls."""
    try:
        node = ast.parse(body.strip(), mode="eval").body
    except SyntaxError:
        return []
    elements = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    calls: list[dict] = []
    for element in elements:
        if not isinstance(element, ast.Call) or not isinstance(element.func, ast.Name):
            continue
        arguments = {}
        for keyword in element.keywords:
            if keyword.arg is None:
                continue
            try:
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
            except Exception:
                arguments[keyword.arg] = ast.unparse(keyword.value)
        calls.append(
            {
                "id": f"call_{start_index + len(calls)}",
                "name": element.func.id,
                "arguments": json.dumps(arguments),
                "inline": True,
            }
        )
    return calls


def _extract_lfm2_calls(text: str) -> tuple[str, list[dict]]:
    """Split LFM2/LFM2.5 pythonic tool-call blocks out of model output.

    Handles the native tagged form, an unclosed tag at stream end, and bare
    pythonic lists some runtimes produce when special tokens are dropped —
    anywhere in the text, on one line or wrapped across lines. Bare spans
    that parse as calls are extracted when they reference a known tool or
    sit on a line of their own (a hallucinated tool name is still an
    attempt: dispatching it returns an error the model can correct from).
    Mid-prose spans with no known name stay as visible text so ordinary
    bracketed prose survives.
    """
    calls: list[dict] = []

    def replace_tagged(match: re.Match) -> str:
        found = _parse_pythonic_calls(match.group("body"), len(calls))
        calls.extend(found)
        return ""

    visible = _LFM_TAGGED_RE.sub(replace_tagged, text)
    visible = _LFM_UNCLOSED_RE.sub(replace_tagged, visible)

    kept: list[str] = []
    pos = 0
    while True:
        match = _BARE_SPAN_CANDIDATE_RE.search(visible, pos)
        if match is None:
            kept.append(visible[pos:])
            break
        close_idx = _matching_bracket(visible, match.start())
        if close_idx == -1:
            kept.append(visible[pos:])
            break

        span = visible[match.start() : close_idx + 1]
        normalized = "".join(_SMART_QUOTES.get(char, char) for char in span)
        found = _parse_pythonic_calls(normalized, len(calls))
        line_start = visible.rfind("\n", 0, match.start()) + 1
        own_line = (
            not visible[line_start : match.start()].strip()
            and not visible[close_idx + 1 :].split("\n", 1)[0].strip()
        )
        if not found or not (any(call["name"] in KNOWN_TOOL_NAMES for call in found) or own_line):
            kept.append(visible[pos : close_idx + 1])
            pos = close_idx + 1
            continue

        calls.extend(found)
        kept.append(visible[pos : match.start()])
        pos = close_idx + 1

    return "".join(kept).strip(), calls


def extract_tool_calls(text: str, tool_format: str = "qwen3_xml") -> tuple[str, list[dict]]:
    """Split tool-call blocks out of model output for the given format.

    Returns (visible text with blocks removed, parsed calls).
    llama-cpp-python renders the tool prompt but never parses generation
    output back into delta.tool_calls, so parsing happens here.
    """
    if tool_format == "lfm2":
        return _extract_lfm2_calls(text)
    return _extract_qwen3_xml_calls(text)


def _extract_qwen3_xml_calls(text: str) -> tuple[str, list[dict]]:
    calls: list[dict] = []

    def replace(match: re.Match) -> str:
        name = match.group("name").strip()
        arguments = {
            key.strip(): value.strip() for key, value in _PARAMETER_RE.findall(match.group("body"))
        }
        calls.append(
            {
                "id": f"call_{len(calls)}",
                "name": name,
                "arguments": json.dumps(arguments),
                "inline": True,
            }
        )
        return ""

    visible = _TOOL_CALL_RE.sub(replace, text)
    return visible.strip(), calls


def _partial_tag_len(buffer: str, tag: str) -> int:
    for size in range(min(len(buffer), len(tag) - 1), 0, -1):
        if buffer.endswith(tag[:size]):
            return size
    return 0


_BARE_HOLDBACK_RE = re.compile(r"\[[A-Za-z_][\w]{0,64}$")


def _split_at_blocks(buffer: str, tool_open: str, tool_close: str) -> tuple[str, str]:
    open_idx = buffer.find(tool_open)
    if open_idx == -1:
        hold = _partial_tag_len(buffer, tool_open)
        split = len(buffer) - hold
        bare_m = _BARE_HOLDBACK_RE.search(buffer[:split])
        if bare_m:
            split = bare_m.start()
        return buffer[:split], buffer[split:]
    close_idx = buffer.find(tool_close, open_idx)
    if close_idx == -1:
        return buffer[:open_idx], buffer[open_idx:]
    head, tail = _split_at_blocks(buffer[close_idx + len(tool_close) :], tool_open, tool_close)
    return buffer[:open_idx] + head, tail


BARE_SPAN_MAX_CHARS = 4000

_BARE_SPAN_START_RE = re.compile(r"\[\s*[A-Za-z_]")
_FUNCTION_NAME_RE = re.compile(r"<function=([^>]*)>")

_BARE_SPAN_CANDIDATE_RE = re.compile(r"\[\s*[A-Za-z_]\w*\(")


def _matching_bracket(text: str, open_idx: int) -> int:
    """Index of the ']' closing the '[' at open_idx, honoring quoted strings.

    Brackets inside string literals don't count toward nesting, so argument
    content like "[Local, Local]" or "it's" can't truncate a span.
    """
    depth = 0
    quote: str | None = None
    i = open_idx
    while i < len(text):
        char = text[i]
        if quote:
            if char == "\\":
                i += 2
                continue
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


_TRAILING_NAME_RE = re.compile(r"^\s*[A-Za-z_]?\w{0,64}\)?$")


_KNOWN_TOOL_RE = re.compile(r"^\s*(" + "|".join(KNOWN_TOOL_NAMES) + r")\s*\(")


def _holdback_start(buffer: str) -> int:
    """Index where visible output must stop because the tail could still grow
    into a bare tool-call span (e.g. '[' arrived but 'bash(' hasn't yet)."""
    idx = buffer.rfind("[")
    if idx == -1:
        return len(buffer)
    line_start = buffer.rfind("\n", 0, idx) + 1
    on_own_line = not buffer[line_start:idx].strip()
    suffix = buffer[idx + 1 :]
    if _KNOWN_TOOL_RE.match(suffix):
        return idx
    if on_own_line and _TRAILING_NAME_RE.search(suffix):
        return idx
    return len(buffer)


def _bare_region_start(buffer: str) -> int:
    """Start of a bare span ready to be suppressed: any span naming a known
    tool regardless of position, or an own-line span."""
    own_line = -1
    for match in _BARE_SPAN_CANDIDATE_RE.finditer(buffer):
        close_idx = _matching_bracket(buffer, match.start())
        if close_idx == -1:
            continue
        names = _region_call_names(buffer[match.start() : close_idx + 1], "lfm2")
        if any(name in KNOWN_TOOL_NAMES for name in names):
            return match.start()
        line_start = buffer.rfind("\n", 0, match.start()) + 1
        after = buffer[close_idx + 1 :].split("\n", 1)[0]
        if (
            own_line == -1
            and not buffer[line_start : match.start()].strip()
            and not after.strip()
            and names
        ):
            own_line = match.start()
    return own_line


def _region_call_names(body: str, tool_format: str) -> list[str]:
    if tool_format != "lfm2":
        return [name.strip() for name in _FUNCTION_NAME_RE.findall(body)]
    normalized = "".join(_SMART_QUOTES.get(char, char) for char in body)
    return [call["name"] for call in _parse_pythonic_calls(normalized, 0)]


def suppress_tool_blocks(pieces, tool_format: str = "qwen3_xml", on_call=None):
    """Yield streamed pieces with tool-call regions hidden.

    Detected calls are reported through ``on_call(name)`` so the UI can
    announce an attempt instead of showing raw call syntax. For lfm2 this
    also covers bare pythonic spans on their own line, including ones that
    name an unknown tool (those are attempts too).
    """
    notify = on_call or (lambda name: None)
    tool_open, tool_close = TOOL_MARKERS[tool_format]
    buffer = ""
    for piece in pieces:
        buffer += piece
        while True:
            region = _bare_region_start(buffer) if tool_format == "lfm2" else -1
            tag_idx = buffer.find(tool_open)
            if tag_idx != -1 and (region == -1 or tag_idx <= region):
                region = tag_idx
                is_tag = True
            else:
                is_tag = False

            if region == -1:
                visible, buffer = _split_at_blocks(buffer, tool_open, tool_close)
                if tool_format == "lfm2" and visible:
                    cut = _holdback_start(visible)
                    buffer = visible[cut:] + buffer
                    visible = visible[:cut]
                if visible:
                    yield visible
                break

            visible, buffer = buffer[:region], buffer[region:]
            if visible:
                yield visible

            if is_tag:
                close_idx = buffer.find(tool_close, len(tool_open))
                if close_idx == -1:
                    break
                body = buffer[len(tool_open) : close_idx]
                for name in _region_call_names(body, tool_format):
                    notify(name)
                buffer = buffer[close_idx + len(tool_close) :]
            else:
                close_idx = _matching_bracket(buffer, 0)
                if close_idx == -1:
                    if len(buffer) > BARE_SPAN_MAX_CHARS and not _BARE_SPAN_START_RE.match(buffer):
                        yield buffer
                        buffer = ""
                    break
                span = buffer[: close_idx + 1]
                names = _region_call_names(span, "lfm2")
                if names:
                    for name in names:
                        notify(name)
                    buffer = buffer[close_idx + 1 :]
                else:
                    yield span
                    buffer = buffer[close_idx + 1 :]

    if not buffer:
        return
    if tool_open in buffer:
        return
    if tool_format != "lfm2":
        yield buffer
        return

    cut = _holdback_start(buffer)
    head, fragment = buffer[:cut], buffer[cut:]
    if head:
        yield head
    if not fragment:
        return
    close_idx = _matching_bracket(fragment, 0)
    if close_idx == -1:
        yield fragment
        return
    for name in _region_call_names(fragment[: close_idx + 1], "lfm2"):
        notify(name)


def _accumulate_calls(calls: dict[int, dict], fragments: list) -> None:
    for fragment in fragments:
        index = fragment.get("index", 0)
        slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if fragment.get("id"):
            slot["id"] = fragment["id"]
        function = fragment.get("function") or {}
        if function.get("name"):
            slot["name"] = slot["name"] or function["name"]
        if function.get("arguments"):
            slot["arguments"] += function["arguments"]


def _response_step(history: list[dict], system: str) -> tuple[str, str, list[dict], str]:
    """Stream one model response; returns (visible text, raw text, tool calls,
    steering note). The note is non-empty when the user paused and injected
    input, in which case the stream was cut short."""
    calls: dict[int, dict] = {}
    raw_pieces: list[str] = []
    note = ""

    def content_pieces():
        messages = ([{"role": "system", "content": system}] if system else []) + history
        for chunk in stream_chat(messages, tools=TOOL_SCHEMAS):
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta", {})
            fragments = delta.get("tool_calls")
            if fragments:
                _accumulate_calls(calls, fragments)
            piece = delta.get("content")
            if piece:
                raw_pieces.append(piece)
                yield piece

    reply = StreamingReply(console)
    tool_format = selected_model().tool_format
    try:
        stream = pausable(suppress_tool_blocks(content_pieces(), tool_format), console)
        for in_thinking, text in split_thinking(stream):
            reply.add(in_thinking, text)
    except SteeringError as steered:
        note = steered.note
    reply.finish()

    ordered = [calls[index] for index in sorted(calls)]
    known = {(call["name"], call["arguments"]) for call in ordered}
    _, inline_calls = extract_tool_calls("".join(raw_pieces), tool_format)
    ordered += [call for call in inline_calls if (call["name"], call["arguments"]) not in known]

    raw_log_path = os.environ.get("GOLDEN_AGENT_RAW_LOG")
    if raw_log_path:
        with open(raw_log_path, "a", encoding="utf-8") as raw_log:
            raw_log.write("".join(raw_pieces) + "\n\n<<<end of response>>>\n\n")
    return reply.text, "".join(raw_pieces), ordered, note


def _preview(text: str, lines: int = PREVIEW_LINES) -> str:
    split = text.splitlines()
    preview = "\n".join(split[:lines])
    if len(split) > lines:
        preview += "\n..."
    return preview


def _lexer_for(path: str) -> str:
    try:
        from pygments.lexers import get_lexer_for_filename

        return get_lexer_for_filename(path).name.lower().replace(" ", "")
    except Exception:
        return "text"


PANEL_TOOLS = ("write_file", "edit_file", "bash")


def _result_status(name: str, args: dict, result: str) -> str:
    """One-line outcome summary shared by panel subtitles and status lines."""
    if result.startswith("error:"):
        return f"[red]✗ {result}[/red]"
    if name == "read_file":
        return f"[dim]✓ {args.get('path', '?')} · {len(result.splitlines())} lines[/dim]"
    if name == "web_search":
        count = max(1, len(result.split(chr(10) * 2)))
        return f"[dim]✓ {args.get('query', '?')} · {count} results[/dim]"
    if name == "bash":
        lines = len(result.splitlines())
        summary = f"{lines} lines of output" if lines else "no output"
        return f"[dim]✓ {summary}[/dim]"
    if name == "web_fetch":
        return f"[dim]✓ {args.get('url', '?')} · {len(result)} chars[/dim]"
    if name == "write_file":
        return f"[dim]✓ wrote {args.get('path', '?')}[/dim]"
    if name == "edit_file":
        return f"[dim]✓ edited {args.get('path', '?')}[/dim]"
    return f"[dim]✓ {len(result)} chars[/dim]"


def _panel_body(name: str, args: dict):
    if name == "write_file":
        return Syntax(_preview(args.get("content", "")), _lexer_for(args.get("path", "")))
    if name == "edit_file":
        return Text.assemble(
            ("- " + _preview(args.get("old_string", "")) + "\n", "red"),
            ("+ " + _preview(args.get("new_string", "")), "green"),
        )
    return Syntax(args.get("command", ""), "bash")


def render_tool_block(console: Console, name: str, args: dict, result: str) -> None:
    """Render one executed tool call with left-border style for file/bash work."""
    status = _result_status(name, args, result)
    if name not in PANEL_TOOLS:
        console.print(f"[cyan]*[/cyan] [bold]{name}[/bold]  {status}")
        console.print()
        return

    target = f" {args.get('path', '?')}" if name != "bash" else ""
    body = _panel_body(name, args)

    console.print()
    console.print(f"[dim]├─[/dim] [cyan]{name}[/cyan]{target}")
    if body:
        console.print("[dim]│[/dim]")
        console.print(body, highlight=False)
        console.print("[dim]│[/dim]")
    console.print(f"[dim]└─[/dim] {status}")
    console.print()


def _dispatch(call: dict, history: list[dict]) -> None:
    try:
        arguments = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError:
        _append_tool_result(history, call, "error: malformed JSON in tool arguments")
        return
    name = call["name"]
    with console.status(f"[dim]· {tool_label(name)}…[/dim]"):
        result = execute_tool(name, arguments)
    render_tool_block(console, name, arguments, result)
    _append_tool_result(history, call, result)


def _append_tool_result(history: list[dict], call: dict, result: str) -> None:
    history.append({"role": "tool", "tool_call_id": call["id"], "content": result})


def run_agent_turn(prompt: str, *, system: str = "", history: list[dict]) -> None:
    """Run one full user turn, looping over tool calls until a final answer."""
    history.append({"role": "user", "content": prompt})

    for _round in range(MAX_TOOL_ROUNDS):
        text, raw, calls, note = _response_step(history, system)
        if note:
            history.append({"role": "assistant", "content": text})
            history.append({"role": "user", "content": note})
            console.print(f"[dim]· continuing with your input: {note}[/dim]")
            continue
        assistant_message: dict = {"role": "assistant", "content": raw}
        structured = [call for call in calls if not call.get("inline")]
        if structured:
            assistant_message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for call in structured
            ]
        history.append(assistant_message)

        if not calls:
            return

        for call in calls:
            _dispatch(call, history)

    console.print(f"[red]halted after {MAX_TOOL_ROUNDS} tool rounds[/red]")


def print_tools() -> None:
    for schema in TOOL_SCHEMAS:
        function = schema["function"]
        console.print(f" {function['name']:<16} {function['description'].splitlines()[0]}")
    console.print("[dim](the model calls these automatically when needed)[/dim]")
