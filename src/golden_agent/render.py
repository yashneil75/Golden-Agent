from collections.abc import Generator, Iterable, Iterator

from markrender import MarkdownRenderer
from rich.console import Console
from rich.text import Text

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TAG_HOLDBACK = max(len(THINK_OPEN), len(THINK_CLOSE)) - 1

THINKING_STYLE = "rgb(136,136,136)"

console = Console()


def ok(message: str, console: Console = console) -> None:
    console.print(f"[green]✓[/green] {message}")


def err(message: str, console: Console = console) -> None:
    console.print(f"[bold red]✗[/bold red] {message}")


def info(message: str, console: Console = console) -> None:
    console.print(f"[dim]· {message}[/dim]")


def turn_separator(console: Console) -> None:
    console.print()


class StreamingReply:
    """Append-only streaming: reasoning prints greyed out, answer renders as
    markdown via markrender.

    Nothing is ever erased or redrawn, so content taller than the terminal
    cannot be re-emitted. Some chat templates open <think> inside the prompt,
    so reasoning may start with no explicit open tag and only be marked by
    </think>. If the stream ends without the block ever being closed, finish()
    treats the reasoning text as the answer for history purposes; visually it
    stays greyed because it was already printed.
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._thinking = ""
        self._answer = ""
        self._closed = False
        self._prev_in_thinking = True
        self._renderer: MarkdownRenderer | None = None
        self._last_width = console.size.width

    def add(self, in_thinking: bool, text: str) -> None:
        if not text:
            return
        if self._prev_in_thinking and not in_thinking:
            self._closed = True
            self._console.print()
            self._console.print()
        self._prev_in_thinking = in_thinking
        if in_thinking:
            self._thinking += text
            self._console.print(Text(text, style=THINKING_STYLE), end="")
        else:
            self._answer += text
            self._drop_stale_renderer()
            self._ensure_renderer().render(text)

    def _drop_stale_renderer(self) -> None:
        """markrender pins terminal width at construction; rebuild it when the
        terminal is resized mid-response so later text wraps correctly."""
        width = self._console.size.width
        if self._renderer is not None and width != self._last_width:
            self._renderer.finalize()
            self._renderer = None
        self._last_width = width

    @property
    def text(self) -> str:
        return self._answer

    @property
    def closed_think_block(self) -> bool:
        return self._closed

    def _ensure_renderer(self) -> MarkdownRenderer:
        if self._renderer is None:
            self._renderer = MarkdownRenderer(
                output=self._console.file,
                use_config=False,
                force_color=self._console.is_terminal,
            )
        return self._renderer

    def finish(self) -> None:
        if not self._closed and self._thinking and not self._answer:
            self._answer = self._thinking.lstrip()
        if self._renderer is not None:
            self._renderer.finalize()
            self._renderer = None


def split_thinking(tokens: Iterable[str]) -> Generator[tuple[bool, str], None, None]:
    """Yield (in_thinking, text) events from streamed model output.

    The chat template opens <think> in the prompt, so the stream always
    starts inside a thinking block even without an explicit <think> tag.
    A tail long enough to hold a partial tag is held back so tags split
    across chunks are still detected.
    """
    buffer = ""
    in_thinking = True
    trim_answer = False

    def emit(text: str) -> Iterator[tuple[bool, str]]:
        nonlocal trim_answer
        if not text:
            return
        if not in_thinking and trim_answer:
            text = text.lstrip()
            if not text:
                return
            trim_answer = False
        yield in_thinking, text

    for token in tokens:
        buffer += token
        while True:
            open_pos = buffer.find(THINK_OPEN)
            close_pos = buffer.find(THINK_CLOSE)

            if close_pos != -1 and (open_pos == -1 or close_pos < open_pos):
                yield from emit(buffer[:close_pos])
                buffer = buffer[close_pos + len(THINK_CLOSE) :]
                in_thinking = False
                trim_answer = True
                continue

            if open_pos != -1:
                if not in_thinking:
                    yield False, buffer[:open_pos]
                buffer = buffer[open_pos + len(THINK_OPEN) :]
                in_thinking = True
                continue

            break

        if len(buffer) > TAG_HOLDBACK:
            safe, buffer = buffer[:-TAG_HOLDBACK], buffer[-TAG_HOLDBACK:]
            yield from emit(safe)

    yield from emit(buffer)


def extract_answer(raw: str) -> str:
    """Return the visible (non-thinking) text from a complete model response."""
    if THINK_CLOSE in raw:
        raw = raw.split(THINK_CLOSE, 1)[1]
    return raw.replace(THINK_OPEN, "").strip()
