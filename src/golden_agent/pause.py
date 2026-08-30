"""Pause/resume and steering controls for streaming generation.

Esc toggles pause/resume mid-stream. Because llama.cpp generation is
pull-based, not consuming the generator genuinely halts compute. While
paused, Enter opens an input field; submitting injects the note into the
conversation and generation continues from there (SteeringError).
"""

import os
import time
from collections.abc import Iterable, Iterator

from rich.console import Console

ESC = "\x1b"
ENTER = "\r"

PAUSED_HINT = "[dim]· paused — esc resume · enter add input[/dim]"
RESUMED_HINT = "[dim]· resumed[/dim]"


class SteeringError(Exception):
    """Raised when the user submits a steering note while paused."""

    def __init__(self, note: str) -> None:
        super().__init__(note)
        self.note = note


class _KeyCapture:
    """Non-blocking stdin key polling; raw mode on POSIX only."""

    def __enter__(self):
        self._restore = None
        if os.name != "nt":
            import termios
            import tty

            self._restore = termios.tcgetattr(0)
            tty.setcbreak(0)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._restore is not None:
            import termios

            termios.tcsetattr(0, termios.TCSADRAIN, self._restore)
            self._restore = None

    def poll(self) -> str | None:
        if os.name == "nt":
            import msvcrt

            if not msvcrt.kbhit():
                return None
            char = msvcrt.getwch()
            if char in ("\x00", "\xe0"):
                msvcrt.getwch()
                return None
            return char

        import select
        import sys

        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        char = sys.stdin.read(1)
        if char == ESC:
            while select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)
        return char


def _collect_note(console: Console) -> str:
    from prompt_toolkit import prompt

    console.print()
    try:
        return prompt("❯ ", message="❯ ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def pausable(stream: Iterable[str], console: Console) -> Iterator[str]:
    """Yield items from ``stream``, honoring Esc pause/resume between chunks."""
    paused = False
    with _KeyCapture() as keys:
        for item in stream:
            while True:
                key = keys.poll()
                if key == ESC:
                    paused = not paused
                    console.print(PAUSED_HINT if paused else RESUMED_HINT)
                elif key == ENTER:
                    if paused:
                        console.print()
                        note = _collect_note(console)
                        if note:
                            raise SteeringError(note)
                        paused = False
                        console.print(RESUMED_HINT)
                    else:
                        break
                elif key == "\x03":
                    raise KeyboardInterrupt
                if not paused:
                    break
                time.sleep(0.05)
            yield item
