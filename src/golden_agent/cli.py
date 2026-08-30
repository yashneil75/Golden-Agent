import argparse
import signal
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TransferSpeedColumn,
)

from golden_agent.agent import print_tools, run_agent_turn
from golden_agent.backend import ensure_server_binary, stop_server
from golden_agent.config import (
    BACKEND_LABELS,
    DEFAULT_MODEL_KEY,
    MODELS,
    Backend,
    detect_backend,
    load_config,
    save_config,
)
from golden_agent.download import model_cached
from golden_agent.model import (
    bootstrap_server,
    call_model_stream,
    count_tokens,
    selected_model,
    set_model,
)
from golden_agent.render import (
    console,
    err,
    extract_answer,
    info,
    ok,
    turn_separator,
)
from golden_agent.system_prompt import sys_prompt
from golden_agent.tools import begin_session

SUMMARY_TRIGGER_TOKENS = 140_000
KEEP_TURNS = 6
SUMMARY_SYSTEM = (
    "You compress conversation transcripts into dense summaries. "
    "Preserve every key decision, fact, file path, snippet, and open task. Output only the summary."
)

_selected_backend: Backend = Backend.CPU

GOLD = "\033[38;2;232;185;35m"
RESET = "\033[0m"

PT_STYLE = Style.from_dict({"rust": "#e8b923 bold", "dim": "#888888", "pointer": "#e8b923 bold"})

COMMANDS = ("/clear", "/exit", "/help", "/tools")


ASCII_ART = f"""{GOLD}
 ██████╗  ██████╗ ██╗     ██████╗ ███████╗███╗   ██╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔════╝ ██╔═══██╗██║     ██╔══██╗██╔════╝████╗  ██║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██║  ███╗██║   ██║██║     ██║  ██║█████╗  ██╔██╗ ██║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
██║   ██║██║   ██║██║     ██║  ██║██╔══╝  ██║╚██╗██║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
╚██████╔╝╚██████╔╝███████╗██████╔╝███████╗██║ ╚████║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
 ╚═════╝  ╚═════╝ ╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═══╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝                                                                                        
 {RESET}"""


def prepare_runtime() -> None:
    """Download the llama-server binary and model weights at startup."""
    global _selected_backend
    with console.status("[dim]preparing llama-server binary ...[/dim]", spinner="dots") as status:

        def _log(msg: str) -> None:
            status.update(f"[dim]{msg}[/dim]")

        try:
            ensure_server_binary(backend=_selected_backend, log=_log)
        except RuntimeError as exc:
            # The selected backend's prebuilt couldn't be fetched; fall back to
            # CPU so the agent still runs.
            if _selected_backend != Backend.CPU:
                err(
                    f"{BACKEND_LABELS[_selected_backend]} backend unavailable "
                    f"({exc}); falling back to CPU"
                )
                _selected_backend = Backend.CPU
                ensure_server_binary(backend=_selected_backend, log=_log)
            else:
                raise
    ok("llama-server binary ready")


def _lazy_bootstrap() -> None:
    """Download model weights and start llama-server (no-op once ready)."""
    global _selected_backend
    spec = selected_model()

    main_cached = model_cached(filename=spec.filename, expected_size=spec.size_bytes)
    draft_cached = (
        spec.draft_filename is None
        or model_cached(filename=spec.draft_filename, expected_size=spec.draft_size_bytes)
    )

    if main_cached and draft_cached:
        with console.status(f"[dim]loading {spec.label} ...[/dim]", spinner="dots") as status:

            def _log(msg: str) -> None:
                status.update(f"[dim]{msg}[/dim]")

            bootstrap_server(backend=_selected_backend, log=_log)
        ok(f"{spec.label} ready")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            spec.label,
            total=spec.size_bytes,
            completed=spec.size_bytes if main_cached else 0,
        )
        draft_task = None
        if spec.draft_filename and spec.draft_size_bytes:
            draft_task = progress.add_task(
                spec.draft_filename,
                total=spec.draft_size_bytes,
                completed=spec.draft_size_bytes if draft_cached else 0,
            )

        def _download_progress(done: int, total: int) -> None:
            progress.update(task, completed=done)

        def _draft_progress(done: int, total: int) -> None:
            if draft_task is not None:
                progress.update(draft_task, completed=done)

        bootstrap_server(
            backend=_selected_backend,
            download_progress=_download_progress,
            draft_progress=_draft_progress,
            log=lambda _: None,
        )
    ok(f"{spec.label} ready")


HELP_TEXT = """\
commands:
  /tools          list the tools available to Golden Agent
  /clear          clear conversation history
  /help           show this help
  /exit           quit

keys:
  esc             pause/resume generation (while streaming)
  enter           add input while paused"""


def print_help() -> None:
    console.print(HELP_TEXT)


def print_startup(console: Console) -> None:
    info("esc to interrupt · /help for commands · /exit to quit", console)
    console.print()


class GoldenAgentCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        parts = text.split()
        if not parts:
            return
        if len(parts) == 1 and parts[0].startswith("/"):
            for command in COMMANDS:
                if command.startswith(parts[0].lower()):
                    yield Completion(command, start_position=-len(parts[0]))


def make_prompt_session() -> PromptSession:
    history_dir = Path.home() / ".golden-agent"
    history_dir.mkdir(exist_ok=True)
    return PromptSession(
        history=FileHistory(history_dir / "history"),
        style=PT_STYLE,
        completer=GoldenAgentCompleter(),
        complete_while_typing=True,
    )


def handle_command(user_input: str, history: list[dict] | None = None) -> bool:
    """Run a slash command; returns False if it was not a known command."""
    parts = user_input.strip().split(maxsplit=1)
    command = parts[0].lower()

    if command == "/help":
        print_help()
        return True
    if command == "/clear":
        if history:
            history.clear()
        info("conversation cleared")
        return True
    if command == "/tools":
        print_tools()
        return True
    return False


def compact_history(history: list[dict], system: str) -> bool:
    """Summarize older turns once history nears the context window."""
    total = count_tokens(system) + sum(count_tokens(m["content"]) for m in history)
    if total <= SUMMARY_TRIGGER_TOKENS:
        return False

    keep = KEEP_TURNS * 2
    old = history[:-keep]
    if not old:
        return False

    transcript = "\n\n".join(f"{m['role']}: {m['content']}" for m in old)
    summary_prompt = (
        "Summarize the following earlier conversation between the user and the "
        "assistant so work can continue seamlessly.\n\n"
        f"{transcript}"
    )

    console.print("[dim]context nearing limit; summarizing earlier turns ...[/dim]")
    raw = "".join(call_model_stream(summary_prompt, system=SUMMARY_SYSTEM))
    summary = extract_answer(raw)

    history[:] = [
        {"role": "user", "content": f"[summary of earlier conversation]\n{summary}"},
        *history[-keep:],
    ]
    info("earlier turns summarized")
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="golden-agent",
        description="A local AI coding agent powered by llama.cpp server",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("setup", help="interactively choose backend and model")

    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default=None,
        help=f"override saved model (default: {DEFAULT_MODEL_KEY})",
    )
    return parser.parse_args(argv)


def setup() -> None:
    """Interactive setup: pick backend and model, save to ~/.golden-agent/config.json."""
    print(ASCII_ART)
    session = make_prompt_session()

    detected = detect_backend()
    backend_keys = list(Backend)
    backend_labels = [
        f"{BACKEND_LABELS[b]} (detected)" if b == detected else BACKEND_LABELS[b]
        for b in backend_keys
    ]
    default_idx = backend_keys.index(detected)

    console.print()
    console.print("[bold]Select backend:[/bold]")
    for i, label in enumerate(backend_labels, 1):
        marker = " <--" if i - 1 == default_idx else ""
        console.print(f"  [{i}] {label}{marker}")
    console.print()

    try:
        choice = session.prompt([("class:rust", "❯ ")])
    except (EOFError, KeyboardInterrupt):
        print("\nsetup cancelled")
        return
    idx = int(choice.strip()) - 1 if choice.strip().isdigit() else default_idx
    if 0 <= idx < len(backend_keys):
        backend = backend_keys[idx]
    else:
        backend = detected
    ok(f"backend: {BACKEND_LABELS[backend]}")

    console.print()
    console.print("[bold]Select model:[/bold]")
    model_keys = list(MODELS)
    for i, key in enumerate(model_keys, 1):
        spec = MODELS[key]
        size_mb = spec.size_bytes / 1_048_576
        draft_mb = spec.draft_size_bytes / 1_048_576 if spec.draft_size_bytes else 0
        draft_info = f" + {draft_mb:.0f}MB draft" if draft_mb else ""
        marker = " <--" if key == DEFAULT_MODEL_KEY else ""
        console.print(f"  [{i}] {spec.label}  ({size_mb:.0f}MB{draft_info}){marker}")
    console.print()

    try:
        choice = session.prompt([("class:rust", "❯ ")])
    except (EOFError, KeyboardInterrupt):
        print("\nsetup cancelled")
        return
    idx = int(choice.strip()) - 1 if choice.strip().isdigit() else model_keys.index(
        DEFAULT_MODEL_KEY
    )
    if 0 <= idx < len(model_keys):
        model_key = model_keys[idx]
    else:
        model_key = DEFAULT_MODEL_KEY
    ok(f"model: {MODELS[model_key].label}")

    save_config({"backend": backend.value, "model": model_key})
    ok("saved to ~/.golden-agent/config.json")
    console.print()
    console.print("Run [bold]golden-agent[/bold] to start chatting.")


def _apply_saved_config(args: argparse.Namespace) -> None:
    """Load saved config and apply backend + model selection."""
    global _selected_backend

    config = load_config()

    backend_val = config.get("backend")
    if backend_val:
        try:
            _selected_backend = Backend(backend_val)
        except ValueError:
            _selected_backend = detect_backend()
    else:
        _selected_backend = detect_backend()

    model_key = args.model or config.get("model") or DEFAULT_MODEL_KEY
    if model_key not in MODELS:
        model_key = DEFAULT_MODEL_KEY
    set_model(model_key)


def _repl_loop(session: PromptSession) -> None:
    """Main REPL loop, shared by repl() after setup."""
    begin_session()
    print_startup(console)

    history: list[dict] = []
    while True:
        console.print()
        try:
            user_input = session.prompt([("class:rust", "❯ ")])
        except (EOFError, KeyboardInterrupt):
            print("\nSee Ya Soon!")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped.startswith("/"):
            if stripped.lower() == "/exit":
                break
            if not handle_command(stripped, history=history):
                err(f"unknown command {stripped.split()[0]!r}")
                print_help()
            turn_separator(console)
            continue

        compact_history(history, sys_prompt)
        console.print()
        try:
            run_agent_turn(stripped, system=sys_prompt, history=history)
        except KeyboardInterrupt:
            console.print()
            turn_separator(console)


def _ensure_server_cleanup() -> None:
    """Register signal handlers and atexit to guarantee server shutdown."""
    import atexit

    atexit.register(stop_server)

    def _handle_signal(signum, frame):
        stop_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def repl() -> None:
    """Main entry point: no-args golden-agent uses saved config, no interactive selectors."""
    args = _parse_args()

    if args.command == "setup":
        setup()
        return

    print(ASCII_ART)
    _apply_saved_config(args)
    ok(f"backend: {BACKEND_LABELS[_selected_backend]}")
    ok(f"model: {selected_model().label}")

    try:
        prepare_runtime()
        _lazy_bootstrap()
    except (EOFError, KeyboardInterrupt):
        print("\ngoodbye")
        return
    except (RuntimeError, KeyError) as error:
        err(str(error))
        return

    _ensure_server_cleanup()

    session = make_prompt_session()
    try:
        _repl_loop(session)
    finally:
        stop_server()


if __name__ == "__main__":
    repl()
