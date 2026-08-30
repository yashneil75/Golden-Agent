import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

MAX_TOOL_RESULT_CHARS = 20_000
MAX_READ_CHARS = 20_000
FETCH_MAX_BYTES = 2_000_000
FETCH_TIMEOUT_SECONDS = 20
RUN_COMMAND_TIMEOUT_SECONDS = 120
USER_AGENT = "golden-agent/0.1"

_launch_dir = Path.cwd()
_allowed_roots: set[Path] = set()


class ToolDeniedError(Exception):
    """Raised when the user refuses a file modification."""


def begin_session(launch_dir: Path | None = None) -> None:
    global _launch_dir
    _launch_dir = (Path(launch_dir) if launch_dir else Path.cwd()).resolve()
    _allowed_roots.clear()


def launch_dir() -> Path:
    return _launch_dir


def _resolve(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = _launch_dir / candidate
    return candidate.resolve()


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _ensure_write_permission(target: Path) -> None:
    if _is_within(target, _launch_dir):
        return
    for root in _allowed_roots:
        if _is_within(target, root):
            return

    from golden_agent.render import console

    console.print(f"[yellow]warning: {target} is outside {_launch_dir}[/yellow]")
    while True:
        choice = input(
            f"allow modifying {target.name}? [y] once / [a] always this session / [n] deny: "
        )
        choice = choice.strip().lower()
        if choice == "y":
            return
        if choice == "a":
            _allowed_roots.add(target.parent)
            return
        if choice == "n":
            raise ToolDeniedError(f"user denied modification of {target}")
        console.print("[dim]answer with y, a, or n[/dim]")


def _cap(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def read_file(path: str, offset: int = 1) -> str:
    """Read a text file starting at a 1-based line offset.

    Returns at most MAX_READ_CHARS characters worth of complete lines and,
    when the file continues past that, a footer stating the range shown and
    the offset for the next call so a long file can be paged in full.
    """
    target = _resolve(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, offset)
    window = lines[start - 1 :]

    body_lines: list[str] = []
    used = 0
    for line in window:
        cost = len(line) + 1
        if body_lines and used + cost > MAX_READ_CHARS:
            break
        body_lines.append(line)
        used += cost

    body = "\n".join(body_lines)
    shown_to = start + len(body_lines) - 1
    if start + len(body_lines) <= len(lines):
        footer = (
            f"\n[showing lines {start}-{shown_to} of {len(lines)}; "
            f"call read_file again with offset={shown_to + 1}]"
        )
        return body + footer
    return body


def write_file(path: str, content: str) -> str:
    target = _resolve(path)
    _ensure_write_permission(target)
    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    action = "overwrote" if existed else "created"
    return f"{action} {target} ({len(content)} chars, {len(content.splitlines())} lines)"


def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    target = _resolve(path)
    _ensure_write_permission(target)
    text = target.read_text(encoding="utf-8", errors="replace")
    count = text.count(old_string)
    if count == 0:
        return f"error: old_string not found in {path}"
    if count > 1 and not replace_all:
        return (
            f"error: old_string matches {count} times in {path}; "
            "include more surrounding context or set replace_all=true"
        )
    occurrences = count if replace_all else 1
    updated = text.replace(old_string, new_string, -1 if replace_all else 1)
    target.write_text(updated, encoding="utf-8")
    return f"edited {target} ({occurrences} replacement(s))"


def web_search(query: str, max_results: int = 5) -> str:
    from ddgs import DDGS

    results = DDGS().text(query, max_results=max_results)
    if not results:
        return f"no results found for {query!r}"
    blocks = [f"{r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}" for r in results]
    return _cap("\n\n".join(blocks))


class _TextExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript", "template"}
    BREAK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def web_fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        raw = response.read(FETCH_MAX_BYTES)
        content_type = response.headers.get_content_type()

    if content_type == "text/html":
        extractor = _TextExtractor()
        extractor.feed(raw.decode("utf-8", errors="replace"))
        text = "".join(extractor.parts)
        lines = [line.strip() for line in text.splitlines()]
        return _cap("\n".join(line for line in lines if line))

    return _cap(raw.decode("utf-8", errors="replace"))


def bash_path() -> str | None:
    """Locate a bash executable, falling back to the one bundled with Git."""
    found = shutil.which("bash")
    if found:
        return found
    git = shutil.which("git")
    if git:
        candidate = Path(git).parent.parent / "bin" / "bash.exe"
        if candidate.exists():
            return str(candidate)
    return None


def run_command(command: str, timeout: int = RUN_COMMAND_TIMEOUT_SECONDS) -> str:
    executable = bash_path()
    if executable is None:
        return "error: no bash executable found on this system"
    completed = subprocess.run(
        [executable, "-c", command],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        cwd=_launch_dir,
    )
    output = (completed.stdout + completed.stderr).strip()
    status = f"[exit {completed.returncode}]"
    return _cap(f"{status}\n{output}" if output else status)


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file. Returns at most 20000 characters per "
                "call; when the file is longer, the result ends with a "
                "[showing lines X-Y of Z ...] footer - keep calling with "
                "the offset given there until you have seen every line. "
                "ALWAYS call this before edit_file so you know the exact "
                "text to replace. Relative paths resolve from the directory "
                "Golden Agent was launched in."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to a single FILE, e.g. 'src/main.py'. "
                            "Not a directory, not a glob pattern."
                        ),
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "1-based line number to start reading from "
                            "(default 1 = beginning). Use the offset from a "
                            "previous call's footer to continue reading a "
                            "long file."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a new file or COMPLETELY overwrite an existing one. "
                "content becomes the entire file. To change only part of an "
                "existing file, use edit_file instead. Writing outside the "
                "working directory asks the user for permission."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file to create or overwrite.",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Complete final file content. Include EVERYTHING "
                            "- never a diff, snippet, or placeholder like "
                            "'... rest unchanged'."
                        ),
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact substring in an existing file. Rules: "
                "(1) call read_file on the path first; "
                "(2) old_string must match the file character-for-character, "
                "including indentation, spaces, and newlines - copy it "
                "directly from the read_file output; "
                "(3) include enough surrounding lines to make old_string "
                "unique; if it matches more than once the call fails unless "
                "replace_all is true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the existing file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": (
                            "Exact existing text to replace, copied verbatim "
                            "from the file. Never paraphrase or reformat it."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": (
                            "Replacement text of the same shape as "
                            "old_string. Pass an empty string to delete."
                        ),
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": ("true = replace every occurrence (default false)."),
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web and return titles, URLs, and snippets. "
                "Use this when you need facts, documentation, or current "
                "information. To read a result in full, pass its URL to "
                "web_fetch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search keywords, e.g. 'pytest fixture scope'. "
                            "Not a full sentence and not a URL."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Results to return, 1-10 (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run ONE command with bash and return stdout, stderr, and "
                "the exit code. Commands run from the directory Golden Agent was "
                "launched in. Even on Windows this is real bash (Git Bash): "
                "use ls/grep/sed, never dir or PowerShell syntax. Prefer "
                "read_file/write_file/edit_file for anything involving file "
                "contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "A single bash command, e.g. 'pytest tests/ -x' or 'git status'."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": ("Seconds before the command is killed (default 120)."),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Download a URL and return its readable text (HTML stripped, "
                "capped at ~2 million characters). url must be a complete "
                "http:// or https:// URL. Use it to read a specific page; if "
                "you only have keywords, search with web_search first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Full URL including scheme, e.g. 'https://example.com/page'."
                        ),
                    },
                },
                "required": ["url"],
            },
        },
    },
]

_REGISTRY = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "bash": run_command,
    "web_search": web_search,
    "web_fetch": web_fetch,
}


def execute_tool(name: str, arguments: dict) -> str:
    func = _REGISTRY.get(name)
    if func is None:
        return f"error: unknown tool {name!r}; valid tools: {', '.join(_REGISTRY)}"
    try:
        return func(**arguments)
    except ToolDeniedError as exc:
        return f"error: {exc}"
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"
