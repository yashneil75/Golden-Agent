import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from golden_agent.config import (
    BACKEND_LABELS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    SPEC_TYPE_NONE,
    Backend,
    ModelSpec,
    _server_download_url,
    load_inference,
)

DEFAULT_NGL_LAYERS = 99

# Control socket the golden-agent uses to talk to the always-on server daemon.
CONTROL_PORT = 2012

LogFn = Callable[[str], None]

# The llama-server is owned by the detached daemon, not by this process, so we
# only track its pid/port. The hookpoint sentinel is a direct child of this
# (agent) process and is reaped with it.
_server_pid: int | None = None
_server_port: int | None = None
_hookpoint_process: subprocess.Popen | None = None

# Windows flags. CREATE_BREAKAWAY_FROM_JOB isn't always exposed by the stdlib
# subprocess module, so fall back to the raw value (0x01000000).
CREATE_BREAKAWAY_FROM_JOB = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)


def _popen_detached(
    args: list[str],
    extra_flags: int = 0,
    debug: bool = False,
) -> subprocess.Popen:
    """Spawn a child that does NOT share the agent's job/process group.

    On Windows we ask to break away from any job the agent belongs to; if the
    agent isn't in a job (or the job forbids breakaway) the flag is ignored or
    we retry without it. On POSIX we start a new session. Either way the child
    survives the agent being killed and is reaped by the daemon instead.

    In ``debug`` mode the child inherits the parent's stdout/stderr so its log
    output is visible in the terminal instead of being discarded.
    """
    std_out = None if debug else subprocess.DEVNULL
    std_err = None if debug else subprocess.DEVNULL

    if sys.platform == "win32":
        flags = extra_flags | CREATE_BREAKAWAY_FROM_JOB
        try:
            return subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=std_out,
                stderr=std_err,
                creationflags=flags,
            )
        except OSError:
            flags &= ~CREATE_BREAKAWAY_FROM_JOB
            return subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=std_out,
                stderr=std_err,
                creationflags=flags,
            )

    return subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=std_out,
        stderr=std_err,
        start_new_session=True,
    )


def _process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x0400, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _ensure_daemon() -> None:
    """Start the always-on server daemon if it isn't already running.

    The daemon is spawned fully detached (see ``_popen_detached``) so it is
    independent of the agent's lifecycle and survives the agent dying.
    """
    runtime = _cache_dir()
    pid_file = runtime / "daemon.pid"
    if pid_file.exists():
        try:
            old = int(pid_file.read_text().strip())
        except Exception:
            old = None
        if old and _process_alive(old):
            return

    daemon_script = str(Path(__file__).resolve().with_name("server_daemon.py"))
    _popen_detached(
        [
            sys.executable,
            daemon_script,
            "--control-port",
            str(CONTROL_PORT),
            "--runtime-dir",
            str(runtime),
        ],
        debug=bool(os.environ.get("GOLDEN_AGENT_DEBUG")),
    )


def _send_to_daemon(payload: dict, retries: int = 25, timeout: float = 0.5) -> dict:
    """Send a JSON command to the daemon and return its parsed response."""
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect(("127.0.0.1", CONTROL_PORT))
                s.sendall((json.dumps(payload) + "\n").encode())
                raw = s.recv(65536)
            return json.loads(raw.decode().strip())
        except Exception as exc:  # daemon may still be starting up
            last_err = exc
            time.sleep(0.1)
    raise RuntimeError(f"server daemon unreachable: {last_err}")


def _log(message: str) -> None:
    print(message)


def _cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path(os.environ.get("HOME") or Path.home()) / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return base / "golden-agent" / "server"


def server_binary_path(backend: Backend = Backend.CPU) -> Path:
    suffix = f"-{backend.value}"
    if sys.platform == "win32":
        return _cache_dir() / f"llama{suffix}.exe"
    return _cache_dir() / f"llama-server{suffix}"


def ensure_server_binary(
    backend: Backend = Backend.CPU,
    log: LogFn = _log,
) -> Path:
    """Download the llama-server binary for ``backend`` if not already cached."""
    binary = server_binary_path(backend)
    if binary.is_file():
        return binary

    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    url = _server_download_url(backend=backend)
    log("downloading llama-server ...")

    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request) as response:
            data = response.read()
    except Exception as exc:
        raise RuntimeError(f"failed to download llama-server: {exc}") from exc

    if sys.platform == "win32":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                lower = name.lower()
                basename = lower.rsplit("/", 1)[-1]
                if basename.endswith((".dll", ".exe")):
                    extracted = zf.read(name)
                    # The main server exe is always `llama-server.exe` in the
                    # archive; write it to the backend-specific `binary` path so
                    # the launcher finds it. Other DLLs keep their own names.
                    target = binary if basename == "llama-server.exe" else binary.parent / basename
                    target.write_bytes(extracted)
        if not binary.exists():
            raise RuntimeError("llama-server not found in downloaded archive")
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for member in tf.getmembers():
                basename = member.name.rsplit("/", 1)[-1]
                if "llama-server" in basename or basename.startswith("llama-"):
                    extracted = tf.extractfile(member)
                    if extracted:
                        binary.write_bytes(extracted.read())
                    break
            else:
                raise RuntimeError("llama-server not found in downloaded archive")

    if sys.platform != "win32":
        os.chmod(binary, 0o755)

    log("llama-server downloaded")
    return binary


def _build_server_args(
    spec: ModelSpec,
    target_path: str,
    draft_path: str | None = None,
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    backend: Backend = Backend.CPU,
    inference: dict | None = None,
) -> list[str]:
    binary = str(server_binary_path(backend))
    args = [binary]

    args.extend([
        "-m", target_path,
        "--host", host,
        "--port", str(port),
        "-fa", "on",
        "-ctk", "q5_0",
        "-ctv", "q5_0",
    ])

    if backend != Backend.CPU:
        args.extend(["-ngl", str(DEFAULT_NGL_LAYERS)])

    if spec.n_ctx:
        args.extend(["-c", str(spec.n_ctx)])

    overrides = (inference or {}).get(spec.key, {}) or {}

    if spec.spec_type != SPEC_TYPE_NONE and draft_path:
        args.extend(["-md", draft_path, "--spec-type", spec.spec_type])
        n_max = overrides.get("spec_draft_n_max")
        if n_max:
            args.extend(["--spec-draft-n-max", str(n_max)])

    if spec.reasoning != "auto":
        # Reasoning flags are only honored with the Jinja chat template.
        args.extend(["--jinja", "--reasoning", spec.reasoning])
        # Keep think tags inline in content; the client renders them itself.
        args.extend(["--reasoning-format", "none"])
        effort = overrides.get("reasoning_effort", spec.reasoning_effort)
        if effort:
            args.extend(["--reasoning-effort", effort])

    for key, value in (spec.sampling_overrides or {}).items():
        if key == "temperature":
            args.extend(["--temp", str(value)])
        elif key == "top_p":
            args.extend(["--top-p", str(value)])
        elif key == "top_k":
            args.extend(["--top-k", str(value)])
        elif key == "repeat_penalty":
            args.extend(["--repeat-penalty", str(value)])

    extra = overrides.get("extra_args") or []
    if isinstance(extra, list):
        args.extend(str(a) for a in extra)

    return args


def _wait_for_server(host: str, port: int, timeout: float = 120.0) -> bool:
    """Poll /health until the server responds or timeout."""
    import http.client

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            conn.close()
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_server(
    spec: ModelSpec,
    target_path: str,
    draft_path: str | None = None,
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    backend: Backend = Backend.CPU,
    log: LogFn = _log,
):
    """Start llama-server via the always-on daemon.

    The agent spawns a hookpoint sentinel (direct child, reaped with the agent)
    and asks the detached daemon to launch the server. The daemon's monitor
    watches the hookpoint and kills the server if the agent disappears, so a
    crashed/killed agent never leaves the model weights in memory.
    """
    global _server_pid, _server_port, _hookpoint_process

    stop_server()

    inference = load_inference()
    args = _build_server_args(spec, target_path, draft_path, port, host, backend, inference)
    log(f"starting llama-server on {host}:{port} ...")

    # Hookpoint is a normal child of the agent: it dies with the agent (via the
    # job object) and signals the daemon's monitor that the agent is gone.
    hookpoint_script = str(Path(__file__).resolve().with_name("hookpoint.py"))
    _hookpoint_process = subprocess.Popen(
        [sys.executable, hookpoint_script, str(os.getpid())],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    _ensure_daemon()
    resp = _send_to_daemon(
        {
            "cmd": "START",
            "host": host,
            "port": port,
            "hookpoint_pid": _hookpoint_process.pid,
            "server_args": args,
        }
    )
    if resp.get("status") != "ok":
        _kill_hookpoint()
        raise RuntimeError(f"server daemon failed to start llama-server: {resp}")

    _server_pid = resp.get("server_pid")
    _server_port = port

    if not _wait_for_server(host, port):
        # The requested GPU backend may have failed to initialize (missing or
        # broken Vulkan/CUDA driver, no usable device, etc.). Fall back to CPU
        # once so the agent still works instead of hard-failing.
        if backend != Backend.CPU:
            stop_server()
            log(
                f"[{BACKEND_LABELS[backend]}] backend failed to start; "
                "retrying with CPU"
            )
            ensure_server_binary(backend=Backend.CPU, log=log)
            start_server(
                spec,
                target_path,
                draft_path=draft_path,
                port=port,
                host=host,
                backend=Backend.CPU,
                log=log,
            )
            return
        stop_server()
        raise RuntimeError(
            f"llama-server failed to start on {host}:{port} within 120s; "
            "try a different model"
        )

    log("llama-server ready")


def _kill_hookpoint() -> None:
    global _hookpoint_process
    proc = _hookpoint_process
    _hookpoint_process = None
    if proc is not None and proc.poll() is None:
        _safe_kill(proc)


def _safe_kill(proc: "subprocess.Popen | object") -> None:
    try:
        proc.terminate()  # type: ignore[attr-defined]
        try:
            proc.wait(timeout=5)  # type: ignore[attr-defined]
        except Exception:
            proc.kill()  # type: ignore[attr-defined]
            try:
                proc.wait(timeout=3)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass


def _safe_kill_pid(pid: int) -> None:
    import subprocess

    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    import os

    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass


def _kill_servers_on_port(port: int = DEFAULT_PORT) -> None:
    """Best-effort reap of any llama-server still bound to ``port``.

    Catches orphaned servers whose tracked ``Popen`` reference was lost (e.g.
    an agent that crashed before recording the handle). Uses psutil when
    available; otherwise falls back to the OS netstat/lsof tooling.
    """
    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]

    pids: set[int] = set()
    if psutil is not None:
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if (
                    conn.laddr
                    and conn.laddr.port == port
                    and conn.status == psutil.CONN_LISTEN
                    and conn.pid
                ):
                    pids.add(conn.pid)
        except Exception:
            pids = _pids_on_port_fallback(port)
    else:
        pids = _pids_on_port_fallback(port)

    for pid in pids:
        if psutil is None:
            _safe_kill_pid(pid)
            continue
        try:
            proc = psutil.Process(pid)
        except Exception:
            continue
        for child in proc.children(recursive=True):
            _safe_kill(child)
        _safe_kill(proc)


def _pids_on_port_fallback(port: int) -> set[int]:
    import subprocess

    pids: set[int] = set()
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            for line in out.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        pids.add(int(parts[-1]))
        else:
            out = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            for line in out.splitlines():
                if line.strip().isdigit():
                    pids.add(int(line.strip()))
    except Exception:
        pass
    return pids


def stop_server() -> None:
    """Stop the running llama-server. Safe to call multiple times."""
    global _server_pid, _server_port

    port = _server_port or DEFAULT_PORT
    # Ask the daemon to stop the server it owns (and its monitor). A dead or
    # not-yet-started daemon is expected here, so fail fast instead of hanging.
    try:
        _send_to_daemon({"cmd": "STOP", "port": port}, retries=2)
    except Exception:
        pass

    # Belt-and-suspenders: reap anything still bound to the port, in case the
    # daemon is gone or missed it.
    _kill_servers_on_port(port)

    _server_pid = None
    _server_port = None
    _kill_hookpoint()


def server_url(port: int = DEFAULT_PORT, host: str = DEFAULT_HOST) -> str:
    return f"http://{host}:{port}"
