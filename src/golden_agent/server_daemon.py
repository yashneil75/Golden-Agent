"""Always-on background daemon that owns the llama-server lifecycle.

The golden-agent spawns a hookpoint sentinel (child of the agent) and then asks
this daemon, over a localhost control socket, to start the llama-server. The
daemon spawns the server plus a monitor subprocess that watches the hookpoint;
if the hookpoint dies (the agent is gone) the monitor kills the server so a
crashed or killed agent never leaves a model server holding weights in memory.

The daemon is fully detached from the agent (started with breakaway flags), so
it survives the agent dying and reliably reaps the orphaned server. It stays
running ("always on") to handle later sessions.
"""
import ctypes
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

STILL_ACTIVE = 259  # Windows exit code for a running process.


def _alive(pid: int) -> bool:
    if pid is None or pid <= 0:
        return False
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x0400, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _kill(pid: int) -> None:
    if pid is None or pid <= 0:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass


def _monitor_loop(hookpoint_pid: int, server_pid: int) -> None:
    """Runs in a subprocess owned by the daemon (not the agent).

    If the hookpoint (agent sentinel) goes away, kill the server. This is what
    reaps the orphaned model server after the agent crashes.
    """
    while True:
        time.sleep(2)
        if not _alive(hookpoint_pid):
            _kill(server_pid)
            return


def _serve(control_port: int, runtime_dir: Path) -> None:
    sessions: dict[int, tuple[subprocess.Popen, subprocess.Popen]] = {}
    daemon_script = str(Path(__file__).resolve())

    def _stop_session(port: int) -> None:
        server_proc, monitor_proc = sessions.pop(port, (None, None))
        if server_proc is not None and server_proc.poll() is None:
            _kill(server_proc.pid)
            try:
                server_proc.wait(timeout=3)
            except Exception:
                pass
        if monitor_proc is not None and monitor_proc.poll() is None:
            _kill(monitor_proc.pid)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", control_port))
        srv.listen(8)
        srv.settimeout(1.0)
        while True:
            # Reap finished sessions so we don't leak monitor processes.
            for port, (server_proc, monitor_proc) in list(sessions.items()):
                if server_proc.poll() is not None and monitor_proc.poll() is not None:
                    sessions.pop(port, None)

            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue

            with conn:
                try:
                    raw = conn.recv(65536).decode().strip()
                    payload = json.loads(raw)
                except Exception:
                    continue

                cmd = payload.get("cmd")
                if cmd == "PING":
                    conn.sendall(b'{"status":"ok"}\n')
                elif cmd == "START":
                    port = int(payload["port"])
                    hookpoint_pid = int(payload["hookpoint_pid"])
                    server_args = payload["server_args"]
                    _stop_session(port)
                    debug = bool(os.environ.get("GOLDEN_AGENT_DEBUG"))
                    server_proc = subprocess.Popen(
                        server_args,
                        stdin=subprocess.DEVNULL,
                        stdout=None if debug else subprocess.DEVNULL,
                        stderr=None if debug else subprocess.DEVNULL,
                    )
                    monitor_proc = subprocess.Popen(
                        [
                            sys.executable,
                            daemon_script,
                            "monitor",
                            "--hookpoint-pid",
                            str(hookpoint_pid),
                            "--server-pid",
                            str(server_proc.pid),
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    sessions[port] = (server_proc, monitor_proc)
                    conn.sendall(
                        json.dumps(
                            {"status": "ok", "server_pid": server_proc.pid}
                        ).encode()
                        + b"\n"
                    )
                elif cmd == "STOP":
                    port = int(payload.get("port", 0)) or None
                    if port:
                        _stop_session(port)
                    else:
                        for p in list(sessions):
                            _stop_session(p)
                    conn.sendall(b'{"status":"ok"}\n')


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("mode", nargs="?", default="daemon")
    args = parser.parse_args()

    if args.mode == "monitor":
        import argparse as _ap

        mp = _ap.ArgumentParser()
        mp.add_argument("--hookpoint-pid", type=int, required=True)
        mp.add_argument("--server-pid", type=int, required=True)
        m = mp.parse_args(sys.argv[2:])
        _monitor_loop(m.hookpoint_pid, m.server_pid)
        return

    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    pid_file = runtime_dir / "daemon.pid"

    if pid_file.exists():
        try:
            old = int(pid_file.read_text().strip())
        except Exception:
            old = None
        if old and _alive(old):
            # Another daemon already owns this runtime; defer to it.
            sys.exit(0)

    pid_file.write_text(str(os.getpid()))
    try:
        _serve(args.control_port, runtime_dir)
    finally:
        try:
            pid_file.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
