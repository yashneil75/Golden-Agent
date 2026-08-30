"""Hookpoint sentinel spawned as a direct child of the golden-agent.

Its only job is to exist while the agent is alive. The server daemon's monitor
watches this process; when it disappears (agent crashed / killed), the daemon
reaps the orphaned llama-server. Because this is a child of the agent it is
reaped with the agent via the job object, and as a belt-and-suspenders check it
also exits on its own if it can no longer see the agent process.
"""
import ctypes
import os
import sys
import time

STILL_ACTIVE = 259  # Windows exit code for a running process.


def _agent_alive(pid: int) -> bool:
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


def main() -> None:
    agent_pid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    while True:
        if agent_pid is not None and not _agent_alive(agent_pid):
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
