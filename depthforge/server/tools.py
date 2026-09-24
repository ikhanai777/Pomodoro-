"""Locating and running external command-line tools (COLMAP, OpenMVS)."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

IS_WIN = sys.platform.startswith("win")


class Cancelled(Exception):
    pass


class ToolError(RuntimeError):
    pass


def _exe_names(name: str) -> list[str]:
    return [name + ".exe", name] if IS_WIN else [name]


def find_tool(name: str, hint: str = "", tools_dir: Optional[Path] = None) -> Optional[Path]:
    """Find an executable by name: an explicit path/folder hint, then tools_dir, then PATH."""
    names = _exe_names(name)
    roots: list[Path] = []
    if hint:
        p = Path(hint).expanduser()
        if p.is_file():
            return p
        roots.append(p)
    if tools_dir:
        roots.append(Path(tools_dir))
    for root in roots:
        if not root.is_dir():
            continue
        for n in names:
            direct = root / n
            if direct.is_file():
                return direct
            for hit in sorted(root.rglob(n)):
                if hit.is_file() and (IS_WIN or os.access(hit, os.X_OK)):
                    return hit
    for n in names:
        found = shutil.which(n)
        if found:
            return Path(found)
    return None


def tool_env(exe: Path) -> dict:
    """Environment for a tool: its sibling lib/ folders on PATH (Windows COLMAP zips need this), no GUI."""
    env = os.environ.copy()
    extra = [str(exe.parent)]
    for lib in (exe.parent / "lib", exe.parent.parent / "lib"):
        if lib.is_dir():
            extra.append(str(lib))
            env["QT_PLUGIN_PATH"] = str(lib) + os.pathsep + env.get("QT_PLUGIN_PATH", "")
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env


@lru_cache(maxsize=64)
def help_text(exe: str, *args: str) -> str:
    """Help output of a tool, used to detect which flags this version supports."""
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=60,
                           env=tool_env(Path(exe)), errors="replace")
        return (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        proc.kill()


def run(cmd: list, *, cwd: Path, log: Callable[[str], None], cancel: threading.Event,
        on_line: Optional[Callable[[str], None]] = None) -> None:
    """Run a tool, streaming its output to `log`. Raises Cancelled or ToolError."""
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd))
    kwargs = dict(cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                  env=tool_env(Path(cmd[0])), text=True, errors="replace", bufsize=1)
    if IS_WIN:
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    watcher_done = threading.Event()

    def watch():
        while not watcher_done.wait(0.5):
            if cancel.is_set():
                _kill_tree(proc)
                return
    threading.Thread(target=watch, daemon=True).start()
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                log(line)
                if on_line:
                    on_line(line)
        proc.wait()
    finally:
        watcher_done.set()
    if cancel.is_set():
        raise Cancelled()
    if proc.returncode != 0:
        raise ToolError(f"{Path(cmd[0]).name} failed with exit code {proc.returncode}")
