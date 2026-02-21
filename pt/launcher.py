#!/usr/bin/env python3
# launcher.py — robust launcher (venv-first, clears shutdown/locks, lockfile-based singleton)

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


LAUNCHER_VERSION = "v5.1 (venv-first + clear flags/locks + tolerant watchdog PID parsing)"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _pick_python(root: Path) -> Path:
    """
    Prefer the project's venv python. Fall back to the python running launcher.
    """
    venv_py = root / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return venv_py.resolve()
    return Path(sys.executable).resolve()


def _pid_is_running_windows(pid: int) -> bool:
    if os.name != "nt" or pid <= 0:
        return False
    try:
        out = subprocess.check_output(
            ["cmd", "/c", "tasklist", "/FI", f"PID eq {pid}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return str(pid) in out
    except Exception:
        return False


def _read_pid_from_text(txt: str) -> int:
    """
    Support both formats:
      - plain integer: "47496"
      - payload: "pid=47496 ts=..." (watchdog_single.py lock payload)
    """
    txt = (txt or "").strip()
    if not txt:
        return -1

    m = re.search(r"\bpid\s*=\s*(\d+)\b", txt)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1

    try:
        return int(txt)
    except Exception:
        return -1


def _watchdog_running(wd_lock: Path, wd_pidlock: Path) -> bool:
    """
    Determine whether watchdog is already running.

    We check BOTH:
      - run/watchdog_single.lock  (may contain payload like 'pid=123 ts=...')
      - run/watchdog_single.pidlock (plain integer PID, if present)

    If either contains a live PID -> treat as running.
    """
    # 1) pidlock (preferred if present)
    if wd_pidlock.exists():
        try:
            pid = _read_pid_from_text(wd_pidlock.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pid = -1
        if _pid_is_running_windows(pid):
            return True

    # 2) lock payload
    if wd_lock.exists():
        try:
            pid = _read_pid_from_text(wd_lock.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pid = -1
        if _pid_is_running_windows(pid):
            return True

    return False


def _safe_remove(p: Path) -> None:
    try:
        p.unlink()
    except Exception:
        pass


def main() -> int:
    root = _repo_root()

    ap = argparse.ArgumentParser()
    ap.add_argument("--args-file", default=str(root / "single.cmdline.txt"))
    ap.add_argument("--hb-path", default=str(root / "run" / "heartbeat.txt"))
    ap.add_argument("--no-hb-monitor", action="store_true")
    args = ap.parse_args()

    # Ensure cwd is repo root (shortcuts can have wrong "Start in")
    try:
        os.chdir(str(root))
    except Exception:
        pass

    run_dir = root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    python_exe = _pick_python(root)
    wd = root / "watchdog_single.py"

    shutdown_flag = run_dir / "SHUTDOWN.flag"
    wd_lock = run_dir / "watchdog_single.lock"
    wd_pidlock = run_dir / "watchdog_single.pidlock"
    pt_lock = run_dir / "paper_trader.lock"
    pt_pidlock = run_dir / "paper_trader.pidlock"
    hb_lock = run_dir / "hb_monitor.lock"

    print(f"[LAUNCHER] {LAUNCHER_VERSION} pid={os.getpid()}")
    print(f"[LAUNCHER] CWD     : {Path.cwd()}")
    print(f"[LAUNCHER] Python  : {python_exe}")
    print(f"[LAUNCHER] Watchdog: {wd}")

    # 1) Clear shutdown + stale *non-watchdog* locks (safe/idempotent)
    _safe_remove(shutdown_flag)
    _safe_remove(pt_lock)
    _safe_remove(pt_pidlock)
    _safe_remove(hb_lock)

    # 2) If watchdog is running, hard block
    if _watchdog_running(wd_lock, wd_pidlock):
        print("[LAUNCHER] Watchdog already running (lock/pidlock + live PID). Exit rc=2.")
        return 2

    # If watchdog artifacts exist but PID is dead, remove them so launch won't be blocked
    _safe_remove(wd_lock)
    _safe_remove(wd_pidlock)

    # 3) Launch watchdog
    cmd = [
        str(python_exe),
        str(wd),
        "--args-file", str(Path(args.args_file).resolve()),
        "--hb-path", str(Path(args.hb_path).resolve()),
    ]
    if args.no_hb_monitor:
        cmd.append("--no-hb-monitor")

    print("[LAUNCHER] Launching watchdog...")

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE

    subprocess.Popen(cmd, cwd=str(root), creationflags=creationflags)

    # Give watchdog time to create its lock/pidlock
    time.sleep(1.5)

    if _watchdog_running(wd_lock, wd_pidlock):
        print("[LAUNCHER] Watchdog launched successfully. Exiting launcher.")
        return 0

    print("[LAUNCHER] Watchdog did not create lock (or exited immediately). "
          "Check run\\paper_trader.child.err.log and run\\watchdog_detached.err.log.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

