#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Simple watchdog runner for paper_trader.py

- Prefers .\.venv\Scripts\python.exe, falls back to "python" on PATH
- Reads PAPER_TRADER_ARGS from the environment (if set); otherwise uses CLI trailing args after "--"
- Creates a timestamped logfile under .\logs\
- Optional:
    --once (do not restart on exit)
    --sleep-sec <n> delay between restarts
    --cwd / --target overrides
    --tee mirror child stdout to console & file (text mode; no buffering warnings)
    --client-id-base <n> and --client-id-offset <k> → adds --clientId n+k (removes any existing --clientId)
"""

import argparse
import os
import sys
import subprocess
import time
from datetime import datetime, timezone
from shlex import split as shsplit
import re

def ts() -> str:
    # timezone-aware UTC timestamp
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def guess_python() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.join(here, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py):
        return venv_py
    return "python"

def strip_existing_client_id(arg_list):
    """
    Remove any existing '--clientId <value>' or '--clientId=<value>' from the list.
    Returns a new list.
    """
    out = []
    skip_next = False
    for i, tok in enumerate(arg_list):
        if skip_next:
            skip_next = False
            continue
        if tok == "--clientId":
            # Skip this and the next token (value)
            skip_next = True
            continue
        if tok.startswith("--clientId="):
            # Skip this single token
            continue
        out.append(tok)
    return out

def add_client_id(arg_list, base: int | None, offset: int | None):
    """
    If base or offset is provided, compute base+offset (treat None as 0) and append '--clientId <n>'.
    """
    if base is None and offset is None:
        return arg_list
    b = int(base or 0)
    o = int(offset or 0)
    n = b + o
    return arg_list + ["--clientId", str(n)]

def build_child_cmd(python_exe: str, target: str, env_args: str, trailing: list[str],
                    client_id_base: int | None, client_id_offset: int | None) -> list[str]:
    # Start with interpreter + script
    cmd = [python_exe, "-u", target]

    # Merge env-first, then trailing; both may be empty
    merged = []
    if env_args:
        merged += shsplit(env_args)
    if trailing:
        merged += trailing

    # Strip any existing --clientId
    merged = strip_existing_client_id(merged)

    # Add computed --clientId if requested
    merged = add_client_id(merged, client_id_base, client_id_offset)

    return cmd + merged

def pump_process_output(proc: subprocess.Popen, logfile_path: str, tee_to_console: bool) -> int:
    """
    Stream child's stdout to logfile (and optionally console).
    The child must be started with text=True for line buffering to work cleanly.
    """
    # Ensure the file exists and is opened line-buffered in text mode
    with open(logfile_path, "a", encoding="utf-8", buffering=1) as lf:
        with proc.stdout:
            for line in proc.stdout:
                lf.write(line)
                if tee_to_console:
                    print(line, end="", flush=True)
    return proc.wait()

def main():
    parser = argparse.ArgumentParser(
        description="Watchdog wrapper for paper_trader.py (env-first args, optional restarts/tee/clientId injection)."
    )
    here = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--cwd", default=here, help="Working directory for the child process.")
    parser.add_argument("--target", default=os.path.join(here, "paper_trader.py"),
                        help="Path to paper_trader.py")
    parser.add_argument("--python", default=guess_python(), help="Python interpreter to use.")
    parser.add_argument("--once", action="store_true", help="Run once; do not restart on exit.")
    parser.add_argument("--sleep-sec", type=float, default=3.0, help="Delay between restarts.")
    parser.add_argument("--logdir", default=os.path.join(".", "logs"),
                        help=r"Directory for logs (default: .\logs)")
    parser.add_argument("--tee", action="store_true",
                        help="Mirror child stdout to console (and always log to file).")
    parser.add_argument("--client-id-base", type=int, default=None,
                        help="Base clientId; combined with --client-id-offset to compute --clientId.")
    parser.add_argument("--client-id-offset", type=int, default=None,
                        help="Offset to add to --client-id-base to compute --clientId.")

    # Everything after '--' goes straight to paper_trader
    parser.add_argument("remainder", nargs=argparse.REMAINDER,
                        help="Optional: use '--' then extra args to pass to paper_trader directly.")

    args = parser.parse_args()

    # PAPER_TRADER_ARGS (env-first)
    env_args = os.environ.get("PAPER_TRADER_ARGS", "").strip()
    using_env = bool(env_args)

    # Trailing args after '--'
    trailing = []
    if args.remainder and args.remainder[0] == "--":
        trailing = args.remainder[1:]

    print("[WD] Using PAPER_TRADER_ARGS from environment" if using_env else "[WD] No PAPER_TRADER_ARGS in env")
    print("[WD] Starting watchdog...")
    print(f"[WD] Python : {args.python}")
    print(f"[WD] CWD    : {args.cwd}")
    print(f"[WD] Target : {args.target}")
    print(f"[WD] Args   : {'from ENV' if using_env else ('from CLI trailing' if trailing else 'none')}")

    os.makedirs(args.logdir, exist_ok=True)

    while True:
        log_path = os.path.join(args.logdir, f"run_{ts()}.log")
        print(f"[WD] Logging to: {log_path}")

        # Build command
        cmd = build_child_cmd(
            python_exe=args.python,
            target=args.target,
            env_args=env_args,
            trailing=trailing,
            client_id_base=args.client_id_base,
            client_id_offset=args.client_id_offset,
        )

        # Human-friendly echo (joined)
        print("[WD] Launching:", " ".join(cmd))

        if args.tee:
            # Text-mode streaming to avoid buffering warnings
            try:
                p = subprocess.Popen(
                    cmd,
                    cwd=args.cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,           # text mode so we iterate strings
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1            # line-buffered (OK in text mode)
                )
                rc = pump_process_output(p, log_path, tee_to_console=True)
            except KeyboardInterrupt:
                print("[WD] Ctrl-C received; exiting.")
                return
            except Exception as e:
                print(f"[WD] Launch failed: {e}")
                rc = -1
        else:
            # File-only logging (fast path)
            try:
                with open(log_path, "a", encoding="utf-8") as lf:
                    p = subprocess.Popen(
                        cmd,
                        cwd=args.cwd,
                        stdout=lf,
                        stderr=lf,
                    )
                    rc = p.wait()
            except KeyboardInterrupt:
                print("[WD] Ctrl-C received; exiting.")
                return
            except Exception as e:
                print(f"[WD] Launch failed: {e}")
                rc = -1

        # Exit or restart
        if args.once:
            return
        print(f"[WD] Child exited rc={rc}; restarting in {args.sleep_sec:.1f}s...")
        time.sleep(max(0.0, args.sleep_sec))

if __name__ == "__main__":
    main()
