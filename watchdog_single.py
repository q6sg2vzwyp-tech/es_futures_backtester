#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple ES paper trader watchdog v2

- Uses the same Python interpreter that runs this script (sys.executable)
- Always runs paper_trader.py in the same directory as this file
- Optionally appends args from a text file (single.cmdline.txt by default)
- Restarts child on crash/unexpected exit with exponential backoff
- Monitors child's stdout for JSON lines with {"evt":"hb"} heartbeats
  and restarts if no heartbeat within --hb-timeout-sec
- Logs to:
    ./logs/watchdog/YYYYMMDD_watchdog.log
    ./logs/child/YYYYMMDD_child.log
"""

from __future__ import annotations
import sys
import os
import time
import json
import argparse
import subprocess
import threading
import queue
import datetime as dt
import shlex
from typing import Optional  # <-- for Python 3.9-safe Optional[float]

# ---------- Paths / logs ----------

def mkdirs(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_paths():
    base = os.path.abspath(os.path.dirname(__file__) or ".")
    wdir = os.path.join(base, "logs", "watchdog")
    cdir = os.path.join(base, "logs", "child")
    mkdirs(wdir)
    mkdirs(cdir)
    stamp = dt.datetime.now().strftime("%Y%m%d")
    wlog = os.path.join(wdir, f"{stamp}_watchdog.log")
    clog = os.path.join(cdir, f"{stamp}_child.log")
    return wlog, clog

def log_write(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")

# ---------- Arg parsing ----------

def build_parser():
    ap = argparse.ArgumentParser(
        description="Resilient watchdog for paper_trader.py (ES bot)"
    )
    ap.add_argument(
        "--args-file",
        default="single.cmdline.txt",
        help="Text file with args for paper_trader.py (relative to this script or absolute).",
    )
    ap.add_argument(
        "--env-file",
        help="Optional KEY=VALUE lines added to child environment.",
    )
    ap.add_argument(
        "--min-backoff",
        type=float,
        default=2.0,
        help="Min restart backoff seconds.",
    )
    ap.add_argument(
        "--max-backoff",
        type=float,
        default=60.0,
        help="Max restart backoff seconds.",
    )
    ap.add_argument(
        "--hb-timeout-sec",
        type=int,
        default=90,
        help="If no heartbeats in this window, restart.",
    )
    ap.add_argument(
        "--print-child",
        action="store_true",
        help="Also print child output to console.",
    )
    return ap

# ---------- Utilities ----------

def load_args_file(path: str) -> list[str]:
    if not path:
        return []
    if not os.path.isabs(path):
        base = os.path.abspath(os.path.dirname(__file__) or ".")
        path = os.path.join(base, path)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = " ".join(line.strip() for line in f if line.strip())
    return shlex.split(content)

def load_env_file(path: str) -> dict:
    env: dict[str, str] = {}
    if not path:
        return env
    if not os.path.isabs(path):
        base = os.path.abspath(os.path.dirname(__file__) or ".")
        path = os.path.join(base, path)
    if not os.path.exists(path):
        return env
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            env[k.strip()] = v.strip()
    return env

# ---------- Child IO reader ----------

class StreamReader(threading.Thread):
    def __init__(self, pipe, q: queue.Queue, also_print: bool,
                 child_log_path: str, is_err: bool):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.q = q
        self.also_print = also_print
        self.child_log_path = child_log_path
        self.is_err = is_err

    def run(self):
        try:
            for raw in iter(self.pipe.readline, b""):
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                except Exception:
                    line = raw.decode("latin-1", errors="replace").rstrip("\n")
                prefix = "ERR" if self.is_err else "OUT"
                log_write(self.child_log_path, f"[{ts()}] {prefix} {line}")
                if self.also_print:
                    print(line)
                self.q.put(line)
        finally:
            try:
                self.pipe.close()
            except Exception:
                pass

# ---------- Main watchdog loop ----------

def main():
    wlog, clog = log_paths()
    ap = build_parser()
    args = ap.parse_args()

    script_dir = os.path.abspath(os.path.dirname(__file__) or ".")
    trader = os.path.join(script_dir, "paper_trader.py")

    print("[WD] simple watchdog v2 starting...")
    print(f"[WD] script_dir: {script_dir}")
    print(f"[WD] trader    : {trader}")

    if not os.path.exists(trader):
        msg = f"[ERROR] paper_trader.py not found at: {trader}"
        print(msg)
        log_write(wlog, f"[{ts()}] {msg}")
        sys.exit(1)

    child_python = sys.executable  # same interpreter that runs this script

    trader_args = load_args_file(args.args_file)
    env = os.environ.copy()
    env.update(load_env_file(args.env_file))

    min_backoff = max(0.5, float(args.min_backoff))
    max_backoff = max(min_backoff, float(args.max_backoff))
    backoff = min_backoff

    hb_timeout = int(args.hb_timeout_sec)
    last_hb: Optional[float] = None  # <-- 3.9-safe union

    child_cwd = script_dir

    print(f"[WD] logging to: {wlog}")
    print(f"[WD] child log : {clog}")
    print(f"[WD] child cmd : {child_python} {trader} {' '.join(trader_args)}")
    log_write(
        wlog,
        f"[{ts()}] WATCHDOG START python={child_python} trader={trader} args={' '.join(trader_args)} cwd={child_cwd}",
    )

    try:
        while True:
            try:
                target_cmd = [child_python, trader] + trader_args
                log_write(
                    wlog,
                    f"[{ts()}] LAUNCH backoff={backoff:.1f}s cmd={' '.join(target_cmd)}",
                )

                qlines: queue.Queue = queue.Queue()
                proc = subprocess.Popen(
                    target_cmd,
                    cwd=child_cwd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                )

                t_out = StreamReader(proc.stdout, qlines, args.print_child, clog, is_err=False)
                t_err = StreamReader(proc.stderr, qlines, args.print_child, clog, is_err=True)
                t_out.start()
                t_err.start()

                last_hb = time.time()
                start_t = time.time()

                while True:
                    # drain lines + detect heartbeats
                    try:
                        while True:
                            line = qlines.get_nowait()
                            s = line.strip()
                            jstart = s.find("{")
                            if jstart >= 0 and s.endswith("}"):
                                try:
                                    obj = json.loads(s[jstart:])
                                    if obj.get("evt") == "hb":
                                        last_hb = time.time()
                                except Exception:
                                    pass
                    except queue.Empty:
                        pass

                    # heartbeat timeout
                    if hb_timeout > 0 and last_hb is not None:
                        if (time.time() - last_hb) > hb_timeout:
                            msg = f"HB_TIMEOUT after {hb_timeout}s — killing child PID={proc.pid}"
                            log_write(wlog, f"[{ts()}] {msg}")
                            print("[WD]", msg)
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
                            break  # out of inner loop → restart

                    rc = proc.poll()
                    if rc is not None:
                        runtime = time.time() - start_t
                        log_write(
                            wlog,
                            f"[{ts()}] EXIT rc={rc} runtime={runtime:.1f}s",
                        )
                        # adjust backoff
                        if runtime < 10:
                            backoff = min(max_backoff, backoff * 2.0)
                        else:
                            backoff = max(min_backoff, backoff * 0.8)
                        break

                    time.sleep(0.5)

            except FileNotFoundError as e:
                msg = f"ERROR: executable or script not found. Details: {e!r}"
                print(msg)
                log_write(wlog, f"[{ts()}] {msg}")
                time.sleep(3)
            except Exception as e:
                msg = f"EXC: {e!r}"
                print("[WD]", msg)
                log_write(wlog, f"[{ts()}] {msg}")
                time.sleep(3)

            log_write(wlog, f"[{ts()}] RESTART in {backoff:.1f}s")
            print(f"[WD] Restarting in {backoff:.1f}s ...")
            time.sleep(backoff)

    except KeyboardInterrupt:
        log_write(wlog, f"[{ts()}] WATCHDOG STOP (Ctrl+C)")
        print("[WD] Stopping…")


if __name__ == "__main__":
    main()
