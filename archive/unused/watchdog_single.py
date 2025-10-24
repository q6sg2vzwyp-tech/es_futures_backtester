#!/usr/bin/env python3

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

# --- logging setup -----------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "watchdog_single.log"

logging.basicConfig(
    level=os.environ.get("WD_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [watchdog]: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# --- config (adjust as needed) ----------------------------------------------
REPO = Path(".").resolve()
PY_EXE = sys.executable  # safer than "python"
CHILD_CMD = [PY_EXE, "paper_trader.py"]  # extend with your CLI args below
CHILD_ARGS = os.environ.get("WD_ARGS", "").strip()
if CHILD_ARGS:
    # if WD_ARGS is a shell-like string, split safely
    CHILD_CMD += shlex.split(CHILD_ARGS)

LOG_CHILD_TO = LOG_DIR / "paper_trader_child.log"
CRASH_COOLDOWN_SEC = 5


def _powershell_path() -> str | None:
    """Resolve full path for PowerShell (B607)."""
    ps = shutil.which("powershell")
    if not ps:
        logger.warning("PowerShell not found in PATH; priority tweaks will be skipped.")
        return None
    return ps


def _bump_process_priority_windows() -> None:
    """
    Best-effort: try to nudge the process/IO priority on Windows using PowerShell.
    B404/B603/B607: we pass a list (shell=False) and a fully resolved exe path.
    """
    if os.name != "nt":
        return
    ps = _powershell_path()
    if not ps:
        return

    ps_script = r"""
$Process = Get-Process -Id $PID
$Process.PriorityClass = 'AboveNormal'
"""
    try:
        subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command", ps_script],
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as e:
        logger.debug("Priority tweak failed: %s", e)


def _iter_child_lines(p: subprocess.Popen) -> Iterable[str]:
    """Yield stdout lines from child safely, decoding errors as replacement."""
    assert p.stdout is not None  # for type-checkers
    for raw in p.stdout:
        # 'universal_newlines=True' + 'encoding="utf-8", errors="replace"' set below
        line = raw.rstrip("\r\n")
        if not line:
            continue
        yield line


def _parse_event(line: str) -> dict | None:
    """Parse a JSON event dict; tolerate non-JSON lines from child."""
    try:
        evt = json.loads(line)
    except json.JSONDecodeError:
        logger.debug("Skip non-JSON line from child: %s", line[:300])
        return None
    except Exception:
        logger.exception("Unexpected JSON parse error")
        return None
    if not isinstance(evt, dict):
        return None
    return evt


def run_once() -> int:
    """Start child, stream its output, mirror lines to file, parse events."""
    logger.info("Starting child: %s", " ".join(map(shlex.quote, CHILD_CMD)))
    try:
        logfh = open(LOG_CHILD_TO, "a", encoding="utf-8")
    except OSError as e:
        logger.error("Failed opening child log file: %s", e)
        return 1

    try:
        p = subprocess.Popen(
            CHILD_CMD,
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as e:
        logger.exception("Failed to start child process: %s", e)
        logfh.close()
        return 1

    rc = -1
    try:
        _bump_process_priority_windows()

        for line in _iter_child_lines(p):
            # Mirror to file
            try:
                logfh.write(line + "\n")
            except OSError as e:
                logger.debug("Child log write failed: %s", e)

            # Try structured event parse (optional)
            evt = _parse_event(line)
            if evt:
                # You can extend: route by evt["evt"] etc.
                pass

        rc = p.wait()
        logger.info("Child exited rc=%s", rc)
    finally:
        try:
            logfh.flush()
            logfh.close()
        except OSError as e:
            logger.debug("Closing child log failed: %s", e)

    return int(rc)


def main() -> int:
    # crash-loop guard
    while True:
        rc = run_once()
        # 0 means clean exit; >0 probably crash — cooldown
        if rc == 0:
            return 0
        logger.warning("Child crashed (rc=%s); sleeping %ss", rc, CRASH_COOLDOWN_SEC)
        time.sleep(CRASH_COOLDOWN_SEC)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Watchdog interrupted by user")
        sys.exit(130)
