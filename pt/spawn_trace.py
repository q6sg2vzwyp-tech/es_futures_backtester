from __future__ import annotations

from typing import Any, Dict, Optional
import json
import os
import threading
import traceback
from datetime import datetime

# Default output path (relative to repo root)
_SPAWN_TRACE_PATH = r".\run\spawn_trace.log"
_lock = threading.Lock()


def _now_ts() -> str:
    # match existing style: "YYYY-MM-DD HH:MM:SS"
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def spawn_trace_write(tag: str, **fields: Any) -> None:
    """Append a spawn/boot trace record as JSON line.

    Safe: never raises.
    Controlled by env:
      PT_SPAWN_TRACE=1 (enable)
      PT_SPAWN_TRACE_STACK=1 (include stack)
    """
    try:
        enable = os.environ.get("PT_SPAWN_TRACE", "").strip().lower() in ("1", "true", "yes", "on")
        if not enable:
            return

        rec: Dict[str, Any] = {
            "ts": _now_ts(),
            "tag": str(tag),
            "pid": os.getpid(),
            "ppid": _get_ppid_safe(),
            "exe": _get_exe_safe(),
            "argv": _get_argv_safe(),
            "cwd": os.getcwd(),
        }
        if fields:
            rec.update(fields)

        if os.environ.get("PT_SPAWN_TRACE_STACK", "").strip().lower() in ("1", "true", "yes", "on"):
            rec["stack"] = "".join(traceback.format_stack(limit=40))

        # ensure run/ exists
        d = os.path.dirname(_SPAWN_TRACE_PATH)
        if d:
            os.makedirs(d, exist_ok=True)

        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with _lock:
            with open(_SPAWN_TRACE_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        return


def spawn_trace_init(file_path: Optional[str] = None) -> None:
    """Optionally override output path."""
    global _SPAWN_TRACE_PATH
    try:
        if file_path:
            _SPAWN_TRACE_PATH = str(file_path)
    except Exception:
        return


def _get_ppid_safe() -> Optional[int]:
    try:
        return os.getppid()
    except Exception:
        return None


def _get_exe_safe() -> Optional[str]:
    try:
        return os.path.abspath(os.sys.executable)
    except Exception:
        return None


def _get_argv_safe():
    try:
        return list(os.sys.argv)
    except Exception:
        return None


# Backward-compatible aliases expected by paper_trader.py
_spawn_trace_write = spawn_trace_write
_spawn_trace_init = spawn_trace_init
