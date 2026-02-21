from __future__ import annotations

from typing import Any
import json
import os


def mkdirs(path: str) -> None:
    """Create directory path if non-empty (Windows-safe)."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def write_text_atomic(path: str, content: str, *, encoding: str = "utf-8") -> None:
    """Atomic text write: write to .tmp then os.replace()."""
    d = os.path.dirname(path)
    if d:
        mkdirs(d)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(content)
    os.replace(tmp, path)


def write_json_line_atomic(path: str, obj: Any, *, ensure_ascii: bool = False) -> None:
    """Write a single JSON line atomically (trailing newline)."""
    write_text_atomic(path, json.dumps(obj, ensure_ascii=ensure_ascii) + "\n")


def write_kv_atomic(path: str, payload: dict) -> None:
    """Write key=value lines atomically (trailing newline)."""
    lines = [f"{k}={v}" for k, v in payload.items()]
    write_text_atomic(path, "\n".join(lines) + "\n")
