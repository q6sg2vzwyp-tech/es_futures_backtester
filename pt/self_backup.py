from __future__ import annotations

from typing import Callable, Optional
import hashlib
import os
import shutil
from datetime import datetime


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def self_backup(
    src_file: str,
    *,
    log_fn: Optional[Callable[..., object]] = None,
    backups_dir: str = r".\backups",
) -> str:
    """Copy the running module file into backups/ with a timestamped name.

    Returns the absolute path to the backup file (or empty string on failure).

    Expected log event (if log_fn is provided):
        log_fn("self_backup_ok", path="<abs path>")
    """
    try:
        src_abs = os.path.abspath(src_file)
        if not os.path.isfile(src_abs):
            return ""
        os.makedirs(backups_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(src_abs))[0]
        dst_name = f"{base}_{ts}.py"
        dst_abs = os.path.abspath(os.path.join(backups_dir, dst_name))

        shutil.copy2(src_abs, dst_abs)

        # Optional integrity stamp: write sidecar .sha1 (non-fatal)
        try:
            sha1 = _sha1_file(dst_abs)
            with open(dst_abs + ".sha1", "w", encoding="utf-8") as f:
                f.write(sha1 + "\n")
        except Exception:
            pass

        try:
            if log_fn is not None:
                log_fn("self_backup_ok", path=dst_abs)
        except Exception:
            pass

        return dst_abs
    except Exception:
        return ""
