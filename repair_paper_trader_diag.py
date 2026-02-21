#!/usr/bin/env python3
"""repair_paper_trader_diag.py

Repairs paper_trader.py after a mangled one-line try/except diagnostic injection.
- Removes any single-line 'try: ... diag_* ... except ... pass' injections.
- Ensures `from __future__ import annotations` is at the very top (after optional shebang/encoding).
- Re-inserts guarded diagnostic log blocks:
    evt=diag_connect_target (after parse_args)
    evt=diag_ib_connect_call (immediately before ib.connect)
Creates a timestamped backup in tools/patches_quarantine_YYYYMMDD_HHMMSS.

Usage:
  python repair_paper_trader_diag.py --file paper_trader.py
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

BAD_KEYS = ("diag_ib_connect_call", "diag_connect_target")

def ts_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def backup(src: Path, root: Path) -> Path:
    qdir = root / "tools" / f"patches_quarantine_{ts_tag()}"
    qdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, qdir / src.name)
    return qdir

def is_bad_one_liner(line: str) -> bool:
    if "try:" in line and "except" in line and "pass" in line:
        return any(k in line for k in BAD_KEYS)
    return False

def ensure_future_import_top(lines: list[str]) -> list[str]:
    future = "from __future__ import annotations"
    idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == future:
            idx = i
            break
    if idx is None:
        return lines

    ln_future = lines.pop(idx).rstrip("\n") + "\n"

    insert_at = 0
    # keep shebang
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    # keep encoding comment (PEP 263) if present in first two lines
    for i in range(min(2, len(lines))):
        if re.search(r"coding[:=]\s*[-\w.]+", lines[i]):
            insert_at = max(insert_at, i + 1)

    # place future import immediately after shebang/encoding (before any other imports)
    lines.insert(insert_at, ln_future)
    # ensure exactly one blank line after future import for readability
    if insert_at + 1 < len(lines) and lines[insert_at + 1].strip() != "":
        lines.insert(insert_at + 1, "\n")
    return lines

def insert_diag_after_parse_args(lines: list[str]) -> list[str]:
    if any("diag_connect_target" in ln for ln in lines):
        return lines
    for i, ln in enumerate(lines):
        if "parse_args(" in ln and "=" in ln:
            indent = re.match(r"^(\s*)", ln).group(1)
            block = [
                f"{indent}try:\n",
                f"{indent}    log({{'evt':'diag_connect_target','host':getattr(args,'host',None),'port':getattr(args,'port',None),'clientId':getattr(args,'clientId',None)}})\n",
                f"{indent}except Exception:\n",
                f"{indent}    pass\n",
            ]
            lines[i+1:i+1] = block
            break
    return lines

def insert_diag_before_ib_connect(lines: list[str]) -> list[str]:
    if any("diag_ib_connect_call" in ln for ln in lines):
        return lines
    for i, ln in enumerate(lines):
        if "ib.connect(" in ln:
            indent = re.match(r"^(\s*)", ln).group(1)
            block = [
                f"{indent}try:\n",
                f"{indent}    log({{'evt':'diag_ib_connect_call','host':args.host,'port':args.port,'cid':cid,'timeout':args.connect_timeout_sec}})\n",
                f"{indent}except Exception:\n",
                f"{indent}    pass\n",
            ]
            lines[i:i] = block
            break
    return lines

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="paper_trader.py")
    ns = ap.parse_args()

    src = Path(ns.file).resolve()
    if not src.exists():
        print(f"[ERR] file not found: {src}")
        return 2
    root = src.parent

    qdir = backup(src, root)

    raw = src.read_text(encoding="utf-8", errors="replace").splitlines(True)
    cleaned = [ln for ln in raw if not is_bad_one_liner(ln)]

    cleaned = ensure_future_import_top(cleaned)
    cleaned = insert_diag_after_parse_args(cleaned)
    cleaned = insert_diag_before_ib_connect(cleaned)

    src.write_text("".join(cleaned), encoding="utf-8")

    print("[PATCH] Repaired paper_trader.py (removed mangled diag one-liners, fixed future import position, reinserted safe diag blocks).") 
    print(f"[PATCH] Backup saved to: {qdir}")
    print("[PATCH] Verify:")
    print(r"  .\\.venv\\Scripts\\python.exe -c \"import py_compile; py_compile.compile('paper_trader.py', doraise=True); print('paper_trader OK')\"")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
