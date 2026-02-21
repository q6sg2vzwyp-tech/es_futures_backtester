"""
patch_meta_localget.py

Fixes UnboundLocalError: meta referenced before assignment in _CLEAN_COPY/paper_trader.py

Patch:
  Replace occurrences of:
      meta=meta,
  with:
      meta=locals().get("meta"),

Usage (from repo root):
  .\.venv\Scripts\python.exe .\tools\patch_meta_localget.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "_CLEAN_COPY" / "paper_trader.py"

def main() -> int:
    if not TARGET.exists():
        print(f"[ERR] missing: {TARGET}")
        return 2

    src = TARGET.read_text(encoding="utf-8", errors="replace")

    needle = "meta=meta,"
    repl = 'meta=locals().get("meta"),'

    if needle not in src:
        print("[OK] no 'meta=meta,' found (nothing to patch)")
        return 0

    bak = TARGET.with_suffix(".py.bak_meta")
    if not bak.exists():
        bak.write_text(src, encoding="utf-8", newline="\n")
        print(f"[BAK] wrote {bak.name}")

    new = src.replace(needle, repl)
    TARGET.write_text(new, encoding="utf-8", newline="\n")

    changed = src.count(needle)
    print(f"[PATCH] replaced {changed} occurrence(s) in {TARGET}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
