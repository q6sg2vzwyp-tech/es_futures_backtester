# log_core.py
from __future__ import annotations

import csv
import os
import time
from typing import Dict, Any, List, Optional

DEFAULT_SHADOW_ROUNDTRIPS_LEDGER_PATH = os.path.join("results", "shadow_roundtrips_ledger.csv")

# Keep this stable. If you change it, you rotate old logs once.
SHADOW_FIELDS: List[str] = [
    "entry_ts",
    "exit_ts",
    "arm",
    "side",
    "entry_px",
    "exit_px",
    "pnl_usd",
    "R",
    "open_gate",
    "close_gate",
    "day",
    "week_R",
    "meta_ema_R",
    "regime",
]

# Your real trades.csv currently uses a different schema.
# Fix it to the same schema so all analytics and hb_monitor become simpler.
REAL_FIELDS: List[str] = SHADOW_FIELDS[:]  # identical on purpose


def _rotate(path: str, reason: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    rotated = f"{path}.ROTATED_{ts}"
    try:
        os.replace(path, rotated)
    except Exception:
        # fallback: best effort copy rename
        try:
            os.rename(path, rotated)
        except Exception:
            rotated = ""
    return rotated


def _read_header(path: str) -> Optional[List[str]]:
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f)
            h = next(r, None)
            return h
    except Exception:
        return None


def ensure_schema(path: str, fields: List[str], *, strict_rotate: bool = True) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
        return

    h = _read_header(path)
    if h is None:
        # unreadable -> rotate
        if strict_rotate:
            _rotate(path, "unreadable_header")
            with open(path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(fields)
        return

    # Normalize header (strip BOM oddities and whitespace)
    h_norm = [str(x).strip() for x in h]
    fields_norm = [str(x).strip() for x in fields]

    if h_norm != fields_norm:
        if strict_rotate:
            _rotate(path, "schema_mismatch")
            with open(path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(fields)
        else:
            # non-strict: do nothing
            return


def append_row(path: str, fields: List[str], row: Dict[str, Any]) -> None:
    # Always enforce schema before append
    ensure_schema(path, fields, strict_rotate=True)

    out = []
    for k in fields:
        v = row.get(k, "")
        out.append("" if v is None else v)

    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(out)
        f.flush()
        os.fsync(f.fileno())


def append_shadow_roundtrip(path: str, row: Dict[str, Any]) -> None:
    append_row(path, SHADOW_FIELDS, row)
    # ALSO append to immutable ledger
    append_row(DEFAULT_SHADOW_ROUNDTRIPS_LEDGER_PATH, SHADOW_FIELDS, row)


def append_real_roundtrip(path: str, row: Dict[str, Any]) -> None:
    append_row(path, REAL_FIELDS, row)
    # ALSO append to immutable ledger
    append_row(DEFAULT_SHADOW_ROUNDTRIPS_LEDGER_PATH, REAL_FIELDS, row)
