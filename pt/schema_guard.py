# schema_guard.py
# Minimal schema validation utilities (compatibility module)
# Keeps older/legacy imports working and centralizes CSV schema checks.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Tuple


class SchemaError(RuntimeError):
    pass


def _read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def assert_header(path: str | Path, expected: Iterable[str], *, allow_empty_file: bool = True) -> Tuple[bool, str]:
    """
    Validate that the CSV file at `path` has the expected header row.

    Returns: (ok, message)
    - If file doesn't exist: ok=False
    - If file exists but is empty:
        - ok=True if allow_empty_file else ok=False
    - If header mismatches: ok=False
    """
    p = Path(path)
    exp = list(expected)

    if not p.exists():
        return False, f"missing file: {p}"

    hdr = _read_header(p)
    if not hdr:
        if allow_empty_file:
            return True, f"empty file (allowed): {p}"
        return False, f"empty file (not allowed): {p}"

    if hdr != exp:
        return False, f"header mismatch: {p} expected={exp} got={hdr}"

    return True, f"OK: {p}"


def assert_trades_and_shadow(trades_csv: str | Path, shadow_csv: str | Path) -> Tuple[bool, str]:
    """
    Convenience check used by the paper trader startup/self-test:
    - trades.csv expected 8 cols
    - shadow_roundtrips.csv expected 14 cols
    """
    trades_expected = ["timestamp", "side", "qty", "entry_px", "exit_px", "pnl", "R", "tags"]
    shadow_expected = [
        "trade_id", "ts_open", "ts_close", "arm", "side", "qty",
        "entry_px", "exit_px", "pnl", "R", "reason_open", "reason_close",
        "gate_at_open", "gate_at_close"
    ]

    ok1, msg1 = assert_header(trades_csv, trades_expected, allow_empty_file=True)
    ok2, msg2 = assert_header(shadow_csv, shadow_expected, allow_empty_file=True)

    if ok1 and ok2:
        return True, "trades.csv=OK (8 cols), shadow_roundtrips.csv=OK (14 cols)"
    return False, f"{msg1} | {msg2}"
