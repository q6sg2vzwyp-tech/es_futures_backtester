#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import os
import sys
import datetime as dt
from typing import Dict, Any, List, Optional


# -----------------------------
# Canonical schemas (ONE truth)
# -----------------------------

TRADE_FIELDNAMES: List[str] = [
    "ts",
    "arm",
    "side",
    "prev_px",
    "last_px",
    "shadow_pnl_usd",
    "shadow_R",
    "gate_reason",
    "caps",
    "day_R",
    "week_R",
    "meta_ema_R",
]

ROUNDTRIP_FIELDNAMES: List[str] = [
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


# -----------------------------
# Legacy compatibility (known bad / old schemas)
# -----------------------------
# Your observed legacy data rows look like:
#   entry_ts, day, regime, arm, side, qty, entry_px, exit_px, pnl_usd, R, close_gate
LEGACY_ROUNDTRIP_FIELDS_V11: List[str] = [
    "entry_ts",
    "day",
    "regime",
    "arm",
    "side",
    "qty",
    "entry_px",
    "exit_px",
    "pnl_usd",
    "R",
    "close_gate",
]

# A conservative set of known headers we can migrate safely
KNOWN_LEGACY_ROUNDTRIP_HEADERS: List[List[str]] = [
    LEGACY_ROUNDTRIP_FIELDS_V11,
]

# Map legacy -> canonical (anything not present becomes "")
LEGACY_ROUNDTRIP_MAP_V11_TO_CANON: Dict[str, str] = {
    "entry_ts": "entry_ts",
    # legacy file often only had a date for "day", and not a real exit timestamp
    # we will map "day" into canonical "day" and leave exit_ts blank unless present
    "day": "day",
    "regime": "regime",
    "arm": "arm",
    "side": "side",
    "entry_px": "entry_px",
    "exit_px": "exit_px",
    "pnl_usd": "pnl_usd",
    "R": "R",
    "close_gate": "close_gate",
    # open_gate/week_R/meta_ema_R/exit_ts will default ""
}

# Optional: legacy trade schema variants could be added similarly if needed.
KNOWN_LEGACY_TRADE_HEADERS: List[List[str]] = []


def _expected_header_line(fieldnames: List[str]) -> str:
    return ",".join(fieldnames)


def _read_header_fields(path: str) -> List[str]:
    """
    Read CSV header fields robustly (handles UTF-8 BOM, whitespace).
    Returns [] if file has no header/empty.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        hdr = next(r, [])
    return [h.strip() for h in hdr if h is not None and str(h).strip()]


def _peek_first_data_row_len(path: str) -> Optional[int]:
    """
    Return number of columns in first non-empty data row (after header), or None if no data rows.
    Uses csv.reader to measure raw column count (detects row-width mismatches).
    """
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f)
            _ = next(r, None)  # header
            for row in r:
                if row and any(str(x).strip() for x in row):
                    return len(row)
    except Exception:
        return None
    return None


def _migrate_subset_schema_in_place(path: str, old_fields: List[str], new_fields: List[str]) -> None:
    """
    Migrate a CSV whose header is an ordered prefix of new_fields.

    Rewrites the file in-place (atomic replace) with the new header and
    preserves existing data by padding missing columns with "".
    """
    tmp_path = path + ".tmp"

    with open(path, "r", encoding="utf-8-sig", newline="") as fin, open(
        tmp_path, "w", encoding="utf-8", newline=""
    ) as fout:
        rin = csv.DictReader(fin)
        read_fields = [h.strip() for h in (rin.fieldnames or [])]
        if read_fields != old_fields:
            raise RuntimeError(f"subset migrate: header changed during read: {read_fields} vs {old_fields}")

        wout = csv.DictWriter(fout, fieldnames=new_fields, extrasaction="ignore")
        wout.writeheader()

        for row in rin:
            if not row:
                continue
            out = {k: (row.get(k, "") if row.get(k, "") is not None else "") for k in new_fields}
            wout.writerow(out)

    os.replace(tmp_path, path)


def _migrate_legacy_schema_in_place(
    path: str,
    legacy_fields: List[str],
    new_fields: List[str],
    mapping: Dict[str, str],
) -> None:
    """
    Migrate a CSV with a known legacy header (exact match) into the canonical schema.

    Rewrites file in-place (atomic replace) with canonical header and maps known columns.
    """
    tmp_path = path + ".tmp"

    with open(path, "r", encoding="utf-8-sig", newline="") as fin, open(
        tmp_path, "w", encoding="utf-8", newline=""
    ) as fout:
        rin = csv.DictReader(fin)
        read_fields = [h.strip() for h in (rin.fieldnames or [])]
        if read_fields != legacy_fields:
            raise RuntimeError(f"legacy migrate: header changed during read: {read_fields} vs {legacy_fields}")

        wout = csv.DictWriter(fout, fieldnames=new_fields, extrasaction="ignore")
        wout.writeheader()

        for row in rin:
            if not row:
                continue
            out: Dict[str, Any] = {k: "" for k in new_fields}
            for src_key, dst_key in mapping.items():
                if dst_key in out:
                    out[dst_key] = row.get(src_key, "") if row.get(src_key, "") is not None else ""

            # Best-effort: if legacy has only "day" and no "exit_ts", keep exit_ts blank
            # If legacy had something that looks like a timestamp in "entry_ts", keep it.

            wout.writerow(out)

    os.replace(tmp_path, path)


def _rotate_bad_file(path: str, label: str, reason: str) -> str:
    """
    Rotate an existing file to a stamped BAD file and return the new bad path.
    """
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(path)
    bad_path = f"{base}.BAD_{label}_{reason}_{stamp}{ext or '.csv'}"
    try:
        os.replace(path, bad_path)
    except Exception:
        # last resort try rename
        try:
            os.rename(path, bad_path)
        except Exception:
            pass
    return bad_path


def _ensure_canonical_header(
    path: str,
    expected_fieldnames: List[str],
    utils_module,
    *,
    label: str,
    known_legacy_headers: Optional[List[List[str]]] = None,
    legacy_migration_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Ensure the file at `path` has the canonical header matching expected_fieldnames.

    Behavior:
      - If file missing/empty -> create canonical file with header.
      - If header matches exactly -> ALSO validate first data row width; if mismatch -> rotate+recreate.
      - If header is an older ordered prefix of expected_fieldnames -> migrate in place then validate width.
      - If header matches a KNOWN legacy header -> migrate via mapping then validate width.
      - Otherwise -> rotate to *.BAD_* and recreate canonical file.

    This prevents schema drift AND catches the case where header matches but data rows are positional/incorrect.
    """
    try:
        parent_dir = os.path.dirname(path) or "."
        try:
            getattr(utils_module, "ensure_dir")(parent_dir)
        except Exception:
            os.makedirs(parent_dir, exist_ok=True)

        # If file doesn't exist or is empty, create header.
        if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=expected_fieldnames)
                w.writeheader()
            return

        existing = _read_header_fields(path)
        expected = expected_fieldnames

        # Helper: validate row width if there is at least one data row
        def _validate_width_or_rotate() -> None:
            w = _peek_first_data_row_len(path)
            if w is None:
                return
            if int(w) != int(len(expected_fieldnames)):
                bad = _rotate_bad_file(path, label=label, reason=f"ROWWIDTH{w}of{len(expected_fieldnames)}")
                print(f"[{label}] rotated row-width mismatch -> {bad}", file=sys.stderr)
                with open(path, "w", encoding="utf-8", newline="") as f2:
                    w2 = csv.DictWriter(f2, fieldnames=expected_fieldnames)
                    w2.writeheader()

        if existing == expected:
            _validate_width_or_rotate()
            return

        # Subset header: ordered prefix
        if 0 < len(existing) < len(expected) and expected[: len(existing)] == existing:
            _migrate_subset_schema_in_place(path, old_fields=existing, new_fields=expected)
            print(
                f"[{label}] migrated subset header {len(existing)} -> {len(expected)} columns in-place",
                file=sys.stderr,
            )
            _validate_width_or_rotate()
            return

        # Known legacy header: migrate with mapping
        if known_legacy_headers:
            for legacy in known_legacy_headers:
                if existing == legacy:
                    if not legacy_migration_map:
                        break
                    _migrate_legacy_schema_in_place(
                        path,
                        legacy_fields=legacy,
                        new_fields=expected,
                        mapping=legacy_migration_map,
                    )
                    print(
                        f"[{label}] migrated legacy header {len(legacy)} -> {len(expected)} columns in-place",
                        file=sys.stderr,
                    )
                    _validate_width_or_rotate()
                    return

        # Unknown mismatch -> rotate and recreate
        bad_path = _rotate_bad_file(path, label=label, reason="BADHEADER")
        with open(path, "w", encoding="utf-8", newline="") as f2:
            w2 = csv.DictWriter(f2, fieldnames=expected_fieldnames)
            w2.writeheader()
        print(f"[{label}] rotated mismatched header -> {bad_path}", file=sys.stderr)

    except Exception as e:
        print(f"[{label}] header ensure/rotation failed: {e}", file=sys.stderr)


def append_shadow_trade_log(row: Dict[str, Any], *, path: str, utils_module) -> None:
    """
    Appends a per-tick/per-eval shadow log row.
    Ensures file header matches TRADE_FIELDNAMES (migrates subset headers, rotates unknown).
    Also detects the (rare) case: header matches but rows have wrong width (rotates).
    """
    try:
        _ensure_canonical_header(
            path,
            TRADE_FIELDNAMES,
            utils_module,
            label="shadow_log",
            known_legacy_headers=KNOWN_LEGACY_TRADE_HEADERS or None,
            legacy_migration_map=None,
        )

        # Pad missing keys deterministically (never shift columns)
        out = {k: (row.get(k, "") if row.get(k, "") is not None else "") for k in TRADE_FIELDNAMES}

        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRADE_FIELDNAMES, extrasaction="ignore")
            w.writerow(out)

    except Exception as e:
        print(f"[shadow_log] failed to append row: {e}", file=sys.stderr)


def append_shadow_roundtrip_log(row: Dict[str, Any], *, path: str, utils_module) -> None:
    """
    Appends a completed shadow roundtrip (entry->exit).
    Ensures file header matches ROUNDTRIP_FIELDNAMES:
      - migrates subset headers
      - migrates known legacy headers (including your 11-col layout)
      - rotates unknown headers
      - ALSO rotates if header matches but data row widths are wrong (your current failure mode)
    """
    try:
        _ensure_canonical_header(
            path,
            ROUNDTRIP_FIELDNAMES,
            utils_module,
            label="shadow_roundtrip_log",
            known_legacy_headers=KNOWN_LEGACY_ROUNDTRIP_HEADERS or None,
            legacy_migration_map=LEGACY_ROUNDTRIP_MAP_V11_TO_CANON,
        )

        # Pad missing keys deterministically (never shift columns; always canonical schema)
        out = {k: (row.get(k, "") if row.get(k, "") is not None else "") for k in ROUNDTRIP_FIELDNAMES}

        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROUNDTRIP_FIELDNAMES, extrasaction="ignore")
            w.writerow(out)

    except Exception as e:
        print(f"[shadow_roundtrip_log] failed to append row: {e}", file=sys.stderr)
