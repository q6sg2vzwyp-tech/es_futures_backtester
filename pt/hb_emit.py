#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hb_emit.py

Single-responsibility heartbeat writer used by paper_trader / hb_core / pt_hb_emit.

Goal: make heartbeat emission resilient on Windows where atomic rename/replace can
transiently fail with WinError 5 ("Access is denied") if ANY process briefly touches
the destination path at the wrong moment.

This module provides:
  - emit_hb_snapshot(payload: dict, hb_path: str) -> None

Behavior:
  - Writes JSON to a uniquely-named temp file in the SAME directory as hb_path
  - Attempts atomic replace via os.replace()
  - Retries briefly on WinError 5
  - As a last resort, falls back to a non-atomic overwrite write (still best-effort)
  - Never raises to caller (heartbeat should never crash the trader)

Notes:
  - If you run multiple watchdog/paper_trader instances, you can still create contention.
    Fix that first (singleton watchdog).
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict

_DEFAULT_RETRIES = 40          # 40 * 0.05s = ~2s max
_DEFAULT_SLEEP_SEC = 0.05


def emit_hb_snapshot(payload: Dict[str, Any], hb_path: str) -> None:
    """
    Best-effort, crash-safe heartbeat emission.

    Never raises.
    """
    try:
        _emit_hb_snapshot_impl(payload, hb_path)
    except Exception:
        # Never let heartbeat failures crash the trading loop.
        return


def _emit_hb_snapshot_impl(payload: Dict[str, Any], hb_path: str) -> None:
    # Ensure directory exists
    hb_dir = os.path.dirname(hb_path) or "."
    os.makedirs(hb_dir, exist_ok=True)

    # Serialize once (avoid inconsistent partials)
    data = json.dumps(payload, ensure_ascii=False)

    # Unique temp name in same directory (required for atomic replace on Windows)
    rand = f"{random.getrandbits(64):016x}"
    tmp_path = os.path.join(hb_dir, f"._tmp_{os.getpid()}_{rand}.txt")

    # Write temp
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)

    # Attempt atomic swap with retries on WinError 5
    last_exc = None
    for _ in range(_DEFAULT_RETRIES):
        try:
            os.replace(tmp_path, hb_path)
            return
        except OSError as e:
            last_exc = e
            if getattr(e, "winerror", None) == 5:
                time.sleep(_DEFAULT_SLEEP_SEC)
                continue
            break

    # If atomic replace failed, try to cleanup tmp and do a direct overwrite fallback
    try:
        # Direct overwrite (not atomic). Still better than no heartbeat.
        with open(hb_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
    except Exception:
        pass

    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass
