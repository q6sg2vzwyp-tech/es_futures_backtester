#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any


def write_hb(payload: Dict[str, Any], *, hb_path: str, utils_module) -> None:
    """
    Normalize / alias heartbeat fields so the dashboard sees what it expects.
    Then write to hb_path via utils_module.write_heartbeat().
    """
    ts = payload.get("ts") or payload.get("timestamp")
    if ts is not None:
        payload["ts"] = ts
        payload.setdefault("timestamp", ts)

    if "px" in payload and "last_px" not in payload:
        payload["last_px"] = payload["px"]

    if "pnl_unreal_usd" in payload and "unreal_pnl" not in payload:
        payload["unreal_pnl"] = payload["pnl_unreal_usd"]

    if "acct_unreal_pnl" in payload and "acct_unreal" not in payload:
        payload["acct_unreal"] = payload["acct_unreal_pnl"]

    if "acct_realized_pnl" in payload and "acct_realized" not in payload:
        payload["acct_realized"] = payload["acct_realized_pnl"]

    if "ib_err" in payload and "last_ib_error" not in payload:
        payload["last_ib_error"] = payload["ib_err"]

    utils_module.write_heartbeat(hb_path, payload)
