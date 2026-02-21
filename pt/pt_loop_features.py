#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pt_loop_features.py

Placeholder module for feature/regime computation extraction.

In v4 we keep this minimal to avoid touching the regime classifier logic.
It currently provides a thin wrapper around ctx-managed last_atr_points/last_adx_val.
"""

from __future__ import annotations

import math
from typing import Any, Dict


def dynamic_contracts(
    *,
    equity: float,
    risk_pct: float,
    risk_ticks: float,
    tick_size: float,
    point_value: float = 50.0,
    min_contracts: int = 1,
    max_contracts: int = 1,
) -> int:
    """Compute position size (contracts) from a fixed-fraction risk budget.

    equity: account equity used for sizing (USD)
    risk_pct: fraction of equity allocated to risk (e.g., 0.002 for 0.2%)
    risk_ticks: stop distance in ticks (or equivalent risk distance)
    tick_size: ES tick size (e.g., 0.25)
    point_value: ES $ per point (ES=50, MES=5)
    min_contracts/max_contracts: clamps for safety
    """
    try:
        eq = float(equity)
        rp = float(risk_pct)
        rt = float(risk_ticks)
        ts = float(tick_size)
        pv = float(point_value)
    except Exception:
        return int(min_contracts)

    # Guardrails
    if eq <= 0.0 or rp <= 0.0:
        return int(min_contracts)
    if rt <= 0.0 or ts <= 0.0 or pv <= 0.0:
        return int(min_contracts)

    risk_budget = eq * rp
    risk_per_contract = rt * ts * pv

    if risk_per_contract <= 0.0:
        return int(min_contracts)

    n = int(math.floor(risk_budget / risk_per_contract))
    if n < int(min_contracts):
        n = int(min_contracts)
    if int(max_contracts) > 0 and n > int(max_contracts):
        n = int(max_contracts)
    return n


def snapshot(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "last_atr_points": float(ctx.get("last_atr_points", 0.0) or 0.0),
        "last_adx_val": float(ctx.get("last_adx_val", 0.0) or 0.0),
        "last_regime": str(ctx.get("last_regime", "unknown") or "unknown"),
    }
