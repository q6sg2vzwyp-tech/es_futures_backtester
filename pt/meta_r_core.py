#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
meta_r_core.py

Tiny helper to keep:
  - meta_ema_R : EMA of per-trade R
  - meta_aggr  : aggression factor derived from performance

You can:
  - call update_meta_from_trade(...) on each CLOSED trade with an R value
  - store meta_ema_R / meta_aggr in your in-memory state
  - write them into heartbeat so the dashboard sees them
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MetaState:
    ema_R: float = 0.0   # rolling EMA of R
    aggr: float = 1.0    # aggression multiplier (1.0 = baseline)


def update_meta_from_trade(
    meta: MetaState,
    trade_R: float,
    *,
    alpha: float = 0.2,
    base_aggr: float = 1.0,
    slope: float = 0.5,
    min_aggr: float = 0.5,
    max_aggr: float = 1.5,
) -> MetaState:
    """
    Incorporate a single trade's R into the meta state.

    - alpha: EMA smoothing (0.2 = "20% new, 80% old")
    - aggression mapping:
         aggr_target = base_aggr + slope * ema_R
         clamped to [min_aggr, max_aggr]
    """
    if trade_R is None:
        return meta

    # EMA of R
    meta.ema_R = (1.0 - alpha) * meta.ema_R + alpha * float(trade_R)

    # Map ema_R â†’ aggression multiplier
    aggr_target = base_aggr + slope * meta.ema_R
    if aggr_target < min_aggr:
        aggr_target = min_aggr
    elif aggr_target > max_aggr:
        aggr_target = max_aggr

    meta.aggr = aggr_target
    return meta

