#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_pnl_snap.py

Small wrapper around pnl_core.snapshot_es_pnl_and_orders() that returns a
dataclass instead of a long tuple.

This is intentionally thin: no logic changes, just structure.

PATCH (2026-01-04):
- Add PnlSnap.zero() convenience constructor for smoke tests and safe defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class PnlSnap:
    # ES position/pnl/order snapshot
    es_avg_px: Optional[float]
    es_unreal_pnl_raw: float

    es_open_orders: int
    es_open_stops: int
    es_open_limits: int

    open_order_ids: List[int]
    open_stop_ids: List[int]
    open_limit_ids: List[int]

    stop_px: Optional[float]
    target_px: Optional[float]

    # Account pnl/equity
    acct_unreal_pnl: float
    acct_realized_pnl: float
    acct_netliq: Optional[float]

    @classmethod
    def zero(cls) -> "PnlSnap":
        """
        Safe “all-zero / None” snapshot for initialization and smoke tests.
        """
        return cls(
            es_avg_px=None,
            es_unreal_pnl_raw=0.0,
            es_open_orders=0,
            es_open_stops=0,
            es_open_limits=0,
            open_order_ids=[],
            open_stop_ids=[],
            open_limit_ids=[],
            stop_px=None,
            target_px=None,
            acct_unreal_pnl=0.0,
            acct_realized_pnl=0.0,
            acct_netliq=None,
        )


def _as_int_list(x: Any) -> List[int]:
    if not x:
        return []
    if isinstance(x, list):
        out: List[int] = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    # tolerate tuples/sets/iterables
    try:
        return [int(v) for v in list(x)]
    except Exception:
        return []


def get_pnl_snap(
    *,
    snapshot_fn,
    ib,
    con,
    last_px: float,
    logger,
) -> PnlSnap:
    """
    Calls snapshot_es_pnl_and_orders and returns PnlSnap.

    snapshot_fn should be pnl_core.snapshot_es_pnl_and_orders (or compatible).
    """
    (
        es_avg_px,
        es_unreal_pnl_raw,
        es_open_orders,
        es_open_stops,
        es_open_limits,
        open_order_ids,
        open_stop_ids,
        open_limit_ids,
        stop_px,
        target_px,
        acct_unreal_pnl,
        acct_realized_pnl,
        acct_netliq,
    ) = snapshot_fn(ib=ib, con=con, last_px=last_px, logger=logger)

    # Normalize types defensively (preserve old “best effort” tolerance)
    try:
        es_unreal_pnl_raw_f = float(es_unreal_pnl_raw or 0.0)
    except Exception:
        es_unreal_pnl_raw_f = 0.0

    try:
        acct_unreal_f = float(acct_unreal_pnl or 0.0)
    except Exception:
        acct_unreal_f = 0.0

    try:
        acct_real_f = float(acct_realized_pnl or 0.0)
    except Exception:
        acct_real_f = 0.0

    try:
        es_open_orders_i = int(es_open_orders or 0)
    except Exception:
        es_open_orders_i = 0

    try:
        es_open_stops_i = int(es_open_stops or 0)
    except Exception:
        es_open_stops_i = 0

    try:
        es_open_limits_i = int(es_open_limits or 0)
    except Exception:
        es_open_limits_i = 0

    es_avg_px_f: Optional[float]
    try:
        es_avg_px_f = None if es_avg_px is None else float(es_avg_px)
    except Exception:
        es_avg_px_f = None

    stop_px_f: Optional[float]
    try:
        stop_px_f = None if stop_px is None else float(stop_px)
    except Exception:
        stop_px_f = None

    target_px_f: Optional[float]
    try:
        target_px_f = None if target_px is None else float(target_px)
    except Exception:
        target_px_f = None

    acct_netliq_f: Optional[float]
    try:
        acct_netliq_f = None if acct_netliq is None else float(acct_netliq)
    except Exception:
        acct_netliq_f = None

    return PnlSnap(
        es_avg_px=es_avg_px_f,
        es_unreal_pnl_raw=es_unreal_pnl_raw_f,
        es_open_orders=es_open_orders_i,
        es_open_stops=es_open_stops_i,
        es_open_limits=es_open_limits_i,
        open_order_ids=_as_int_list(open_order_ids),
        open_stop_ids=_as_int_list(open_stop_ids),
        open_limit_ids=_as_int_list(open_limit_ids),
        stop_px=stop_px_f,
        target_px=target_px_f,
        acct_unreal_pnl=acct_unreal_f,
        acct_realized_pnl=acct_real_f,
        acct_netliq=acct_netliq_f,
    )
