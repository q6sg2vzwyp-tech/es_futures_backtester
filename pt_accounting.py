#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_accounting.py

Centralizes:
- pnl snapshot (via pnl_core.snapshot_es_pnl_and_orders -> pt_pnl_snap.PnlSnap)
- equity + HWM update (via equity_core.update_equity_and_hwm)
- margin snapshot update (via margin_core.MarginSnap + MarginManager.update_snapshot)

Goal: keep loop_core.py thinner and reduce tuple plumbing.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from margin_core import MarginSnap
from pt_pnl_snap import PnlSnap, get_pnl_snap


def compute_net_liq(
    *,
    equity: float,
    last_acct_netliq: Optional[float],
    use_ib_pnl: bool,
) -> float:
    """Return the net liquidation value to use for sizing/margin dashboards."""
    if use_ib_pnl and (last_acct_netliq is not None):
        try:
            return float(last_acct_netliq)
        except Exception:
            return float(equity)
    return float(equity)


def update_margin_snapshot(
    *,
    margin_mgr: Any,
    product: str,
    per_contract_init: float,
    net_qty: int,
    net_liq: float,
) -> float:
    """Update MarginManager snapshot. Returns available_funds used."""
    try:
        current_used = abs(int(net_qty)) * float(per_contract_init)
    except Exception:
        current_used = 0.0
    try:
        available_funds = max(0.0, float(net_liq) - float(current_used))
    except Exception:
        available_funds = 0.0

    try:
        if margin_mgr is not None and hasattr(margin_mgr, "update_snapshot"):
            margin_mgr.update_snapshot(
                MarginSnap(
                    product=str(product),
                    per_contract_init=float(per_contract_init),
                    available_funds=float(available_funds),
                    net_liq=float(net_liq),
                )
            )
    except Exception:
        pass

    return float(available_funds)


def snapshot_and_update_accounting(
    *,
    snapshot_fn,
    update_equity_fn,
    ib,
    con,
    last_px: float,
    use_ib_pnl: bool,
    hwm_stepdown: bool,
    hwm_stepdown_dollars: float,
    acct_netliq_prev: Optional[float],
    equity: float,
    equity_hwm: float,
    net_qty: int,
    per_contract_init: float,
    margin_mgr: Any,
    logger,
) -> Tuple[PnlSnap, float, float, float, Optional[float], float, float]:
    """
    Returns:
      pnl, equity, equity_hwm, hwm_factor, last_acct_netliq, net_liq, available_funds
    """

    pnl = get_pnl_snap(snapshot_fn=snapshot_fn, ib=ib, con=con, last_px=float(last_px), logger=logger)

    # equity/HWM
    equity2, equity_hwm2, hwm_factor, last_acct_netliq2 = update_equity_fn(
        use_ib_pnl=bool(use_ib_pnl),
        hwm_stepdown=bool(hwm_stepdown),
        hwm_stepdown_dollars=float(hwm_stepdown_dollars or 0.0),
        acct_netliq=pnl.acct_netliq,
        equity=float(equity),
        equity_hwm=float(equity_hwm),
        last_acct_netliq=acct_netliq_prev,
    )

    net_liq = compute_net_liq(equity=float(equity2), last_acct_netliq=last_acct_netliq2, use_ib_pnl=bool(use_ib_pnl))
    available_funds = update_margin_snapshot(
        margin_mgr=margin_mgr,
        product="ES",
        per_contract_init=float(per_contract_init),
        net_qty=int(net_qty),
        net_liq=float(net_liq),
    )

    return pnl, float(equity2), float(equity_hwm2), float(hwm_factor), last_acct_netliq2, float(net_liq), float(available_funds)
