#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
equity_core.py

Equity / NetLiq / HWM stepdown helpers for ES Paper Trader.

- Keeps the logic that:
    * Updates equity / equity_hwm from IB account NetLiq
    * Computes hwm_factor based on a drawdown threshold
"""

from __future__ import annotations

from typing import Optional, Tuple


def update_equity_and_hwm(
    use_ib_pnl: bool,
    hwm_stepdown: bool,
    hwm_stepdown_dollars: float,
    acct_netliq: Optional[float],
    equity: float,
    equity_hwm: float,
    last_acct_netliq: Optional[float],
) -> Tuple[float, float, float, Optional[float]]:
    """
    Update equity / equity_hwm / hwm_factor given the latest IB NetLiq.

    Args:
        use_ib_pnl:            args.use_ib_pnl
        hwm_stepdown:          args.hwm_stepdown
        hwm_stepdown_dollars:  args.hwm_stepdown_dollars
        acct_netliq:           latest NetLiq from IB (float or None)
        equity:                current equity tracker
        equity_hwm:            current equity high-water mark
        last_acct_netliq:      last seen NetLiq (float or None)

    Returns:
        equity, equity_hwm, hwm_factor, last_acct_netliq
    """
    # Update equity / HWM from IB NetLiq
    if use_ib_pnl and acct_netliq is not None:
        try:
            last_acct_netliq = float(acct_netliq)
            equity = last_acct_netliq
            if equity > equity_hwm:
                equity_hwm = equity
        except Exception:
            # If parsing fails, keep previous values
            pass

    # Default: no risk cut
    hwm_factor = 1.0

    # Apply HWM stepdown if enabled
    if hwm_stepdown and hwm_stepdown_dollars > 0 and equity_hwm > 0:
        try:
            dd = equity_hwm - equity
            if dd >= float(hwm_stepdown_dollars):
                # Simple behavior: cut risk 50% on large drawdown
                hwm_factor = 0.5
        except Exception:
            # On any error, fall back to neutral
            hwm_factor = 1.0

    return equity, equity_hwm, hwm_factor, last_acct_netliq

