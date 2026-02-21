from __future__ import annotations

import math
from typing import Optional, Any


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def per_contract_risk_dollars(*, args: Any, px_mult: float, risk_ticks: Optional[int] = None) -> float:
    rt = int(getattr(args, "risk_ticks", 0) if risk_ticks is None else risk_ticks)
    return float(rt) * float(getattr(args, "tick_size", 0.0)) * float(px_mult)


def equity_ladder_size(*, args: Any, eq: float) -> int:
    acct_base = float(getattr(args, "acct_base", 0.0))
    scale_step = max(1.0, float(getattr(args, "scale_step", 1.0)))
    start_contracts = int(getattr(args, "start_contracts", 1))
    max_contracts = int(getattr(args, "max_contracts", 1))
    steps = 0 if eq < acct_base else math.floor((eq - acct_base) / scale_step)
    return int(clamp(start_contracts + steps, 1, max_contracts))


def risk_budget_size(*, args: Any, px_mult: float, eq: float, risk_ticks: Optional[int] = None) -> int:
    max_contracts = int(getattr(args, "max_contracts", 1))
    risk_budget = float(eq) * float(getattr(args, "risk_pct", 0.0))
    pc_risk = per_contract_risk_dollars(args=args, px_mult=px_mult, risk_ticks=risk_ticks)
    if pc_risk <= 0:
        return 1
    return int(clamp(math.floor(risk_budget / pc_risk), 1, max_contracts))


def apply_hwm_stepdown(*, args: Any, qty_suggested: int, equity_hwm: float, equity: float) -> int:
    if not bool(getattr(args, "hwm_stepdown", False)):
        return int(qty_suggested)
    dd = max(0.0, float(equity_hwm) - float(equity))
    hwm_stepdown_dollars = float(getattr(args, "hwm_stepdown_dollars", 0.0))
    if hwm_stepdown_dollars <= 0:
        return int(qty_suggested)
    steps_down = int(math.floor(dd / hwm_stepdown_dollars))
    if steps_down <= 0:
        return int(qty_suggested)
    return max(1, int(qty_suggested) - steps_down)


def margin_cap_qty(*, args: Any, eq_now: float) -> int:
    reserve = max(0.0, float(getattr(args, "margin_reserve_pct", 0.0)))
    eff_eq = max(0.0, float(eq_now) * (1.0 - reserve))
    per = max(1.0, float(getattr(args, "margin_per_contract", 1.0)))
    max_contracts = int(getattr(args, "max_contracts", 1))
    return int(clamp(math.floor(eff_eq / per), 0, max_contracts))


def determine_order_qty(
    *,
    current_net_qty: int,
    risk_ticks_for_trade: Optional[int],
    args: Any,
    px_mult: float,
    equity: float,
    equity_hwm: float,
    ib_netliq: Optional[float],
) -> int:
    """Compute additional contracts to add, respecting ladder, risk budget, HWM stepdown, and margin cap."""
    max_contracts = int(getattr(args, "max_contracts", 1))

    if bool(getattr(args, "use_ib_pnl", False)) and (ib_netliq is not None):
        eq_now = float(ib_netliq)
    else:
        eq_now = float(equity)

    if bool(getattr(args, "static_size", False)):
        qty = int(max(1, round(float(getattr(args, "qty", 1)))))
        return int(max(0, min(qty, max_contracts - abs(int(current_net_qty)))))

    eq_size = equity_ladder_size(args=args, eq=eq_now)
    rb_size = risk_budget_size(args=args, px_mult=px_mult, eq=eq_now, risk_ticks=risk_ticks_for_trade)

    base_qty = int(clamp(min(eq_size, rb_size), 1, max_contracts))
    stepdown_qty = apply_hwm_stepdown(args=args, qty_suggested=base_qty, equity_hwm=equity_hwm, equity=equity)
    mcap_total = margin_cap_qty(args=args, eq_now=eq_now)

    desired_total = min(
        int(clamp(stepdown_qty, 0, max_contracts)),
        int(clamp(mcap_total, 0, max_contracts)),
    )

    if abs(int(current_net_qty)) >= desired_total:
        return 0

    addable = desired_total - abs(int(current_net_qty))
    final_qty = int(max(0, addable))

    if (abs(int(current_net_qty)) + final_qty) > max_contracts:
        final_qty = max(0, max_contracts - abs(int(current_net_qty)))

    return int(final_qty)
