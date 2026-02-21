#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pt_loop_real_entry.py

Extracted from loop_core.py (v7): real-entry sizing + shadow filter + placement orchestration.

Goal: keep loop_core as coordinator; preserve behavior.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# loop_core provides these at runtime; import guarded to avoid circulars
from pt_loop_features import dynamic_contracts


def maybe_place_real_entry(
    *,
    ctx: Dict[str, Any],
    args: Any,
    shadow: Any,
    margin_mgr: Any,
    day_risk: Any,
    logger: Any,
    ib: Any,
    con: Any,
    can_enter: bool,
    arm: Optional[str],
    side: Optional[str],
    last_regime: Optional[str],
    meta_factor: float,
    hwm_factor: float,
    boost_factor: float,
    equity: float,
    last_acct_netliq: Optional[float],
    per_contract_init: float,
    net: int,
    stop_dist: float,
    tp_dist: float,
    last_px: float,
    now_ct: Any,
    current_arm: Optional[str],
    current_side: Optional[str],
    pos_entry_ct: Optional[str],
    pos_entry_px: Optional[float],
    pos_entry_ts: Optional[str],
) -> Dict[str, Any]:
    """
    Returns dict with:
      - did_attempt (bool)
      - caps (List[str])
      - current_arm/current_side/pos_entry_ct/pos_entry_px/pos_entry_ts (possibly updated)
      - res_place (dict or None)
    """
    from pt_loop_place_orders import place_entry_and_orphans  # local import to avoid cycles

    caps = ctx.get("caps", None)

    if not (can_enter and side and arm):
        return {
            "did_attempt": False,
            "caps": caps or [],
            "current_arm": current_arm,
            "current_side": current_side,
            "pos_entry_ct": pos_entry_ct,
            "pos_entry_px": pos_entry_px,
            "pos_entry_ts": pos_entry_ts,
            "res_place": None,
        }

    base_risk_pct = float(getattr(args, "risk_pct", 0.015) or 0.015)
    reg_for_shadow = ("chop" if (not last_regime or last_regime == "unknown") else last_regime)
    shadow_mult, veto = shadow.entry_multiplier(regime=reg_for_shadow, arm=arm, side=side, default=1.0)

    if shadow_mult <= 0.0:
        logger.info(
            "[shadow_filter] BLOCKED real entry for arm=%s side=%s shadow_mult=%.2f",
            arm,
            side,
            shadow_mult,
        )
        caps = (caps or []) + ([veto] if veto else ["shadow_block"])
        return {
            "did_attempt": True,
            "caps": caps,
            "current_arm": current_arm,
            "current_side": current_side,
            "pos_entry_ct": pos_entry_ct,
            "pos_entry_px": pos_entry_px,
            "pos_entry_ts": pos_entry_ts,
            "res_place": None,
        }

    if veto:
        caps = (caps or []) + [veto]

    effective_risk_pct = base_risk_pct * float(meta_factor) * float(hwm_factor) * float(boost_factor) * float(shadow_mult)

    SHORT_RISK_MULT = 0.5
    if side.upper() == "SELL":
        effective_risk_pct *= SHORT_RISK_MULT
        logger.info(
            "[short_risk] applying SHORT_RISK_MULT=%.2f -> effective_risk_pct=%.5f",
            SHORT_RISK_MULT,
            effective_risk_pct,
        )

    equity_for_sizing = float(equity)
    if bool(getattr(args, "use_ib_pnl", False)) and (last_acct_netliq is not None):
        equity_for_sizing = float(last_acct_netliq)

    boosted_max_contracts = max(
        1, int(round(float(getattr(args, "max_contracts", 6) or 6) * min(float(boost_factor), 2.0)))
    )

    contracts = dynamic_contracts(
        equity=equity_for_sizing,
        risk_pct=effective_risk_pct,
        risk_ticks=args.risk_ticks,
        tick_size=args.tick_size,
        multiplier=50.0,
        max_contracts=boosted_max_contracts,
    )

    desired_delta = contracts if side.upper() == "BUY" else -contracts
    clamped_delta = margin_mgr.clamp_entry_size(
        product="ES",
        desired_qty_delta=desired_delta,
        current_net_qty=net,
        per_contract_init=per_contract_init,
    )

    res_place = place_entry_and_orphans(
        ctx=ctx,
        ib=ib,
        con=con,
        side=side,
        arm=arm,
        clamped_delta=clamped_delta,
        desired_delta=desired_delta,
        stop_dist=stop_dist,
        tp_dist=tp_dist,
        last_px=last_px,
        now_ct=now_ct,
        net=net,
        day_risk=day_risk,
        logger=logger,
        current_arm=current_arm,
        current_side=current_side,
        pos_entry_ct=pos_entry_ct,
        pos_entry_px=pos_entry_px,
        pos_entry_ts=pos_entry_ts,
    )

    return {
        "did_attempt": True,
        "caps": caps or [],
        "current_arm": res_place.get("current_arm", current_arm),
        "current_side": res_place.get("current_side", current_side),
        "pos_entry_ct": res_place.get("pos_entry_ct", pos_entry_ct),
        "pos_entry_px": res_place.get("pos_entry_px", pos_entry_px),
        "pos_entry_ts": res_place.get("pos_entry_ts", pos_entry_ts),
        "res_place": res_place,
    }
