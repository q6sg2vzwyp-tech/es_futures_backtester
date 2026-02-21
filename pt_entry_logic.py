#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_entry_logic.py

Extracts the REAL ENTRY PATH from paper_trader.py:
- shadow veto/mult
- effective risk pct calculation (meta * hwm * boost * shadow)
- dynamic sizing
- margin clamp
- stop/target computation
- order_core.place_protected_entry()

This module is intentionally "thin" and depends on existing project modules:
- position_core.dynamic_contracts
- order_core.place_protected_entry
- margin_core.MarginManager
- shadow_orchestrator.ShadowOrchestrator (entry_multiplier)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Any


@dataclass
class EntryDecision:
    attempted: bool = False
    placed: bool = False
    final_qty: int = 0
    desired_delta: int = 0
    clamped_delta: int = 0
    stop_px: Optional[float] = None
    target_px: Optional[float] = None
    current_arm: Optional[str] = None
    current_side: Optional[str] = None
    caps_added: List[str] = None

    def __post_init__(self):
        if self.caps_added is None:
            self.caps_added = []


def compute_effective_risk_pct(
    base_risk_pct: float,
    meta_factor: float,
    hwm_factor: float,
    boost_factor: float,
    shadow_mult: float,
    side: str,
    short_risk_mult: float = 0.5,
) -> float:
    """
    Computes effective risk_pct including all multipliers.
    Applies short_risk_mult when side is SELL.
    """
    eff = float(base_risk_pct) * float(meta_factor) * float(hwm_factor) * float(boost_factor) * float(shadow_mult)
    if (side or "").upper() == "SELL":
        eff *= float(short_risk_mult)
    return float(eff)


def compute_bracket_prices(
    side: str,
    last_px: float,
    stop_dist: float,
    tp_dist: float,
) -> Tuple[float, float]:
    """
    Converts stop/target distances into absolute stop_px/target_px.
    """
    s = (side or "").upper()
    if s == "BUY":
        return (float(last_px) - float(stop_dist), float(last_px) + float(tp_dist))
    # SELL
    return (float(last_px) + float(stop_dist), float(last_px) - float(tp_dist))


def try_real_entry(
    *,
    can_enter: bool,
    arm: Optional[str],
    side: Optional[str],
    last_px: float,
    stop_dist: float,
    tp_dist: float,
    args: Any,
    equity_for_sizing: float,
    es_multiplier: float,
    boost_factor: float,
    meta_factor: float,
    hwm_factor: float,
    net: int,
    last_regime: str,
    shadow: Any,
    margin_mgr: Any,
    dynamic_contracts_fn,
    place_protected_entry_fn,
    logger,
) -> EntryDecision:
    """
    Attempts an entry if can_enter and arm/side are available.

    Required callables:
      - dynamic_contracts_fn(equity, risk_pct, risk_ticks, tick_size, multiplier, max_contracts) -> int
      - place_protected_entry_fn(ib, contract, action, qty, stop_px, target_px, px_hint, logger) -> (ok, parent_id, stp_id, tgt_id)
        NOTE: we do not require ib/contract here; caller can bind them via functools.partial or lambdas.

    Returns EntryDecision (including any caps_added like shadow veto).
    """
    dec = EntryDecision()

    if not (can_enter and arm and side):
        return dec

    dec.attempted = True

    base_risk_pct = float(getattr(args, "risk_pct", 0.0) or 0.0)
    risk_ticks = int(getattr(args, "risk_ticks", 0) or 0)
    tick_size = float(getattr(args, "tick_size", 0.0) or 0.0)
    max_contracts = int(getattr(args, "max_contracts", 1) or 1)

    # Shadow veto/mult
    shadow_mult, veto = shadow.entry_multiplier(regime=last_regime, arm=arm, side=side, default=1.0)

    if float(shadow_mult) <= 0.0:
        # blocked
        if veto:
            dec.caps_added.append(str(veto))
        else:
            dec.caps_added.append("shadow_block")
        logger.info(
            "[shadow_filter] BLOCKED real entry for arm=%s side=%s shadow_mult=%.2f",
            arm,
            side,
            float(shadow_mult),
        )
        return dec

    if veto:
        dec.caps_added.append(str(veto))

    eff_risk_pct = compute_effective_risk_pct(
        base_risk_pct=base_risk_pct,
        meta_factor=meta_factor,
        hwm_factor=hwm_factor,
        boost_factor=boost_factor,
        shadow_mult=float(shadow_mult),
        side=side,
        short_risk_mult=0.5,
    )

    # Boosted max contracts (preserves your current behavior)
    boosted_max_contracts = max(1, int(round(float(max_contracts) * min(float(boost_factor), 2.0))))

    contracts = int(
        dynamic_contracts_fn(
            equity=float(equity_for_sizing),
            risk_pct=float(eff_risk_pct),
            risk_ticks=int(risk_ticks),
            tick_size=float(tick_size),
            multiplier=float(es_multiplier),
            max_contracts=int(boosted_max_contracts),
        )
    )

    desired_delta = contracts if (side or "").upper() == "BUY" else -contracts
    dec.desired_delta = int(desired_delta)

    clamped_delta = int(
        margin_mgr.clamp_entry_size(
            product="ES",
            desired_qty_delta=int(desired_delta),
            current_net_qty=int(net),
            per_contract_init=float(risk_ticks) * float(tick_size) * float(es_multiplier),
        )
    )
    dec.clamped_delta = clamped_delta

    final_qty = abs(int(clamped_delta))
    dec.final_qty = int(final_qty)

    if final_qty <= 0:
        logger.warning("[entry] margin_core blocked entry: desired_delta=%s side=%s", desired_delta, side)
        return dec

    stop_px, target_px = compute_bracket_prices(
        side=side,
        last_px=float(last_px),
        stop_dist=float(stop_dist),
        tp_dist=float(tp_dist),
    )
    dec.stop_px = float(stop_px)
    dec.target_px = float(target_px)

    ok, parent_id, stp_id, tgt_id = place_protected_entry_fn(
        action=(side or "").upper(),
        qty=int(final_qty),
        stop_px=float(stop_px),
        target_px=float(target_px),
        px_hint=float(last_px),
        logger=logger,
    )

    if ok:
        dec.placed = True
        dec.current_arm = str(arm)
        dec.current_side = "LONG" if (side or "").upper() == "BUY" else "SHORT"
    else:
        logger.error("[entry] market entry failed or not filled; no children were sent - CHECK TWS.")

    return dec
