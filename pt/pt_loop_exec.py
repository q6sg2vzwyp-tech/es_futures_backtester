#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pt_loop_exec.py

Signal/build bands/regime bookkeeping extracted from loop_core.py.

This module intentionally stays close to the original block: it computes arm/side
and keeps last_signal_* and last_regime synchronized.

It does NOT place orders; it only produces the candidate signal outputs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_signal_phase(
    *,
    ctx: Dict[str, Any],
    bars: Any,
    last_px: float,
    net: int,
    regime: str,
    logger: Any,
) -> Dict[str, Any]:
    args = ctx["args"]
    bandit = ctx["bandit"]

    # These are local-only outputs; return them in a dict.
    arm: Optional[str] = None
    side: Optional[str] = None
    stop_dist: float = 0.0
    tp_dist: float = 0.0
    dyn_tp_R: float = float(getattr(args, "tp_R", 1.0) or 1.0)
    atr_points: float = 0.0
    adx_val: float = 0.0

    last_signal_arm = ctx.get("last_signal_arm", None)
    last_signal_side = ctx.get("last_signal_side", None)
    last_regime = ctx.get("last_regime", "unknown")

    signal_ready = (int(net) == 0 and bool(getattr(bars, "ready", lambda _n: False)(20)))
    if signal_ready:
        real_arms_override = ctx.get("real_arms", None)

        build_signal_and_bands = ctx["build_signal_and_bands"]
        (
            arm,
            side,
            stop_dist,
            tp_dist,
            dyn_tp_R,
            atr_points,
            adx_val,
        ) = build_signal_and_bands(
            bars=bars,
            last_px=last_px,
            bandit=bandit,
            risk_ticks=args.risk_ticks,
            tick_size=args.tick_size,
            base_tp_R=args.tp_R,
            real_arms_override=real_arms_override,
        )

        try:
            last_atr_points = float(atr_points or 0.0)
            last_adx_val = float(adx_val or 0.0)
        except Exception:
            last_atr_points = 0.0
            last_adx_val = 0.0

        try:
            if (arm is not None) and (side is not None):
                side = str(side).upper()
        except Exception:
            pass

        # veto invalid outputs
        try:
            if arm is not None:
                # side=None is NORMAL = "no signal" (do not warn / do not spam logs)
                if side is None:
                    arm = None
                    side = None

                # Anything else must be one of the allowed directions
                elif side not in ("BUY", "SELL", "LONG", "SHORT"):
                    logger.warning("[signal] invalid side=%s for arm=%s; dropping signal", side, arm)
                    arm = None
                    side = None
        except Exception:
            pass

        if arm is not None:
            last_signal_arm = arm
            last_signal_side = side.upper() if side else None
            last_regime = regime

        ctx["last_atr_points"] = float(last_atr_points)
        ctx["last_adx_val"] = float(last_adx_val)

    return {
        "arm": arm,
        "side": side,
        "stop_dist": float(stop_dist or 0.0),
        "tp_dist": float(tp_dist or 0.0),
        "dyn_tp_R": float(dyn_tp_R or getattr(args, "tp_R", 1.0) or 1.0),
        "atr_points": float(ctx.get("last_atr_points", 0.0) or 0.0),
        "adx_val": float(ctx.get("last_adx_val", 0.0) or 0.0),
        "last_signal_arm": last_signal_arm,
        "last_signal_side": last_signal_side,
        "last_regime": last_regime,
    }
