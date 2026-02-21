#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pt_orders.py

Order maintenance helpers extracted from loop_core.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import time


def maybe_guard_naked_position(*, ctx: Dict[str, Any], net: int, es_open_stops: int, es_open_limits: int,
                              last_px: float, last_fill_ts: Optional[float], logger: Any) -> float:
    args = ctx["args"]
    ib = ctx["ib"]
    con = ctx["con"]

    last_nostop_guard_ts = float(ctx.get("last_nostop_guard_ts", 0.0) or 0.0)

    safety_grace_sec = 1.0
    recently_filled = (last_fill_ts is not None) and ((time.time() - float(last_fill_ts)) < safety_grace_sec)
    nostop_cooldown_sec = 30.0

    if (
        bool(getattr(args, "place_orders", False))
        and net != 0
        and int(es_open_stops or 0) == 0
        and int(es_open_limits or 0) == 0
        and (not recently_filled)
        and ((time.time() - float(last_nostop_guard_ts)) >= nostop_cooldown_sec)
    ):
        try:
            logger.warning("[safety_nostop] net position detected with NO protective STOP/TARGET; attaching protection")
        except Exception:
            pass
        last_nostop_guard_ts = time.time()
        ctx["last_nostop_guard_ts"] = last_nostop_guard_ts
        try:
            import order_core
            order_core.guard_naked_position(ib=ib, contract=con, net_qty=net, last_px=last_px, args=args, logger=logger)
        except Exception as e:
            try:
                logger.error("[safety_nostop] failed to attach protection: %s", e)
            except Exception:
                pass

    return float(last_nostop_guard_ts)


def maybe_reconcile_orphans(*, ctx: Dict[str, Any], net: int, last_orphan_sweep_ts: float,
                           orphan_sweep_cooldown: float, logger: Any) -> float:
    if net != 0:
        return float(last_orphan_sweep_ts or 0.0)

    ib = ctx["ib"]
    con = ctx["con"]

    if (time.time() - float(last_orphan_sweep_ts or 0.0)) >= float(orphan_sweep_cooldown):
        try:
            import order_core
            cancelled = order_core.reconcile_orphans(ib, con, net_qty=net, logger=logger)
            if cancelled and int(cancelled) > 0:
                try:
                    logger.info("[reconcile_orphans] cancelled %d orphan orders (net=%s)", int(cancelled), net)
                except Exception:
                    pass
            last_orphan_sweep_ts = time.time()
        except Exception as e:
            try:
                logger.error("[loop] reconcile_orphans error: %s", e)
            except Exception:
                pass

    return float(last_orphan_sweep_ts or 0.0)
