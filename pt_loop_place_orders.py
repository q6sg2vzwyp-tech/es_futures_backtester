#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pt_loop_place_orders.py

Entry placement + orphan reconcile extracted from loop_core.py.

Goal: keep loop_core as a coordinator and isolate the highest-risk operational code path
(order placement and post-entry hygiene) into a single module.

Behavior:
- Exactly mirrors the v4 loop_core block:
    * final_qty computed from clamped_delta
    * bracket prices computed from last_px and stop_dist/tp_dist
    * order_core.place_protected_entry called
    * pos_entry_* snapshots set on success
    * day_risk.last_entry_time set on success (best-effort)
    * orphan sweep (reconcile_orphans) runs when net==0 on cooldown
    * LAST_ORPHAN_SWEEP_TS stored back into ctx
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import time

import order_core


def place_entry_and_orphans(
    *,
    ctx: Dict[str, Any],
    ib: Any,
    con: Any,
    side: str,
    arm: str,
    clamped_delta: int,
    desired_delta: int,
    stop_dist: float,
    tp_dist: float,
    last_px: float,
    now_ct,
    net: int,
    day_risk: Any,
    logger: Any,
    current_arm: Optional[str],
    current_side: Optional[str],
    pos_entry_ct,
    pos_entry_px,
    pos_entry_ts,
) -> Dict[str, Any]:
    LAST_ORPHAN_SWEEP_TS = float(ctx.get("LAST_ORPHAN_SWEEP_TS", 0.0) or 0.0)
    ORPHAN_SWEEP_COOLDOWN = float(ctx.get("ORPHAN_SWEEP_COOLDOWN", 60.0) or 60.0)

    # --- entry placement ---
    final_qty = abs(int(clamped_delta))

    if final_qty <= 0:
        logger.warning("[entry] margin_core blocked entry: desired_delta=%s side=%s", desired_delta, side)
    else:
        if side.upper() == "BUY":
            stop_px = float(last_px) - float(stop_dist)
            target_px = float(last_px) + float(tp_dist)
        else:
            stop_px = float(last_px) + float(stop_dist)
            target_px = float(last_px) - float(tp_dist)

        ok, parent_id, stp_id, tgt_id = order_core.place_protected_entry(
            ib=ib,
            contract=con,
            action=side.upper(),
            qty=final_qty,
            stop_px=stop_px,
            target_px=target_px,
            px_hint=float(last_px),
            logger=logger,
        )

        if ok:
            current_arm = arm
            current_side = "LONG" if side.upper() == "BUY" else "SHORT"
            try:
                day_risk.last_entry_time = time.time()
            except Exception:
                pass
            LAST_ORPHAN_SWEEP_TS = time.time()

            if pos_entry_ct is None:
                pos_entry_ct = now_ct
            if pos_entry_ts is None:
                try:
                    pos_entry_ts = now_ct.isoformat(timespec="seconds")
                except Exception:
                    pos_entry_ts = str(now_ct)
            if pos_entry_px is None:
                try:
                    pos_entry_px = float(last_px)
                except Exception:
                    pos_entry_px = None
        else:
            logger.error("[entry] market entry failed or not filled; CHECK TWS.")

    # --- orphan sweep ---
    if int(net) == 0 and (time.time() - float(LAST_ORPHAN_SWEEP_TS)) >= ORPHAN_SWEEP_COOLDOWN:
        try:
            cancelled = order_core.reconcile_orphans(ib, con, net_qty=int(net), logger=logger)
            if cancelled and int(cancelled) > 0:
                logger.info("[reconcile_orphans] cancelled %d orphan orders (net=%s)", int(cancelled), net)
            LAST_ORPHAN_SWEEP_TS = time.time()
        except Exception as e:
            logger.error("[loop] reconcile_orphans error: %s", e)

    ctx["LAST_ORPHAN_SWEEP_TS"] = float(LAST_ORPHAN_SWEEP_TS)

    return {
        "current_arm": current_arm,
        "current_side": current_side,
        "pos_entry_ct": pos_entry_ct,
        "pos_entry_px": pos_entry_px,
        "pos_entry_ts": pos_entry_ts,
    }
