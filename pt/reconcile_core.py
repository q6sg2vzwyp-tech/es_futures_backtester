# reconcile_core.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReconcileState:
    last_orphan_sweep_ts: float = 0.0


def maybe_reconcile_protectives(
    *,
    net: int,
    stop_px: Optional[float],
    target_px: Optional[float],
    reconcile_fn,
    logger,
) -> None:
    if net == 0:
        return
    if stop_px is None or target_px is None:
        return
    try:
        reconcile_fn(stop_px=stop_px, target_px=target_px)
    except Exception as e:
        logger.error("[order_core] reconcile_protective_orders failed: %s", e)


def maybe_orphan_sweep(
    *,
    net: int,
    cooldown_sec: float,
    state: ReconcileState,
    sweep_fn,
    logger,
) -> None:
    if net != 0:
        return
    now = time.time()
    if (now - state.last_orphan_sweep_ts) < float(cooldown_sec):
        return
    try:
        cancelled = sweep_fn()
        if cancelled and cancelled > 0:
            logger.info("[reconcile_orphans] cancelled %d orphan orders (net=%s)", cancelled, net)
    except Exception as e:
        logger.error("[loop] reconcile_orphans raised unexpected error: %s", e)
    finally:
        state.last_orphan_sweep_ts = now
