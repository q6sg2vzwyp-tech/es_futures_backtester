#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations


def compute_boost_factor(
    boost_mode: str,
    meta,
    day_risk,
    week_state,
    equity: float,
    equity_hwm: float,
    logger,
) -> float:
    if boost_mode == "off":
        return 1.0

    ema_R = float(getattr(meta, "ema_R", 0.0) or 0.0)
    consec_losses = int(getattr(day_risk, "consec_losses", 0) or 0)
    week_R = float(getattr(week_state, "week_R", 0.0) or 0.0)

    drawdown = 0.0
    if equity_hwm > 0:
        drawdown = max(0.0, (equity_hwm - equity) / equity_hwm)

    factor = 1.0

    if ema_R > 0.3 and week_R > 0:
        factor *= 1.15
    if ema_R > 0.7 and week_R > 1.5:
        factor *= 1.15

    if consec_losses >= 2:
        factor *= 0.7
    if consec_losses >= 3:
        factor *= 0.5

    if drawdown > 0.05:
        factor *= 0.5
    elif drawdown > 0.02:
        factor *= 0.75

    if boost_mode == "war":
        factor *= 1.15

    factor = max(0.25, min(2.0, factor))

    if logger:
        logger.debug(
            f"[boost] mode={boost_mode} ema_R={ema_R:.4f} week_R={week_R:.4f} "
            f"consec_losses={consec_losses} dd={drawdown:.4%} factor={factor:.3f}"
        )

    return factor
