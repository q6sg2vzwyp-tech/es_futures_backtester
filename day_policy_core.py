#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
day_policy_core.py

Day-level policy orchestration:
- Friday flatten + lockout
- Holiday-eve flatten + lockout
- Auto-flat time flatten + lockout (optional; only if auto_flat_ct configured)
- Preclose sweep flatten + lockout (optional; only if preclose_sweep_ct configured)
- Weekend flatten (Sat/Sun) flatten + lockout (optional; only if weekend_flatten enabled)

Exported API (paper_trader/loop_core depend on these exact names):
- DayPolicyState
- apply_day_policies(...)

PATCH (2026-01-08):
- Latch set BEFORE attempting flatten to prevent re-trigger loops if errors occur.
- Auto-flat no longer logs twice when net==0 (skip fallback attempt if no action possible).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
from typing import Callable, List, Optional


@dataclass
class DayPolicyResult:
    did_flatten: bool = False
    reason: str = ""  # "friday_flat" | "holiday_eve_flat" | "auto_flat" | "preclose_sweep" | "weekend_flat" | ""
    hard_caps: List[str] = field(default_factory=list)


@dataclass
class DayPolicyState:
    """
    Persistent-ish state across loop iterations (kept in memory).
    paper_trader creates one instance and passes it each loop.
    """
    friday_flat_done: bool = False
    friday_flat_date: Optional[dt.date] = None

    holiday_flat_done: bool = False
    holiday_flat_date: Optional[dt.date] = None

    auto_flat_done: bool = False
    auto_flat_date: Optional[dt.date] = None

    preclose_sweep_done: bool = False
    preclose_sweep_date: Optional[dt.date] = None

    weekend_flat_done: bool = False
    weekend_flat_date: Optional[dt.date] = None


def _is_friday(d: dt.date) -> bool:
    return d.weekday() == 4  # Monday=0 ... Friday=4


def _is_weekend(d: dt.date) -> bool:
    return d.weekday() in (5, 6)  # Sat/Sun


def _caps_for_state(
    *,
    now_ct: dt.datetime,
    is_us_market_holiday: Callable[[dt.date], bool],
    state: DayPolicyState,
    auto_flat_ct: Optional[dt.time],
    preclose_sweep_ct: Optional[dt.time],
    weekend_flatten: bool,
) -> List[str]:
    """
    Emit lockout caps only for policies that are actually configured/enabled.
    This prevents weekend/disabled schedules from leaking stale lockouts into caps.
    """
    caps: List[str] = []
    today = now_ct.date()

    if _is_friday(today) and state.friday_flat_done and state.friday_flat_date == today:
        caps.append("friday_flat_lockout")

    tomorrow = today + dt.timedelta(days=1)
    if is_us_market_holiday(tomorrow) and state.holiday_flat_done and state.holiday_flat_date == today:
        caps.append("holiday_eve_lockout")

    if auto_flat_ct is not None and state.auto_flat_done and state.auto_flat_date == today:
        caps.append("auto_flat_lockout")

    if preclose_sweep_ct is not None and state.preclose_sweep_done and state.preclose_sweep_date == today:
        caps.append("preclose_sweep_lockout")

    if weekend_flatten and state.weekend_flat_done and state.weekend_flat_date == today:
        caps.append("weekend_flat_lockout")

    return caps


def _attempt_flatten(
    *,
    reason_tag: str,
    net: int,
    place_orders: bool,
    logger,
    flatten_all: Optional[Callable[[], None]],
    place_market_flat: Optional[Callable[[int], None]],
) -> bool:
    """
    Try to flatten safely:
    - Prefer flatten_all() if provided (cancel+flatten)
    - Else fallback to place_market_flat(net) if provided
    Returns True if an action was attempted (orders sent / flatten called), else False.
    """
    if not place_orders or net == 0:
        logger.info("[%s] triggered but no action (place_orders=%s net=%s)", reason_tag, place_orders, net)
        return False

    if callable(flatten_all):
        try:
            logger.warning("[%s] flatten_all() net=%s", reason_tag, net)
            flatten_all()
            return True
        except Exception as e:
            logger.exception("[%s] flatten_all failed: %s", reason_tag, e)

    if callable(place_market_flat):
        try:
            logger.warning("[%s] fallback place_market_flat(net=%s)", reason_tag, net)
            place_market_flat(net)
            return True
        except Exception as e:
            logger.exception("[%s] place_market_flat failed: %s", reason_tag, e)

    logger.error("[%s] no flatten function available (flatten_all/place_market_flat missing)", reason_tag)
    return False


def apply_day_policies(
    *,
    now_ct: dt.datetime,
    net: int,
    auto_flat_ct: Optional[dt.time],
    preclose_sweep_ct: Optional[dt.time] = None,
    weekend_flatten: bool = False,
    place_orders: bool,
    is_us_market_holiday: Callable[[dt.date], bool],
    flatten_all: Optional[Callable[[], None]] = None,
    place_market_flat: Optional[Callable[[int], None]] = None,
    logger=None,
    state: DayPolicyState,
) -> DayPolicyResult:
    """
    Applies day-level flatten policies.
    """
    res = DayPolicyResult(did_flatten=False, reason="", hard_caps=[])

    if logger is None:
        class _NullLogger:
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass
            def exception(self, *a, **k): pass
        logger = _NullLogger()

    today = now_ct.date()
    now_time = now_ct.time()

    # Reset per-day flags when date changes
    if state.friday_flat_date != today:
        state.friday_flat_done = False
        state.friday_flat_date = None

    if state.holiday_flat_date != today:
        state.holiday_flat_done = False
        state.holiday_flat_date = None

    if state.auto_flat_date != today:
        state.auto_flat_done = False
        state.auto_flat_date = None

    if state.preclose_sweep_date != today:
        state.preclose_sweep_done = False
        state.preclose_sweep_date = None

    if state.weekend_flat_date != today:
        state.weekend_flat_done = False
        state.weekend_flat_date = None

    # ---- 0) WEEKEND FLATTEN ----
    if weekend_flatten and _is_weekend(today) and (not state.weekend_flat_done):
        # LATCH FIRST
        state.weekend_flat_done = True
        state.weekend_flat_date = today

        did = _attempt_flatten(
            reason_tag="weekend_flat",
            net=net,
            place_orders=place_orders,
            logger=logger,
            flatten_all=flatten_all,
            place_market_flat=place_market_flat,
        )
        res.did_flatten = bool(did)
        res.reason = "weekend_flat"
        res.hard_caps = _caps_for_state(
            now_ct=now_ct,
            is_us_market_holiday=is_us_market_holiday,
            state=state,
            auto_flat_ct=auto_flat_ct,
            preclose_sweep_ct=preclose_sweep_ct,
            weekend_flatten=weekend_flatten,
        )
        return res

    # ---- 1) PRECLOSE SWEEP ----
    if preclose_sweep_ct is not None and (not state.preclose_sweep_done) and now_time >= preclose_sweep_ct:
        # LATCH FIRST
        state.preclose_sweep_done = True
        state.preclose_sweep_date = today

        logger.warning(
            "[preclose_sweep] triggered at %s CT (preclose_sweep_ct=%s) net=%s",
            now_time, preclose_sweep_ct, net
        )
        did = _attempt_flatten(
            reason_tag="preclose_sweep",
            net=net,
            place_orders=place_orders,
            logger=logger,
            flatten_all=flatten_all,
            place_market_flat=place_market_flat,
        )
        res.did_flatten = bool(did)
        res.reason = "preclose_sweep"
        res.hard_caps = _caps_for_state(
            now_ct=now_ct,
            is_us_market_holiday=is_us_market_holiday,
            state=state,
            auto_flat_ct=auto_flat_ct,
            preclose_sweep_ct=preclose_sweep_ct,
            weekend_flatten=weekend_flatten,
        )
        return res

    # ---- 2) AUTO-FLAT ----
    if auto_flat_ct is not None and (not state.auto_flat_done) and now_time >= auto_flat_ct:
        # LATCH FIRST
        state.auto_flat_done = True
        state.auto_flat_date = today

        logger.warning("[auto_flat] triggered at %s CT (auto_flat_ct=%s) net=%s", now_time, auto_flat_ct, net)

        # If no action is possible, do exactly ONE "no action" log (no fallback attempt).
        if (not place_orders) or (net == 0):
            _attempt_flatten(
                reason_tag="auto_flat",
                net=net,
                place_orders=place_orders,
                logger=logger,
                flatten_all=None,
                place_market_flat=None,
            )
            did = False
        else:
            # Prefer market-flat first (fast safety), then flatten_all fallback.
            did = _attempt_flatten(
                reason_tag="auto_flat",
                net=net,
                place_orders=place_orders,
                logger=logger,
                flatten_all=None,
                place_market_flat=place_market_flat if callable(place_market_flat) else None,
            )
            if not did:
                did = _attempt_flatten(
                    reason_tag="auto_flat",
                    net=net,
                    place_orders=place_orders,
                    logger=logger,
                    flatten_all=flatten_all,
                    place_market_flat=None,
                )

        res.did_flatten = bool(did)
        res.reason = "auto_flat"
        res.hard_caps = _caps_for_state(
            now_ct=now_ct,
            is_us_market_holiday=is_us_market_holiday,
            state=state,
            auto_flat_ct=auto_flat_ct,
            preclose_sweep_ct=preclose_sweep_ct,
            weekend_flatten=weekend_flatten,
        )
        return res

    # ---- 3) FRIDAY FLAT ----
    if _is_friday(today):
        cutoff = dt.time(hour=15, minute=14)
        if now_time >= cutoff and (not state.friday_flat_done):
            # LATCH FIRST
            state.friday_flat_done = True
            state.friday_flat_date = today

            logger.info("[friday_flat] cutoff reached (cutoff=%s) net=%s", cutoff, net)
            did = _attempt_flatten(
                reason_tag="friday_flat",
                net=net,
                place_orders=place_orders,
                logger=logger,
                flatten_all=flatten_all,
                place_market_flat=place_market_flat,
            )
            res.did_flatten = bool(did)
            res.reason = "friday_flat"
            res.hard_caps = _caps_for_state(
                now_ct=now_ct,
                is_us_market_holiday=is_us_market_holiday,
                state=state,
                auto_flat_ct=auto_flat_ct,
                preclose_sweep_ct=preclose_sweep_ct,
                weekend_flatten=weekend_flatten,
            )
            return res

    # ---- 4) HOLIDAY-EVE FLAT ----
    tomorrow = today + dt.timedelta(days=1)
    if is_us_market_holiday(tomorrow):
        cutoff = dt.time(hour=15, minute=0)
        if now_time >= cutoff and (not state.holiday_flat_done):
            # LATCH FIRST
            state.holiday_flat_done = True
            state.holiday_flat_date = today

            logger.info("[holiday_flat] tomorrow %s is holiday; cutoff=%s net=%s", tomorrow, cutoff, net)
            did = _attempt_flatten(
                reason_tag="holiday_eve_flat",
                net=net,
                place_orders=place_orders,
                logger=logger,
                flatten_all=flatten_all,
                place_market_flat=place_market_flat,
            )
            res.did_flatten = bool(did)
            res.reason = "holiday_eve_flat"
            res.hard_caps = _caps_for_state(
                now_ct=now_ct,
                is_us_market_holiday=is_us_market_holiday,
                state=state,
                auto_flat_ct=auto_flat_ct,
                preclose_sweep_ct=preclose_sweep_ct,
                weekend_flatten=weekend_flatten,
            )
            return res

    # No policy fired; just return current caps (lockouts if any)
    res.hard_caps = _caps_for_state(
        now_ct=now_ct,
        is_us_market_holiday=is_us_market_holiday,
        state=state,
        auto_flat_ct=auto_flat_ct,
        preclose_sweep_ct=preclose_sweep_ct,
        weekend_flatten=weekend_flatten,
    )
    return res
