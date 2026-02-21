#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loop_core.py

Runs ONE iteration of the ES Paper Trader main loop.

Key updates (2025-12-31+):
- One canonical shadow_enabled truth:
    args.shadow_enabled (if present) else ctx["shadow_enabled"] else True
- Shadow stepping reliability:
    step at bucket cadence inside shadow window even with no (arm, side)
- Shadow liveness counters:
    synchronized from shadow.heartbeat_fields() (authoritative)
- Shadow roundtrip logging:
    loop passes a dict-row callback through; no signature drift

PATCH (2025-12-31, shadow overtrading rails):
- Pass shadow overtrading controls into shadow.step():
    * max_roundtrips_per_day / max_roundtrips_per_hour
    * post_close_cooldown_sec / post_loss_cooldown_sec
- Ensure shadow.step() can still be called in force-flat path with same rails.
- Surface shadow gating reasons via ctx["shadow_last_status"] and heartbeat fields.

PATCH (2026-01-01, log accuracy hardening):
- Capture stable position entry fields in ctx:
    * pos_entry_px (stable entry price snapshot)
    * pos_entry_ts (stable ISO entry timestamp)
  This prevents entry_px “drift” in close/roundtrip logs and downstream analytics.

PATCH (2026-01-02, HARD DAY-RISK ENTRY GATE):
- Force-consult DayRisk.gate_reason() after compute_gate() and before can_enter/entry placement.
  This guarantees day loss cap / max trades / max consec-losses cannot be bypassed by timing or
  compute_gate() drift. When tripped, we add the gate reason into caps and block entry.

PATCH (2026-01-02, EOD BAYES LATCH):
- EOD Bayes gating is attempted at most once per retry window (default 10 minutes) after EOD trigger.
- Prevents “skip spam” logs when conditions are not met.
"""

from __future__ import annotations

import time
import datetime as dt
import os
import csv
import json
import tempfile
from typing import Any, Dict, Optional, List, Tuple

from ib_insync import MarketOrder

import utils
import order_core

REGIME_ALLOWLIST = {
    "trend": {
        "trend_ema",
        "trend_sma",
        "breakout_atr",
        "pullback_vwap",
        "momentum_rsi",
        "trend_pullback",
        "ma50_intraday",
    },
    "chop": {
        "range_fade",
        "range_fade_strict",
        "mean_revert_ema",
        "pullback_vwap",
    },
    "unknown": set(),
}

# Shadow cadence controls (prevents "dead" shadow while avoiding NOOP spam)
SHADOW_DECISION_BUCKET_SEC_DEFAULT = 60
SHADOW_MIN_HOLD_SEC_DEFAULT = 300

STATE_SAVE_EVERY_SEC_DEFAULT = 5.0

from gate_core import compute_gate
from session_core import reset_daily_flags, reset_caps_for_new_session
from equity_core import update_equity_and_hwm
from pnl_core import snapshot_es_pnl_and_orders
from position_core import compute_position, dynamic_contracts

from pt_utils import compute_sharpe_from_trades
from state_core import save_runtime_state

from day_policy_core import apply_day_policies
from eod_core import maybe_run_eod_bayes_gated
from strategy_core import build_signal_and_bands, regime_from_adx_value, BarBuffer

from trade_bridge import handle_realized_pnl_event
from margin_core import MarginSnap

from trade_bridge import log_event, new_trade_id
from order_core import flatten_until_flat




def _bars15m_cache_path() -> str:
    # Keep it local and portable.
    return os.path.join(".", "run", "bars_15m_seed.csv")


def _bars15m_save_cache(bars_15m: BarBuffer, path: str, logger=None) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "close"])
            # BarBuffer stores arrays; fall back safely.
            ts_list = getattr(bars_15m, "ts", [])
            close_list = getattr(bars_15m, "close", [])
            n = min(len(ts_list), len(close_list))
            for i in range(n):
                ts = ts_list[i]
                px = close_list[i]
                w.writerow([getattr(ts, "isoformat", lambda: str(ts))(), float(px)])
        if logger:
            logger.info("[bars15m] cached %d bars -> %s", n, path)
    except Exception as e:
        if logger:
            logger.warning("[bars15m] cache save failed: %s", e)


def _bars15m_load_cache(path: str, logger=None) -> Optional[BarBuffer]:
    if not os.path.exists(path):
        return None
    try:
        buf = BarBuffer(maxlen=1024)
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                ts = row.get("ts")
                px = row.get("close")
                if not ts or px is None:
                    continue
                # Parse ISO timestamp; tolerate date-only.
                try:
                    t = dt.datetime.fromisoformat(ts)
                except Exception:
                    # last resort: keep raw string
                    t = ts
                buf.add(t, float(px))
        if logger:
            logger.info("[bars15m] loaded %d bars from cache %s", len(getattr(buf, "close", [])), path)
        return buf
    except Exception as e:
        if logger:
            logger.warning("[bars15m] cache load failed: %s", e)
        return None


def _bars15m_preload_from_ibkr(ib, con, logger=None, duration_str: str = "3 D", useRTH: bool = False) -> Optional[BarBuffer]:
    """Preload 15-minute bars from IBKR so EMA50/ATR/ADX are ready at session start.

    Returns a populated BarBuffer on success, else None.
    """
    try:
        # ib_insync: endDateTime="" means now.
        hist = ib.reqHistoricalData(
            con,
            endDateTime="",
            durationStr=duration_str,
            barSizeSetting="15 mins",
            whatToShow="TRADES",
            useRTH=useRTH,
            formatDate=1,
            keepUpToDate=False,
        )
        if not hist:
            if logger:
                logger.warning("[bars15m] IBKR preload returned no bars")
            return None
        buf = BarBuffer(maxlen=1024)
        for b in hist:
            ts = getattr(b, "date", None)
            close = getattr(b, "close", None)
            if ts is None or close is None:
                continue
            # ib_insync can return datetime or string
            if isinstance(ts, str):
                try:
                    ts = dt.datetime.fromisoformat(ts)
                except Exception:
                    pass
            buf.add(ts, float(close))
        if logger:
            logger.info("[bars15m] preloaded %d x 15m bars from IBKR (useRTH=%s, duration=%s)", len(getattr(buf, "close", [])), useRTH, duration_str)
        return buf
    except Exception as e:
        if logger:
            logger.warning("[bars15m] IBKR preload failed: %s", e)
        return None


def _atomic_write_json(path: str, payload: dict) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

    tmp_path = None
    fd = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="atomic_", suffix=".tmp", dir=d, text=True)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _safe_set(obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    if not path or (not os.path.exists(path)):
        return []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _arm_meanR_from_rows(
    rows: List[Dict[str, Any]],
    arm_key_candidates: List[str],
    r_key_candidates: List[str],
) -> Dict[str, Tuple[int, float]]:
    per: Dict[str, List[float]] = {}
    for row in rows:
        arm = ""
        for k in arm_key_candidates:
            v = row.get(k)
            if v:
                arm = str(v).strip()
                break
        if not arm:
            continue

        r_val = None
        for rk in r_key_candidates:
            v = row.get(rk)
            if v is None:
                continue
            s = str(v).strip()
            if s in ("", "NA", "na", "None", "null", "-"):
                continue
            try:
                r_val = float(s)
            except Exception:
                r_val = None
            if r_val is not None:
                break

        if r_val is None:
            continue

        per.setdefault(arm, []).append(float(r_val))

    out: Dict[str, Tuple[int, float]] = {}
    for arm, rs in per.items():
        if rs:
            out[arm] = (len(rs), sum(rs) / float(len(rs)))
    return out


def _maybe_promote_shadow_to_real(ctx: Dict[str, Any], now_ct: dt.datetime, logger) -> Optional[str]:
    args = ctx["args"]
    eod_state = ctx.get("eod_state", None)
    if eod_state is None:
        return None

    if not bool(getattr(args, "promote_shadow_to_real", False)):
        return None

    today = now_ct.date()
    if getattr(eod_state, "promoted_date", None) == today:
        return None

    shadow_csv = ctx.get("SHADOW_ROUNDTRIP_LOG", "")
    real_trades_csv = ctx.get("TRADE_LOG_CSV", "")
    real_arms_json = ctx.get("REAL_ARMS_JSON", "")

    min_shadow = int(getattr(args, "promote_min_shadow_trades", 30) or 30)
    min_real = int(getattr(args, "promote_min_real_trades", 10) or 10)
    thresh = float(getattr(args, "promote_meanR_threshold", 0.10) or 0.10)

    shadow_rows = _read_csv_rows(shadow_csv)
    real_rows = _read_csv_rows(real_trades_csv)

    shadow_stats = _arm_meanR_from_rows(
        shadow_rows,
        arm_key_candidates=["arm", "strategy", "signal"],
        r_key_candidates=["R", "r"],
    )
    real_stats = _arm_meanR_from_rows(
        real_rows,
        arm_key_candidates=["arm", "strategy", "signal"],
        r_key_candidates=["R", "r"],
    )

    real_arms = ctx.get("real_arms", None)
    if not isinstance(real_arms, list) or not real_arms:
        return None
    real_arms = [str(a).strip() for a in real_arms if str(a).strip()]
    if not real_arms:
        return None

    best_shadow_arm = None
    best_shadow_mean = None
    best_shadow_n = 0
    for arm, (n, meanR) in shadow_stats.items():
        if n < min_shadow:
            continue
        if best_shadow_mean is None or meanR > best_shadow_mean:
            best_shadow_arm = arm
            best_shadow_mean = float(meanR)
            best_shadow_n = int(n)

    if best_shadow_arm is None or best_shadow_mean is None:
        logger.info("[promote] no eligible shadow arm yet (need >=%d shadow trades per arm)", min_shadow)
        return None

    worst_real_arm = None
    worst_real_score = None
    worst_real_n = 0
    for arm in real_arms:
        n, meanR = real_stats.get(arm, (0, 0.0))
        score = -999.0 if n < min_real else float(meanR)
        if worst_real_score is None or score < worst_real_score:
            worst_real_arm = arm
            worst_real_score = score
            worst_real_n = int(n)

    if worst_real_arm is None or worst_real_score is None:
        return None

    if best_shadow_arm in real_arms:
        logger.info("[promote] best shadow arm already in real allowlist: %s", best_shadow_arm)
        try:
            eod_state.promoted_date = today
        except Exception:
            pass
        return None

    if best_shadow_mean < (float(worst_real_score) + thresh):
        logger.info(
            "[promote] no promote: best_shadow=%s (n=%d meanR=%.3f) vs worst_real=%s (n=%d score=%.3f) + thresh=%.3f",
            best_shadow_arm,
            best_shadow_n,
            best_shadow_mean,
            worst_real_arm,
            worst_real_n,
            float(worst_real_score),
            thresh,
        )
        try:
            eod_state.promoted_date = today
        except Exception:
            pass
        return None

    new_real_arms = [a for a in real_arms if a != worst_real_arm]
    new_real_arms.append(best_shadow_arm)

    try:
        if real_arms_json:
            payload = {
                "updated": now_ct.isoformat(timespec="seconds"),
                "real_arms": new_real_arms,
                "promoted_in": {
                    "added": best_shadow_arm,
                    "removed": worst_real_arm,
                    "best_shadow_n": best_shadow_n,
                    "best_shadow_meanR": best_shadow_mean,
                    "worst_real_n": worst_real_n,
                    "worst_real_score": float(worst_real_score),
                    "threshold": thresh,
                },
            }
            _atomic_write_json(real_arms_json, payload)

        ctx["real_arms"] = new_real_arms
        logger.warning(
            "[promote] UPDATED real_arms: removed=%s added=%s -> %s",
            worst_real_arm,
            best_shadow_arm,
            new_real_arms,
        )
    except Exception as e:
        logger.error("[promote] failed to write real arms json: %s", e)
        return None

    try:
        eod_state.promoted_date = today
    except Exception:
        pass

    return f"promoted:{best_shadow_arm}:replaced:{worst_real_arm}"


def run_loop_iteration(ctx: Dict[str, Any]) -> Dict[str, Any]:
    args = ctx["args"]
    logger = ctx["logger"]
    ib = ctx["ib"]
    con = ctx["con"]
    ticker = ctx["ticker"]
    bars = ctx["bars"]
    # 15m aggregated bars (used only by ma50_intraday; does not affect other arms)
    bars_15m = ctx.get("bars_15m")
    if bars_15m is None:
        bars_15m = BarBuffer(maxlen=1024)
        ctx["bars_15m"] = bars_15m
        ctx["bars_15m_bucket_ts"] = None

    # Current time (CT) must be defined before any seeding logic uses it
    now_ct = utils.ct_now()
    now_time = now_ct.time()


    # --- Option B automation: preload 15m bars so EMA50 is ready at the open ---
    # Seed once per day (and after reconnect if needed). This prevents long warmup waits.
    #
    # Important: For ES, using useRTH=True early in the morning can return too few 15m bars
    # (only regular trading hours). That keeps the seed under the >=50-bar threshold and
    # causes repeated "seed not ready" logging each loop.
    #
    # Defaulting to useRTH=False pulls overnight/session bars as well, which reliably yields
    # >50 bars and eliminates the preload retry spam.
    today_iso = now_ct.date().isoformat() if hasattr(now_ct, "date") else str(now_ct)
    if ctx.get("bars_15m_seeded_for") != today_iso:

        # Backoff: avoid calling IBKR every loop while waiting for the daily seed.
        now_ts = now_ct.timestamp()
        last_try = float(ctx.get("bars15m_seed_last_try_ts") or 0.0)
        min_retry = float(getattr(args, "bars15m_seed_retry_sec", 60) or 60)

        seeded = None
        if now_ts - last_try >= min_retry:
            ctx["bars15m_seed_last_try_ts"] = now_ts
            try:
                useRTH = bool(getattr(args, "bars15m_seed_useRTH", False))
                seeded = _bars15m_preload_from_ibkr(ib, con, logger=logger, useRTH=useRTH)
            except Exception:
                seeded = None

        if seeded is None:
            seeded = _bars15m_load_cache(_bars15m_cache_path(), logger=logger)

        if seeded:
            ctx["bars_15m_seeded_for"] = today_iso
        else:
            # Throttle the "seed not ready" log so we don't spam once per loop.
            last_log = float(ctx.get("bars15m_seed_last_log_ts") or 0.0)
            if now_ts - last_log >= 60:
                ctx["bars15m_seed_last_log_ts"] = now_ts
                logger.info("[bars15m] seed not ready yet; continuing without preload")
    day_risk = ctx["day_risk"]
    week_state = ctx["week_state"]
    bandit = ctx["bandit"]
    meta = ctx["meta"]
    shadow = ctx["shadow"]
    margin_mgr = ctx["margin_mgr"]
    day_policy_state = ctx["day_policy_state"]
    eod_state = ctx["eod_state"]

    HB_PATH = ctx["HB_PATH"]
    TRADE_LOG_CSV = ctx["TRADE_LOG_CSV"]
    RUNTIME_STATE_JSON = ctx["RUNTIME_STATE_JSON"]
    SHADOW_START_CT = ctx["SHADOW_START_CT"]
    SHADOW_END_CT = ctx["SHADOW_END_CT"]
    DAILY_RESTART_CT = ctx["DAILY_RESTART_CT"]
    DAILY_RESTART_JSON = ctx["DAILY_RESTART_JSON"]
    ORPHAN_SWEEP_COOLDOWN = ctx["ORPHAN_SWEEP_COOLDOWN"]
    IB_ERROR_DECAY_SEC = ctx["IB_ERROR_DECAY_SEC"]
    LEARN_MODEL_PATH = ctx["LEARN_MODEL_PATH"]
    last_regime: str = ctx.get("last_regime", "unknown")

    is_us_market_holiday = ctx["is_us_market_holiday"]
    maybe_daily_restart = ctx["maybe_daily_restart"]
    build_and_write_heartbeat = ctx["build_and_write_heartbeat"]
    roll_week_if_needed = ctx["roll_week_if_needed"]
    append_shadow_roundtrip_log = ctx["append_shadow_roundtrip_log"]
    compute_boost_factor = ctx["compute_boost_factor"]
    build_bandit_hb_fields = ctx["build_bandit_hb_fields"]

    BAYES_SOURCE = ctx["BAYES_SOURCE"]
    AUTO_FLAT_CT = ctx["AUTO_FLAT_CT"]
    PRE_CLOSE_SWEEP_CT = ctx.get("PRE_CLOSE_SWEEP_CT", None)
    WEEKEND_FLATTEN = bool(ctx.get("WEEKEND_FLATTEN", False))

    last_fill_ts = ctx.get("last_fill_ts", None)
    last_nostop_guard_ts = float(ctx.get("last_nostop_guard_ts", 0.0) or 0.0)
    last_ib_err = ctx.get("last_ib_err", None)
    LAST_ORPHAN_SWEEP_TS = float(ctx.get("LAST_ORPHAN_SWEEP_TS", 0.0))
    last_sharpe_update_ts = float(ctx.get("last_sharpe_update_ts", 0.0))
    sharpe_R_value = float(ctx.get("sharpe_R_value", 0.0))
    last_state_save_ts = float(ctx.get("last_state_save_ts", 0.0))

    day_date = ctx["day_date"]
    caps_reset_date = ctx.get("caps_reset_date", None)
    bayes_ran_today = bool(ctx.get("bayes_ran_today", False))
    safety_halt_for_today = bool(ctx.get("safety_halt_for_today", False))
    safety_last_ts = ctx.get("safety_last_ts", None)

    equity = float(ctx.get("equity", 100000.0))
    equity_hwm = float(ctx.get("equity_hwm", equity))
    hwm_factor = float(ctx.get("hwm_factor", 1.0))
    last_acct_netliq = ctx.get("last_acct_netliq", None)

    trades_today = int(ctx.get("trades_today", 0))
    total_trades = int(ctx.get("total_trades", 0))
    running_pnl_today = float(ctx.get("running_pnl_today", 0.0))
    wins_today = int(ctx.get("wins_today", 0))
    losses_today = int(ctx.get("losses_today", 0))
    last_trade_close_ts = ctx.get("last_trade_close_ts", None)
    last_acct_realized = ctx.get("last_acct_realized", None)

    pos_entry_ct = ctx.get("pos_entry_ct", None)

    # NEW: stable entry fields for accurate logs
    pos_entry_px = ctx.get("pos_entry_px", None)
    pos_entry_ts = ctx.get("pos_entry_ts", None)

    current_arm = ctx.get("current_arm", None)
    current_side = ctx.get("current_side", None)
    last_signal_arm = ctx.get("last_signal_arm", None)
    last_signal_side = ctx.get("last_signal_side", None)

    trade_start = ctx["trade_start"]
    trade_end = ctx["trade_end"]

    now_ts = time.time()

    # ---------------------------
    # SHADOW ENABLE SWITCH (one truth)
    # ---------------------------
    try:
        shadow_enabled = bool(getattr(args, "shadow_enabled"))
    except Exception:
        shadow_enabled = bool(ctx.get("shadow_enabled", True))
    ctx["shadow_enabled"] = bool(shadow_enabled)

    # Persisted indicators for shadow when no new calc
    last_atr_points = float(ctx.get("last_atr_points", 0.0) or 0.0)
    last_adx_val = float(ctx.get("last_adx_val", 0.0) or 0.0)

    es_multiplier = 50.0
    per_contract_init = float(args.risk_ticks) * float(args.tick_size) * es_multiplier

    # ---------------------------
    # Shadow overtrading rails (read once, pass through)
    # ---------------------------
    shadow_decision_bucket_sec = int(
        getattr(args, "shadow_decision_bucket_sec", SHADOW_DECISION_BUCKET_SEC_DEFAULT) or SHADOW_DECISION_BUCKET_SEC_DEFAULT
    )
    shadow_min_hold_sec = int(
        getattr(args, "shadow_min_hold_sec", SHADOW_MIN_HOLD_SEC_DEFAULT) or SHADOW_MIN_HOLD_SEC_DEFAULT
    )
    shadow_max_hold_sec = getattr(args, "shadow_max_hold_sec", None)
    try:
        shadow_max_hold_sec = int(shadow_max_hold_sec) if shadow_max_hold_sec is not None else None
    except Exception:
        shadow_max_hold_sec = None

    shadow_max_rts_day = int(getattr(args, "shadow_max_roundtrips_per_day", 0) or 0)
    shadow_max_rts_hour = int(getattr(args, "shadow_max_roundtrips_per_hour", 0) or 0)
    shadow_post_close_cd = int(getattr(args, "shadow_post_close_cooldown_sec", 0) or 0)
    shadow_post_loss_cd = int(getattr(args, "shadow_post_loss_cooldown_sec", 0) or 0)

    _safe_set(shadow, "max_roundtrips_per_day", shadow_max_rts_day)
    _safe_set(shadow, "max_roundtrips_per_hour", shadow_max_rts_hour)
    _safe_set(shadow, "post_close_cooldown_sec", shadow_post_close_cd)
    _safe_set(shadow, "post_loss_cooldown_sec", shadow_post_loss_cd)

    # market data
    last_price = ticker.last or ticker.marketPrice()
    if last_price is None:
        logger.warning("[md] no last price yet (ticker.last/marketPrice None); waiting...")

        try:
            net0 = int(round(compute_position(ib, con)))
        except Exception:
            net0 = 0
        hb_pos_state0 = "flat" if net0 == 0 else (f"long{net0}" if net0 > 0 else f"short{abs(net0)}")

        extra0: Dict[str, Any] = {}
        try:
            extra0.update(build_bandit_hb_fields(bandit))
        except Exception:
            pass
        try:
            extra0.update(margin_mgr.heartbeat_fields())
        except Exception:
            pass
        try:
            extra0["shadow_enabled"] = bool(shadow_enabled)
            extra0["in_shadow_window"] = False
            extra0["shadow_max_roundtrips_per_day"] = int(shadow_max_rts_day)
            extra0["shadow_max_roundtrips_per_hour"] = int(shadow_max_rts_hour)
            extra0["shadow_post_close_cooldown_sec"] = int(shadow_post_close_cd)
            extra0["shadow_post_loss_cooldown_sec"] = int(shadow_post_loss_cd)
        except Exception:
            pass

        sh0: Dict[str, Any] = {}
        try:
            sh0.update(shadow.heartbeat_fields() or {})
        except Exception:
            pass

        build_and_write_heartbeat(
            ib=ib,
            con=con,
            hb_path=HB_PATH,
            now_ct=now_ct,
            hb_state="idle",
            idle_reason="md_no_last_price",
            hb_pos_state=hb_pos_state0,
            net=net0,
            day_risk=day_risk,
            week_state=week_state,
            last_px=0.0,
            bars_len=int(bars.count()) if hasattr(bars, "count") else 0,
            caps=["md_no_last_price"],
            last_ib_err=last_ib_err,
            bayes_source=BAYES_SOURCE,
            restart_ct_str=DAILY_RESTART_CT.isoformat(timespec="minutes"),
            meta=meta,
            meta_factor=float(ctx.get("meta_factor", 1.0) or 1.0),
            boost_mode=getattr(args, "boost_mode", "off"),
            boost_factor=float(ctx.get("boost_factor", 1.0) or 1.0),
            sharpe_R=float(ctx.get("sharpe_R_value", 0.0) or 0.0),
            current_arm=ctx.get("current_arm", None),
            current_side=ctx.get("current_side", None),
            last_signal_arm=ctx.get("last_signal_arm", None),
            last_signal_side=ctx.get("last_signal_side", None),
            regime=str(ctx.get("last_regime", "unknown") or "unknown"),
            equity=float(ctx.get("equity", 100000.0) or 100000.0),
            equity_hwm=float(ctx.get("equity_hwm", ctx.get("equity", 100000.0)) or 100000.0),
            hwm_factor=float(ctx.get("hwm_factor", 1.0) or 1.0),
            shadow_fields=sh0,
            trades_today=int(ctx.get("trades_today", 0) or 0),
            total_trades=int(ctx.get("total_trades", 0) or 0),
            running_pnl_today=float(ctx.get("running_pnl_today", 0.0) or 0.0),
            extra_fields=extra0,
            logger=logger,
        )

        time.sleep(1.0)
        return ctx

    last_px = float(last_price)

    in_real_window = utils.in_time_window(now_time, trade_start, trade_end)
    in_shadow_window = utils.in_time_window(now_time, SHADOW_START_CT, SHADOW_END_CT)

    # --- DAILY FLAGS + DAY ROLLOVER ---

    # Track prior-tick day/px so we can approximate "settlement" close at day rollover.
    # This enables a lightweight daily MA50 bias filter (hybrid mode) without new data dependencies.
    _prev_tick_day = ctx.get("_day_date_at_last_tick")
    _prev_tick_px = ctx.get("_px_at_last_tick")
    old_day_date = day_date
    (
        day_date,
        bayes_ran_today,
        _ff_done_legacy,
        _ff_date_legacy,
        _hf_done_legacy,
        _hf_date_legacy,
        safety_halt_for_today,
        safety_last_ts,
        trades_today,
        running_pnl_today,
        wins_today,
        losses_today,
        last_trade_close_ts,
    ) = reset_daily_flags(
        now_ct=now_ct,
        day_date=day_date,
        bayes_ran_today=bayes_ran_today,
        friday_flat_done=False,
        friday_flat_date=None,
        holiday_flat_done=False,
        holiday_flat_date=None,
        safety_halt_for_today=safety_halt_for_today,
        safety_last_ts=safety_last_ts,
        trades_today=trades_today,
        running_pnl_today=running_pnl_today,
        wins_today=wins_today,
        losses_today=losses_today,
        last_trade_close_ts=last_trade_close_ts,
    )

    if day_date != old_day_date:

        # --- HYBRID DAILY BIAS (approx settlement close from prior tick) ---
        try:
            daily_closes = ctx.get("daily_closes")
            if not isinstance(daily_closes, list):
                daily_closes = []
            # Only append if we have a prior-tick close from the prior day.
            if _prev_tick_day == old_day_date and _prev_tick_px is not None:
                daily_closes.append(float(_prev_tick_px))
                # keep a reasonable cap
                if len(daily_closes) > 260:
                    daily_closes = daily_closes[-260:]
            ctx["daily_closes"] = daily_closes

            # Compute daily SMA50 bias from prior-day closes
            daily_sma50 = None
            # Use up to 50 closes; require at least 10 to avoid meaningless bias early.
            if len(daily_closes) >= 10:
                n = 50 if len(daily_closes) >= 50 else len(daily_closes)
                daily_sma50 = sum(daily_closes[-n:]) / float(n)
            ctx["daily_sma50"] = daily_sma50

            # Bias is evaluated using the most recent "settlement" close we appended
            daily_bias = "neutral"
            if daily_sma50 is not None and len(daily_closes) >= 1:
                last_settle = float(daily_closes[-1])
                # Small deadband to avoid churn around the daily MA
                deadband = max(0.25 * float(ctx.get("last_atr_points") or 0.0), 1.0)
                if last_settle > daily_sma50 + deadband:
                    daily_bias = "bull"
                elif last_settle < daily_sma50 - deadband:
                    daily_bias = "bear"
            ctx["daily_bias"] = daily_bias
            logger.info("[hybrid] daily_bias=%s daily_sma50=%s closes=%d", daily_bias, daily_sma50, len(daily_closes))
        except Exception as e:
            logger.warning("[hybrid] daily bias update failed: %s", e)
        try:
            shadow.reset_day()
            ctx["shadow_eod_ran_today"] = False
            logger.info("[shadow] reset_day on day rollover: %s -> %s", old_day_date, day_date)
        except Exception as e:
            logger.error("[shadow] reset_day failed on rollover: %s", e)
        # IMPORTANT: IB RealizedPnL baseline can reset on new day/session.
        # If we carry last_acct_realized across days, we can create a fake “close trade”
        # at the session open. Force a reseed on next tick.
        last_acct_realized = None
        ctx["last_acct_realized"] = None


    (
        caps_reset_date,
        trades_today,
        running_pnl_today,
        wins_today,
        losses_today,
        safety_halt_for_today,
        safety_last_ts,
        last_trade_close_ts,
    ) = reset_caps_for_new_session(
        now_ct=now_ct,
        caps_reset_date=caps_reset_date,
        day_risk=day_risk,
        safety_halt_for_today=safety_halt_for_today,
        safety_last_ts=safety_last_ts,
        trades_today=trades_today,
        running_pnl_today=running_pnl_today,
        wins_today=wins_today,
        losses_today=losses_today,
        last_trade_close_ts=last_trade_close_ts,
        logger=logger,
    )

    maybe_daily_restart(
        now_ct=now_ct,
        logger=logger,
        restart_json_path=DAILY_RESTART_JSON,
        cutoff_time=DAILY_RESTART_CT,
    )

    net = int(round(compute_position(ib, con)))
    hb_pos_state = "flat"
    if net > 0:
        hb_pos_state = f"long{net}"
    elif net < 0:
        hb_pos_state = f"short{abs(net)}"

    def _place_market_flat(current_net: int) -> None:
        action = "SELL" if current_net > 0 else "BUY"
        qty = int(round(abs(current_net)))
        logger.warning("[policy_flat] sending %s %s @ MKT (net=%s)", action, qty, current_net)
        order = MarketOrder(action, qty)
        tid = new_trade_id('PFLAT')
        try:
            order.orderRef = tid
        except Exception:
            pass
        try:
            log_event('policy_flat_submit', tid, side=action, qty=int(qty or 0), expected_px=None, reason='loop_core_policy_flat', net=current_net)
        except Exception:
            pass
        tr = ib.placeOrder(con, order)
        try:
            # attach fill logger if available (ib_insync Trade)
            fe = getattr(tr, 'filledEvent', None)
            if fe is not None:
                def _h(*a, **k):
                    try:
                        os_ = getattr(tr, 'orderStatus', None)
                        avg_px = None
                        filled_qty = None
                        try:
                            avg_px = float(getattr(os_, 'avgFillPrice', None)) if os_ is not None else None
                        except Exception:
                            avg_px = None
                        try:
                            filled_qty = int(getattr(os_, 'filled', None)) if os_ is not None else None
                        except Exception:
                            filled_qty = None
                        oid = getattr(getattr(tr, 'order', None), 'orderId', None)
                        log_event('policy_flat_fill', tid, order_id=str(oid) if oid is not None else '', side=action, qty=int(filled_qty if filled_qty is not None else qty), fill_px=avg_px, reason='loop_core_policy_flat')
                    except Exception:
                        pass
                try:
                    fe += _h
                except Exception:
                    pass
        except Exception:
            pass

    def _flatten_all() -> None:
        ok = flatten_until_flat(ib, con, logger=logger, max_attempts=10, sleep_sec=1.0)
        if not ok:
            raise RuntimeError("flatten_until_flat returned False")

    # ----------------------------
    # DAY POLICIES
    # ----------------------------
    # ----------------------------
    # Day-policy trigger hygiene
    # ----------------------------
    # If the process starts after the configured preclose/auto-flat times and we are already FLAT,
    # the naive "now >= cutover" checks can cause noisy trigger logs every iteration.
    #
    # Safety rule:
    #   - If net != 0 and we're past a cutover, we still want policies to fire and flatten.
    #   - If net == 0 and we're well past the cutover, suppress those triggers for the rest of the
    #     day by pushing the effective cutovers to end-of-day.
    effective_auto_flat_ct = AUTO_FLAT_CT
    effective_preclose_sweep_ct = PRE_CLOSE_SWEEP_CT
    try:
        if net == 0:
            now_min = int(now_ct.hour) * 60 + int(now_ct.minute)

            def _to_min(t: dt.time) -> int:
                return int(t.hour) * 60 + int(t.minute)

            def _is_late(t: dt.time, grace_min: int = 10) -> bool:
                return now_min > (_to_min(t) + int(grace_min))

            if AUTO_FLAT_CT is not None and _is_late(AUTO_FLAT_CT, grace_min=10):
                effective_auto_flat_ct = dt.time(23, 59, 59)
            if PRE_CLOSE_SWEEP_CT is not None and _is_late(PRE_CLOSE_SWEEP_CT, grace_min=10):
                effective_preclose_sweep_ct = dt.time(23, 59, 59)
    except Exception:
        # Never let policy hygiene break the loop.
        pass

    policy_res = apply_day_policies(
        now_ct=now_ct,
        net=net,
        auto_flat_ct=effective_auto_flat_ct,
        preclose_sweep_ct=effective_preclose_sweep_ct,
        weekend_flatten=WEEKEND_FLATTEN,
        place_orders=bool(getattr(args, "place_orders", False)),
        is_us_market_holiday=is_us_market_holiday,
        flatten_all=_flatten_all,
        place_market_flat=_place_market_flat,
        logger=logger,
        state=day_policy_state,
    )

    hard_caps: List[str] = []
    if safety_halt_for_today:
        hard_caps.append("safety_halt_for_today")
    if policy_res is not None:
        hard_caps.extend(list(getattr(policy_res, "hard_caps", []) or []))

    def _hb_emit(*, hb_state: str, idle_reason: str, caps: List[str]) -> None:
        extra: Dict[str, Any] = {}
        try:
            extra.update(build_bandit_hb_fields(bandit))
        except Exception:
            pass
        try:
            extra.update(margin_mgr.heartbeat_fields())
        except Exception:
            pass

        try:
            extra["real_arms"] = ctx.get("real_arms", [])
        except Exception:
            pass

        try:
            extra["shadow_roundtrips_csv"] = str(ctx.get("SHADOW_ROUNDTRIP_LOG", "") or "")
            extra["shadow_enabled"] = bool(shadow_enabled)
            extra["atr_points"] = float(last_atr_points)
            extra["adx_val"] = float(last_adx_val)
            extra["in_shadow_window"] = bool(in_shadow_window)
            extra["in_real_window"] = bool(in_real_window)
            extra["shadow_max_roundtrips_per_day"] = int(shadow_max_rts_day)
            extra["shadow_max_roundtrips_per_hour"] = int(shadow_max_rts_hour)
            extra["shadow_post_close_cooldown_sec"] = int(shadow_post_close_cd)
            extra["shadow_post_loss_cooldown_sec"] = int(shadow_post_loss_cd)
        except Exception:
            pass

        sh: Dict[str, Any] = {}
        try:
            sh.update(shadow.heartbeat_fields() or {})
        except Exception:
            pass
        try:
            sh_last = ctx.get("shadow_last_status", {}) or {}
            if isinstance(sh_last, dict):
                sh.update(sh_last)
        except Exception:
            pass

        try:
            ctx["shadow_last_eval_ts"] = float(sh.get("shadow_last_eval_ts", 0.0) or 0.0)
            ctx["shadow_eval_count_today"] = int(sh.get("shadow_eval_count_today", 0) or 0)
        except Exception:
            pass

        build_and_write_heartbeat(
            ib=ib,
            con=con,
            hb_path=HB_PATH,
            now_ct=now_ct,
            hb_state=hb_state,
            idle_reason=idle_reason,
            hb_pos_state=hb_pos_state,
            net=net,
            day_risk=day_risk,
            week_state=week_state,
            last_px=last_px,
            bars_len=bars.count(),
            caps=caps,
            last_ib_err=last_ib_err,
            bayes_source=BAYES_SOURCE,
            restart_ct_str=DAILY_RESTART_CT.isoformat(timespec="minutes"),
            meta=meta,
            meta_factor=float(ctx.get("meta_factor", 1.0) or 1.0),
            boost_mode=getattr(args, "boost_mode", "off"),
            boost_factor=float(ctx.get("boost_factor", 1.0) or 1.0),
            sharpe_R=float(ctx.get("sharpe_R_value", 0.0) or 0.0),
            current_arm=current_arm,
            current_side=current_side,
            last_signal_arm=last_signal_arm,
            last_signal_side=last_signal_side,
            regime=last_regime,
            equity=equity,
            equity_hwm=equity_hwm,
            hwm_factor=hwm_factor,
            shadow_fields=sh,
            trades_today=trades_today,
            total_trades=total_trades,
            running_pnl_today=running_pnl_today,
            extra_fields=extra,
            logger=logger,
        )

    if bool(getattr(policy_res, "did_flatten", False)):
        net2 = int(round(compute_position(ib, con)))
        hb_pos_state2 = "flat" if net2 == 0 else (f"long{net2}" if net2 > 0 else f"short{abs(net2)}")
        net = net2
        hb_pos_state = hb_pos_state2

        if net == 0:
            pos_entry_ct = None
            pos_entry_px = None
            pos_entry_ts = None

        reason = str(getattr(policy_res, "reason", "") or "policy_flat")
        _hb_emit(hb_state="idle", idle_reason=reason, caps=[reason] + hard_caps)

        ctx.update(
            {
                "day_date": day_date,
                "caps_reset_date": caps_reset_date,
                "bayes_ran_today": bayes_ran_today,
                "safety_halt_for_today": safety_halt_for_today,
                "safety_last_ts": safety_last_ts,
                "trades_today": trades_today,
                "running_pnl_today": running_pnl_today,
                "wins_today": wins_today,
                "losses_today": losses_today,
                "last_trade_close_ts": last_trade_close_ts,
                "week_state": week_state,
                "bandit": bandit,
                "meta": meta,
                "last_fill_ts": last_fill_ts,
                "last_ib_err": last_ib_err,
                "LAST_ORPHAN_SWEEP_TS": LAST_ORPHAN_SWEEP_TS,
                "last_sharpe_update_ts": last_sharpe_update_ts,
                "sharpe_R_value": sharpe_R_value,
                "last_state_save_ts": last_state_save_ts,
                "equity": equity,
                "equity_hwm": equity_hwm,
                "hwm_factor": hwm_factor,
                "last_acct_netliq": last_acct_netliq,
                "pos_entry_ct": pos_entry_ct,
                "pos_entry_px": pos_entry_px,
                "pos_entry_ts": pos_entry_ts,
                "current_arm": current_arm,
                "current_side": current_side,
                "last_signal_arm": last_signal_arm,
                "last_signal_side": last_signal_side,
                "last_regime": last_regime,
                "total_trades": total_trades,
                "last_acct_realized": last_acct_realized,
                "last_atr_points": float(last_atr_points),
                "last_adx_val": float(last_adx_val),
                "shadow_enabled": bool(shadow_enabled),
            }
        )
        time.sleep(1.0)
        return ctx

    # ---------------------------------------------------------------------
    # EOD bayes gating (RUN-ONCE latch) - MUST be above outside-window return
    # ---------------------------------------------------------------------
    eod_time = dt.time(15, 10)
    bayes_ignore_reasons = [
        "safety_flatten",
        "naked_position_guard",
        "weekend",
        "manual_flat",
        "startup_protect",
        "daily_restart",
        "friday_flat",
    ]

    today_key = now_ct.date().isoformat()
    attempt_day = ctx.get("eod_bayes_attempt_day")
    attempt_ts = float(ctx.get("eod_bayes_attempt_ts", 0.0) or 0.0)

    # retry cadence (10 minutes). Set huge (10**9) if you want strictly once/day.
    retry_sec = 600.0

    try:
        should_attempt = bool((now_ct.time() >= eod_time) or (now_ct.time() >= trade_end))
    except Exception:
        should_attempt = False

    eod_res = None
    if should_attempt:
        if (attempt_day != today_key) or ((time.time() - attempt_ts) >= retry_sec):
            ctx["eod_bayes_attempt_day"] = today_key
            ctx["eod_bayes_attempt_ts"] = float(time.time())

            eod_res = maybe_run_eod_bayes_gated(
                now_ct=now_ct,
                trade_end=trade_end,
                eod_time=eod_time,
                state=eod_state,
                trades_csv=TRADE_LOG_CSV,
                bayes_train_csv=ctx["BAYES_TRAIN_CSV"],
                best_params_path=ctx["LEARN_BAYES_BEST"],
                param_space={
                    "risk_ticks": (8, 20),
                    "tp_R": (0.75, 1.5),
                    "pos_age_cap_sec": (600, 1800),
                    "min_seconds_between_entries": (10, 30),
                    "strategy_cooldown_sec": (10, 30),
                    "parent_to_mkt_sec": (3, 12),
                },
                ignore_reasons=bayes_ignore_reasons,
                min_trades=int(getattr(args, "eod_min_trades", 3) or 3),
                build_bayes_training_set=ctx["build_bayes_training_set"],
                run_eod_bayes_opt_filtered=ctx["run_eod_bayes_opt_filtered"],
                logger=logger,
            )

    if getattr(eod_res, "ran", False):
        bayes_ran_today = True

        if bool(ctx.get("shadow_eod_ran_today", False)):
            logger.info("[shadow_core] EOD shadow update already ran today; skipping")
        else:
            ctx["shadow_eod_ran_today"] = True
            try:
                logger.info("[shadow_core] EOD update from shadow_roundtrips.csv")
                shadow.maybe_update_model_eod(logger=logger)
            except Exception as e:
                logger.exception("[shadow_core] EOD update failed: %s", e)

            promo_msg = _maybe_promote_shadow_to_real(ctx, now_ct, logger)
            if promo_msg:
                ctx["last_promo_msg"] = promo_msg
    # ---------------------------------------------------------------------

    week_state = roll_week_if_needed(week_state)

    if (not in_real_window) and (not in_shadow_window):
        try:
            sh_status = shadow.step(
                now_ct=now_ct,
                last_px=last_px,
                bars=bars,
                bars_15m=bars_15m,
                in_shadow_window=False,
                arm=None,
                side=None,
                per_contract_init=per_contract_init,
                last_regime=last_regime,
                week_R=float(getattr(week_state, "week_R", 0.0) or 0.0),
                meta_ema_R=float(getattr(meta, "ema_R", 0.0) or 0.0),
                append_shadow_roundtrip_log=append_shadow_roundtrip_log,
                atr_points=float(last_atr_points),
                tick_size=float(getattr(args, "tick_size", 0.25) or 0.25),
                decision_bucket_sec=int(shadow_decision_bucket_sec),
                min_hold_sec=int(shadow_min_hold_sec),
                atr_floor_ticks=2.0,
                max_hold_sec=shadow_max_hold_sec,
                shadow_enabled=shadow_enabled,
                max_roundtrips_per_day=shadow_max_rts_day,
                max_roundtrips_per_hour=shadow_max_rts_hour,
                post_close_cooldown_sec=shadow_post_close_cd,
                post_loss_cooldown_sec=shadow_post_loss_cd,
            )
            ctx["shadow_last_status"] = dict(sh_status) if isinstance(sh_status, dict) else {}
        except Exception as e:
            logger.error("[shadow] step (force-flat) failed: %s", e)

        if net == 0:
            pos_entry_ct = None
            pos_entry_px = None
            pos_entry_ts = None

        caps_idle = ["outside_all_trading_windows"] + hard_caps
        if ctx.get("last_promo_msg"):
            caps_idle = caps_idle + [str(ctx.get("last_promo_msg"))]
        _hb_emit(hb_state="idle", idle_reason="outside_all_trading_windows", caps=caps_idle)

        time.sleep(1.0)
        ctx.update(
            {
                "day_date": day_date,
                "caps_reset_date": caps_reset_date,
                "bayes_ran_today": bayes_ran_today,
                "safety_halt_for_today": safety_halt_for_today,
                "safety_last_ts": safety_last_ts,
                "trades_today": trades_today,
                "running_pnl_today": running_pnl_today,
                "wins_today": wins_today,
                "losses_today": losses_today,
                "last_trade_close_ts": last_trade_close_ts,
                "week_state": week_state,
                "bandit": bandit,
                "meta": meta,
                "last_fill_ts": last_fill_ts,
                "last_ib_err": last_ib_err,
                "LAST_ORPHAN_SWEEP_TS": LAST_ORPHAN_SWEEP_TS,
                "last_sharpe_update_ts": last_sharpe_update_ts,
                "sharpe_R_value": sharpe_R_value,
                "last_state_save_ts": last_state_save_ts,
                "equity": equity,
                "equity_hwm": equity_hwm,
                "hwm_factor": hwm_factor,
                "last_acct_netliq": last_acct_netliq,
                "pos_entry_ct": pos_entry_ct,
                "pos_entry_px": pos_entry_px,
                "pos_entry_ts": pos_entry_ts,
                "current_arm": current_arm,
                "current_side": current_side,
                "last_signal_arm": last_signal_arm,
                "last_signal_side": last_signal_side,
                "last_regime": last_regime,
                "total_trades": total_trades,
                "last_acct_realized": last_acct_realized,
                "last_atr_points": float(last_atr_points),
                "last_adx_val": float(last_adx_val),
                "shadow_enabled": bool(shadow_enabled),
                "eod_bayes_attempt_day": ctx.get("eod_bayes_attempt_day"),
                "eod_bayes_attempt_ts": float(ctx.get("eod_bayes_attempt_ts", 0.0) or 0.0),
            }
        )
        return ctx

    # ---------------------------
    # Main loop logic
    # ---------------------------

    # Add bar
    try:
        last_bar_ts = bars.ts[-1] if getattr(bars, "ts", None) else None
    except Exception:
        last_bar_ts = None

    if (last_bar_ts is None) or (now_ct > last_bar_ts):
        bars.add(now_ct, last_px)
    else:
        try:
            bars.close[-1] = float(last_px)
        except Exception:
            pass

    # Update 15-minute aggregated bars (close-only).
    # We maintain one bar per 15-minute bucket and continuously update the current bucket close.
    try:
        bucket_ts = now_ct.replace(minute=(now_ct.minute // 15) * 15, second=0, microsecond=0)
    except Exception:
        bucket_ts = now_ct
    last_bucket_ts = ctx.get("bars_15m_bucket_ts")
    try:
        last_15m_ts = bars_15m.ts[-1] if getattr(bars_15m, "ts", None) else None
    except Exception:
        last_15m_ts = None

    if last_bucket_ts is None:
        ctx["bars_15m_bucket_ts"] = bucket_ts
        # Initialize the first bucket bar
        if last_15m_ts is None or bucket_ts > last_15m_ts:
            bars_15m.add(bucket_ts, last_px)
        else:
            try:
                bars_15m.close[-1] = float(last_px)
            except Exception:
                pass
    else:
        if bucket_ts > last_bucket_ts:
            ctx["bars_15m_bucket_ts"] = bucket_ts
            bars_15m.add(bucket_ts, last_px)
        else:
            try:
                bars_15m.close[-1] = float(last_px)
            except Exception:
                pass

    post_flat_cd = int(getattr(args, "post_flat_cooldown_sec", 0) or 0)
    gate_reason, caps, idle_reason = compute_gate(
        now_ct=now_ct,
        day_risk=day_risk,
        min_seconds_between_entries=int(getattr(args, "min_seconds_between_entries", 0) or 0),
        hard_caps=hard_caps,
        post_flat_cooldown_sec=post_flat_cd,
        last_trade_close_ts=last_trade_close_ts,
    )

    # ------------------------------------------------------------------
    # HARD DAY-RISK ENTRY GATE (cannot be bypassed)
    # ------------------------------------------------------------------
    try:
        dr_gr = day_risk.gate_reason()
    except Exception:
        dr_gr = None

    if dr_gr:
        gate_reason = gate_reason or str(dr_gr)
        if caps is None:
            caps = []
        if str(dr_gr) not in caps:
            caps = list(caps) + [str(dr_gr)]
        try:
            logger.warning(
                "[DAY_RISK_BLOCK] gate=%s day_R=%.3f cap=%.3f trades=%s consec_losses=%s",
                str(dr_gr),
                float(getattr(day_risk, "day_R", 0.0) or 0.0),
                float(getattr(day_risk, "loss_cap_R", 0.0) or 0.0),
                int(getattr(day_risk, "trades", trades_today) or trades_today),
                int(getattr(day_risk, "consec_losses", 0) or 0),
            )
        except Exception:
            pass
    # ------------------------------------------------------------------

    (
        es_avg_px,
        es_unreal_pnl_raw,
        es_open_orders,
        es_open_stops,
        es_open_limits,
        open_order_ids,
        open_stop_ids,
        open_limit_ids,
        stop_px,
        target_px,
        acct_unreal_pnl,
        acct_realized_pnl,
        acct_netliq,
    ) = snapshot_es_pnl_and_orders(ib=ib, con=con, last_px=last_px, logger=logger)

    # stable entry state capture
    if net != 0:
        if pos_entry_ct is None:
            pos_entry_ct = now_ct
        if pos_entry_ts is None:
            try:
                pos_entry_ts = now_ct.isoformat(timespec="seconds")
            except Exception:
                pos_entry_ts = str(now_ct)
        if pos_entry_px is None:
            try:
                pos_entry_px = float(es_avg_px) if es_avg_px is not None else float(last_px)
            except Exception:
                pos_entry_px = float(last_px)
    else:
        pos_entry_ct = None
        pos_entry_px = None
        pos_entry_ts = None

    equity, equity_hwm, hwm_factor, last_acct_netliq = update_equity_and_hwm(
        use_ib_pnl=bool(args.use_ib_pnl),
        hwm_stepdown=bool(args.hwm_stepdown),
        hwm_stepdown_dollars=float(getattr(args, "hwm_stepdown_dollars", 0.0) or 0.0),
        acct_netliq=acct_netliq,
        equity=equity,
        equity_hwm=equity_hwm,
        last_acct_netliq=last_acct_netliq,
    )

    boost_factor = compute_boost_factor(
        boost_mode=getattr(args, "boost_mode", "off"),
        meta=meta,
        day_risk=day_risk,
        week_state=week_state,
        equity=equity,
        equity_hwm=equity_hwm,
        logger=logger,
    )

    meta_factor = float(ctx.get("meta_factor", 1.0) or 1.0)

    can_enter = (
        gate_reason is None
        and in_real_window
        and net == 0
        and es_open_orders == 0
        and es_open_stops == 0
        and es_open_limits == 0
    )

    # safety nostop guard
    safety_grace_sec = 1.0
    recently_filled = (last_fill_ts is not None) and ((time.time() - float(last_fill_ts)) < safety_grace_sec)
    nostop_cooldown_sec = 30.0
    if (
        bool(args.place_orders)
        and net != 0
        and es_open_stops == 0
        and es_open_limits == 0
        and (not recently_filled)
        and ((time.time() - float(last_nostop_guard_ts)) >= nostop_cooldown_sec)
    ):
        logger.warning("[safety_nostop] net position detected with NO protective STOP/TARGET; attaching protection")
        last_nostop_guard_ts = time.time()
        ctx["last_nostop_guard_ts"] = last_nostop_guard_ts
        try:
            order_core.guard_naked_position(ib=ib, contract=con, net_qty=net, last_px=last_px, args=args, logger=logger)
        except Exception as e:
            logger.error("[safety_nostop] failed to attach protection: %s", e)

    current_used = abs(net) * per_contract_init
    net_liq = float(last_acct_netliq) if (bool(getattr(args, "use_ib_pnl", False)) and last_acct_netliq is not None) else float(equity)
    available_funds = max(0.0, net_liq - current_used)
    margin_mgr.update_snapshot(
        MarginSnap(product="ES", per_contract_init=per_contract_init, available_funds=available_funds, net_liq=net_liq)
    )

    # position age cap
    pos_age_cap = int(getattr(args, "pos_age_cap_sec", 0) or 0)
    if net != 0:
        if pos_entry_ct is None:
            pos_entry_ct = now_ct
        elif pos_age_cap > 0:
            age_sec = (now_ct - pos_entry_ct).total_seconds()
            if age_sec >= pos_age_cap:
                logger.info("[pos_age] flattening position after %.0fs (net=%s)", age_sec, net)
                order_core.flatten_all(ib, con, logger=logger)
                net = int(round(compute_position(ib, con)))
                if net == 0:
                    pos_entry_ct = None
                    pos_entry_px = None
                    pos_entry_ts = None
                else:
                    pos_entry_ct = now_ct
    else:
        pos_entry_ct = None
        pos_entry_px = None
        pos_entry_ts = None

    # -----------------------------------------------------------
    # IBKR nightly reset guard (RealizedPnL can reset between ~23:00–00:00 CT)
    # Prevents a bogus "close trade" row when IB drops realized PnL back to 0.
    # -----------------------------------------------------------
    try:
        if acct_realized_pnl is not None and last_acct_realized is not None:
            ar = float(acct_realized_pnl)
            lr = float(last_acct_realized)
            # Detect a meaningful downward jump (typically a reset to ~0)
            if ar < (lr - 1.0):
                near_midnight = (now_time >= dt.time(23, 0)) or (now_time <= dt.time(0, 30))
                near_zero = abs(ar) <= 5.0
                if near_midnight or near_zero:
                    logger.warning(
                        "[realized_reset_guard] IB realizedPnL appears to have reset (%.2f -> %.2f); reseeding baseline; no trade logged",
                        lr,
                        ar,
                    )
                    last_acct_realized = None
                    ctx["last_acct_realized"] = None
    except Exception:
        pass
    (
        last_acct_realized,
        last_trade_close_ts,
        trades_today,
        total_trades,
        running_pnl_today,
        wins_today,
        losses_today,
        day_risk,
        week_state,
        bandit,
        meta,
        current_arm,
        current_side,
    ) = handle_realized_pnl_event(
        ib=ib,
        con=con,
        now_ct=now_ct,
        acct_realized_pnl=acct_realized_pnl,
        last_acct_realized=last_acct_realized,
        args=args,
        day_risk=day_risk,
        week_state=week_state,
        bandit=bandit,
        meta=meta,
        current_arm=current_arm,
        current_side=current_side,
        gate_reason=gate_reason,
        trades_today=trades_today,
        total_trades=total_trades,
        running_pnl_today=running_pnl_today,
        wins_today=wins_today,
        losses_today=losses_today,
        last_trade_close_ts=last_trade_close_ts,
        es_avg_px=es_avg_px,
        last_px=last_px,
        trade_log_csv=TRADE_LOG_CSV,
        learn_model_path=LEARN_MODEL_PATH,
        day_date=day_date,
        caps=caps,
        net=net,
        logger=logger,
    )

    _safe_set(day_risk, "trades", int(trades_today))

    if getattr(args, "learn_mode", "advisory") != "off":
        try:
            meta_factor = float(meta.aggressiveness_factor())
            meta_factor = max(0.5, min(1.5, meta_factor))
            try:
                meta.last_factor = meta_factor  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception as e:
            logger.error("[meta] aggressiveness_factor failed: %s", e)
            meta_factor = 1.0

    arm: Optional[str] = None
    side: Optional[str] = None
    stop_dist: float = 0.0
    tp_dist: float = 0.0
    dyn_tp_R: float = float(getattr(args, "tp_R", 1.0) or 1.0)
    atr_points: float = 0.0
    adx_val: float = 0.0

    signal_ready = (net == 0 and bars.ready(20))
    if signal_ready:
        real_arms_override = ctx.get("real_arms", None)

        arm, side, stop_dist, tp_dist, dyn_tp_R, atr_points, adx_val = build_signal_and_bands(
            bars=bars,
            bars_15m=bars_15m,
            last_px=last_px,
            bandit=bandit,
            risk_ticks=args.risk_ticks,
            tick_size=args.tick_size,
            base_tp_R=args.tp_R,
            real_arms_override=real_arms_override,
        )

    # Hybrid daily bias gate for MA50 intraday arm:
    # - bull day: only allow BUY
    # - bear day: only allow SELL
    # - neutral: stand down (prevents chop around the daily MA)
    if arm == "ma50_intraday" and side in ("BUY", "SELL"):
        daily_bias = ctx.get("daily_bias", "neutral")
        if daily_bias == "bull" and side == "SELL":
            logger.info("[hybrid] block ma50_intraday SELL on bull day")
            side = None
        elif daily_bias == "bear" and side == "BUY":
            logger.info("[hybrid] block ma50_intraday BUY on bear day")
            side = None
        elif daily_bias not in ("bull", "bear"):
            logger.info("[hybrid] block ma50_intraday trade on neutral day")
            side = None

        try:
            last_atr_points = float(atr_points or 0.0)
            last_adx_val = float(adx_val or 0.0)
        except Exception:
            pass

        regime = regime_from_adx_value(adx_val)
        last_regime = regime

        allowed_arms = REGIME_ALLOWLIST.get(regime, set())
        if allowed_arms and arm and arm not in allowed_arms:
            caps = (caps or []) + [f"regime_block:{regime}:{arm}"]
            arm = None
            side = None

        if arm is not None:
            last_signal_arm = arm
            last_signal_side = side.upper() if side else None
            last_regime = regime

    # ---------------------------
    # SHADOW STEP
    # ---------------------------
    if in_shadow_window:
        decision_bucket_sec = int(shadow_decision_bucket_sec)
        min_hold_sec = int(shadow_min_hold_sec)

        should_step = False
        step_reason = "tick"
        if (arm is not None) and (side is not None):
            should_step = True
            step_reason = "signal"
        else:
            try:
                last_eval = float((shadow.heartbeat_fields() or {}).get("shadow_last_eval_ts", 0.0) or 0.0)
            except Exception:
                last_eval = 0.0
            if (time.time() - last_eval) >= float(decision_bucket_sec):
                should_step = True
                step_reason = "bucket"

        if should_step:
            try:
                sh_status = shadow.step(
                    now_ct=now_ct,
                    last_px=last_px,
                    bars=bars,
                    bars_15m=bars_15m,
                    in_shadow_window=True,
                    arm=arm,
                    side=side,
                    per_contract_init=per_contract_init,
                    last_regime=("chop" if (not last_regime or last_regime == "unknown") else last_regime),
                    week_R=float(getattr(week_state, "week_R", 0.0) or 0.0),
                    meta_ema_R=float(getattr(meta, "ema_R", 0.0) or 0.0),
                    append_shadow_roundtrip_log=append_shadow_roundtrip_log,
                    atr_points=float(last_atr_points),
                    tick_size=float(getattr(args, "tick_size", 0.25) or 0.25),
                    decision_bucket_sec=decision_bucket_sec,
                    min_hold_sec=min_hold_sec,
                    atr_floor_ticks=2.0,
                    max_hold_sec=shadow_max_hold_sec,
                    shadow_enabled=shadow_enabled,
                    max_roundtrips_per_day=shadow_max_rts_day,
                    max_roundtrips_per_hour=shadow_max_rts_hour,
                    post_close_cooldown_sec=shadow_post_close_cd,
                    post_loss_cooldown_sec=shadow_post_loss_cd,
                )

                ctx["shadow_last_status"] = dict(sh_status) if isinstance(sh_status, dict) else {}
                if isinstance(ctx["shadow_last_status"], dict):
                    ctx["shadow_last_status"]["shadow_step_reason"] = step_reason
                    ctx["shadow_last_status"]["shadow_enabled"] = bool(shadow_enabled)
            except Exception as e:
                logger.error("[shadow] step failed: %s", e)

    # ---------------------------
    # REAL ENTRY
    # ---------------------------
    if can_enter and side and arm:
        base_risk_pct = float(getattr(args, "risk_pct", 0.015) or 0.015)
        reg_for_shadow = ("chop" if (not last_regime or last_regime == "unknown") else last_regime)
        shadow_mult, veto = shadow.entry_multiplier(regime=reg_for_shadow, arm=arm, side=side, default=1.0)

        if shadow_mult <= 0.0:
            logger.info("[shadow_filter] BLOCKED real entry for arm=%s side=%s shadow_mult=%.2f", arm, side, shadow_mult)
            caps = (caps or []) + ([veto] if veto else ["shadow_block"])
        else:
            if veto:
                caps = (caps or []) + [veto]

            effective_risk_pct = base_risk_pct * meta_factor * hwm_factor * boost_factor * float(shadow_mult)

            SHORT_RISK_MULT = 0.5
            if side.upper() == "SELL":
                effective_risk_pct *= SHORT_RISK_MULT
                logger.info("[short_risk] applying SHORT_RISK_MULT=%.2f -> effective_risk_pct=%.5f", SHORT_RISK_MULT, effective_risk_pct)

            equity_for_sizing = equity
            if bool(getattr(args, "use_ib_pnl", False)) and (last_acct_netliq is not None):
                equity_for_sizing = float(last_acct_netliq)

            boosted_max_contracts = max(
                1, int(round(float(getattr(args, "max_contracts", 6) or 6) * min(boost_factor, 2.0)))
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
            final_qty = abs(int(clamped_delta))

            if final_qty <= 0:
                logger.warning("[entry] margin_core blocked entry: desired_delta=%s side=%s", desired_delta, side)
            else:
                if side.upper() == "BUY":
                    stop_px = last_px - stop_dist
                    target_px = last_px + tp_dist
                else:
                    stop_px = last_px + stop_dist
                    target_px = last_px - tp_dist

                ok, parent_id, stp_id, tgt_id = order_core.place_protected_entry(
                    ib=ib,
                    contract=con,
                    action=side.upper(),
                    qty=final_qty,
                    stop_px=stop_px,
                    target_px=target_px,
                    px_hint=last_px,
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

    # orphan sweep
    if net == 0 and (time.time() - LAST_ORPHAN_SWEEP_TS) >= ORPHAN_SWEEP_COOLDOWN:
        try:
            cancelled = order_core.reconcile_orphans(ib, con, net_qty=net, logger=logger)
            if cancelled and cancelled > 0:
                logger.info("[reconcile_orphans] cancelled %d orphan orders (net=%s)", cancelled, net)
            LAST_ORPHAN_SWEEP_TS = time.time()
        except Exception as e:
            logger.error("[loop] reconcile_orphans error: %s", e)

    # ib error decay
    if last_ib_err is not None:
        try:
            err_ts = dt.datetime.fromisoformat(last_ib_err.get("ts", "1970-01-01T00:00:00"))
        except Exception:
            err_ts = dt.datetime.now()
        if (dt.datetime.now() - err_ts).total_seconds() > IB_ERROR_DECAY_SEC:
            last_ib_err = None

    if (now_ts - last_sharpe_update_ts) >= 5.0:
        sharpe_R_value = compute_sharpe_from_trades(TRADE_LOG_CSV, max_trades=50)
        last_sharpe_update_ts = now_ts

    # runtime state save (throttled)
    state_save_every = float(getattr(args, "state_save_every_sec", STATE_SAVE_EVERY_SEC_DEFAULT) or STATE_SAVE_EVERY_SEC_DEFAULT)
    if state_save_every < 1.0:
        state_save_every = 1.0

    if (now_ts - float(last_state_save_ts or 0.0)) >= state_save_every:
        try:
            ts_list = list(getattr(bars, "ts", []) or [])
            close_list = list(getattr(bars, "close", []) or [])
            start_i = max(0, len(ts_list) - 256)

            bars_tail = []
            for i in range(start_i, len(ts_list)):
                try:
                    ts_i = ts_list[i]
                    close_i = close_list[i]
                    ts_str = ts_i.isoformat(timespec="seconds") if hasattr(ts_i, "isoformat") else str(ts_i)
                    bars_tail.append({"ts": ts_str, "close": float(close_i)})
                except Exception:
                    continue

            runtime_state_out = {
                "day_date": day_date.isoformat() if hasattr(day_date, "isoformat") else str(day_date),
                "trades_today": int(trades_today),
                "running_pnl_today": float(running_pnl_today),
                "wins_today": int(wins_today),
                "losses_today": int(losses_today),
                "day_R": float(getattr(day_risk, "day_R", 0.0) or 0.0),
                "consec_losses": int(getattr(day_risk, "consec_losses", 0) or 0),
                "week_R": float(getattr(week_state, "week_R", 0.0) or 0.0),
                "meta_ema_R": float(getattr(meta, "ema_R", 0.0) or 0.0),
                "meta_n": int(getattr(meta, "n_trades", getattr(meta, "n", 0)) or 0),
                "equity": float(equity),
                "equity_hwm": float(equity_hwm),
                "last_acct_netliq": float(last_acct_netliq) if last_acct_netliq is not None else None,
                "last_regime": str(last_regime or ""),
                "bars": bars_tail,
                "pos_entry_px": float(pos_entry_px) if pos_entry_px is not None else None,
                "pos_entry_ts": str(pos_entry_ts) if pos_entry_ts is not None else None,
            }

            save_runtime_state(RUNTIME_STATE_JSON, runtime_state_out, logger=logger)
            last_state_save_ts = now_ts

        except Exception as e:
            logger.error("[state_core] failed to save runtime state: %s", e)

    ctx.update(
        {
            "BAYES_SOURCE": BAYES_SOURCE,
            "AUTO_FLAT_CT": AUTO_FLAT_CT,
            "day_date": day_date,
            "caps_reset_date": caps_reset_date,
            "bayes_ran_today": bayes_ran_today,
            "safety_halt_for_today": safety_halt_for_today,
            "safety_last_ts": safety_last_ts,
            "equity": equity,
            "equity_hwm": equity_hwm,
            "hwm_factor": hwm_factor,
            "last_acct_netliq": last_acct_netliq,
            "trades_today": trades_today,
            "total_trades": total_trades,
            "running_pnl_today": running_pnl_today,
            "wins_today": wins_today,
            "losses_today": losses_today,
            "last_trade_close_ts": last_trade_close_ts,
            "last_acct_realized": last_acct_realized,
            "current_arm": current_arm,
            "current_side": current_side,
            "last_signal_arm": last_signal_arm,
            "last_signal_side": last_signal_side,
            "last_regime": last_regime,
            "last_atr_points": float(last_atr_points),
            "last_adx_val": float(last_adx_val),
            "week_state": week_state,
            "bandit": bandit,
            "meta": meta,
            "last_fill_ts": last_fill_ts,
            "last_ib_err": last_ib_err,
            "LAST_ORPHAN_SWEEP_TS": LAST_ORPHAN_SWEEP_TS,
            "sharpe_R_value": sharpe_R_value,
            "last_nostop_guard_ts": last_nostop_guard_ts,
            "last_state_save_ts": last_state_save_ts,
            "shadow_enabled": bool(shadow_enabled),
            "meta_factor": float(meta_factor),
            "boost_factor": float(boost_factor),
            "pos_entry_ct": pos_entry_ct,
            "pos_entry_px": pos_entry_px,
            "pos_entry_ts": pos_entry_ts,
            "eod_bayes_attempt_day": ctx.get("eod_bayes_attempt_day"),
            "eod_bayes_attempt_ts": float(ctx.get("eod_bayes_attempt_ts", 0.0) or 0.0),
        }
    )


    # Persist last tick px/day for hybrid daily close approximation on rollover.
    try:
        ctx["_px_at_last_tick"] = float(last_px) if last_px is not None else ctx.get("_px_at_last_tick")
        ctx["_day_date_at_last_tick"] = day_date
    except Exception:
        pass

    try:
        _hb_emit(hb_state="run", idle_reason="", caps=(caps or []) + hard_caps)
    except Exception:
        pass

    time.sleep(1.0)
    return ctx
