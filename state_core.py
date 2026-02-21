#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- state_core.py ---------------------------------------------------------
from __future__ import annotations

import os
import json
import time
import datetime as dt
import tempfile
from typing import Optional, Tuple, Any, Dict

# Canonical default path (used if caller passes empty/None)
RUNTIME_STATE_DEFAULT = os.path.join("data", "state", "runtime_state.json")


# -------------------------------------------------------------------------
# Debug / provenance helper (call once at startup if you want)
# -------------------------------------------------------------------------
def log_state_core_provenance(logger=None) -> None:
    """
    Logs the absolute path of the imported state_core module file so you can
    confirm you're running the expected copy (prevents "stale file" issues).
    """
    try:
        here = os.path.abspath(__file__)
        if logger:
            logger.info("[state_core] module file = %s", here)
        else:
            print("[state_core] module file =", here)
    except Exception:
        pass


# -------------------------------------------------------------------------
# Low-level load/save helpers
# -------------------------------------------------------------------------
def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_runtime_state(runtime_state_json: str, logger=None) -> Optional[dict]:
    """
    Load runtime state JSON from disk. Returns dict or None.
    Never raises (so trading loop can't crash because of state I/O).
    """
    try:
        path = runtime_state_json or RUNTIME_STATE_DEFAULT
        if not path:
            return None
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            if logger:
                logger.error("[state_core] load_runtime_state: JSON was not an object/dict (%s)", path)
            return None
        return data
    except Exception as e:
        if logger:
            logger.error("[state_core] load_runtime_state failed: %s", e)
        return None


def save_runtime_state(runtime_state_json: str, out: dict, logger=None) -> bool:
    """
    Persist runtime state dict to JSON at runtime_state_json (or default path).

    Writes via temp file + replace to reduce partial-write risk.
    Uses fsync for durability before replace.
    Never raises.
    """
    try:
        path = runtime_state_json or RUNTIME_STATE_DEFAULT
        if not path:
            return False

        if out is None or not isinstance(out, dict):
            if logger:
                logger.error("[state_core] save_runtime_state: out is not a dict")
            return False

        _ensure_parent_dir(path)

        # Add a saved timestamp for debugging/provenance
        payload = dict(out)
        payload["_saved_ts"] = time.time()

        parent = os.path.dirname(os.path.abspath(path))
        tmp_dir = parent or "."

        fd, tmp = tempfile.mkstemp(prefix="runtime_state_", suffix=".json", dir=tmp_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception as e:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            if logger:
                logger.error("[state_core] failed to save runtime state: %s", e)
            return False

        if logger:
            logger.debug("[state_core] saved runtime state to %s", path)
        return True

    except Exception as e:
        if logger:
            logger.error("[state_core] failed to save runtime state: %s", e)
        return False


def save_runtime_state_from_ctx(ctx: Dict[str, Any], logger=None) -> bool:
    """
    Convenience wrapper: saves runtime state using paths stored in ctx.

    Expected keys (any one is fine):
      - ctx["runtime_state_out"]  or ctx["RUNTIME_STATE_OUT"]  or ctx["STATE_OUT"]
    And the state dict:
      - ctx["runtime_state"] or ctx["state"]

    This avoids callers referencing an undefined runtime_state_out variable.
    """
    try:
        path = (
            ctx.get("runtime_state_out")
            or ctx.get("RUNTIME_STATE_OUT")
            or ctx.get("STATE_OUT")
            or ""
        )
        state = ctx.get("runtime_state") or ctx.get("state") or None

        if not path:
            # fall back to default path rather than failing hard
            path = RUNTIME_STATE_DEFAULT

        if state is None or not isinstance(state, dict):
            if logger:
                logger.error("[state_core] save_from_ctx: ctx runtime state missing or not a dict")
            return False

        return save_runtime_state(path, state, logger=logger)
    except Exception as e:
        if logger:
            logger.error("[state_core] save_from_ctx failed: %s", e)
        return False


# -------------------------------------------------------------------------
# High-level restore into live objects/scalars
# -------------------------------------------------------------------------
def restore_runtime_into_objects(
    *,
    runtime_state_json: str,
    logger,
    # objects to mutate
    bars,
    day_risk,
    week_state,
    meta,
    # defaults / current values
    default_day_date: dt.date,
    default_trades_today: int,
    default_running_pnl_today: float,
    default_wins_today: int,
    default_losses_today: int,
    default_equity: float,
    default_equity_hwm: float,
    default_last_acct_netliq: Optional[float],
    default_last_regime: str,
) -> Tuple[dt.date, int, float, int, int, float, float, Optional[float], str]:
    """
    Loads runtime_state_json, mutates bars/day_risk/week_state/meta, and returns
    the scalar values that paper_trader owns.

    Returns:
      day_date, trades_today, running_pnl_today, wins_today, losses_today,
      equity, equity_hwm, last_acct_netliq, last_regime
    """
    day_date = default_day_date
    trades_today = default_trades_today
    running_pnl_today = default_running_pnl_today
    wins_today = default_wins_today
    losses_today = default_losses_today
    equity = default_equity
    equity_hwm = default_equity_hwm
    last_acct_netliq = default_last_acct_netliq
    last_regime = default_last_regime

    runtime_state = load_runtime_state(runtime_state_json, logger=logger)
    if not runtime_state:
        return (
            day_date,
            trades_today,
            running_pnl_today,
            wins_today,
            losses_today,
            equity,
            equity_hwm,
            last_acct_netliq,
            last_regime,
        )

    try:
        # day_date
        day_str = runtime_state.get("day_date")
        if isinstance(day_str, str):
            try:
                day_date = dt.date.fromisoformat(day_str)
            except Exception:
                pass

        trades_today = int(runtime_state.get("trades_today", trades_today))
        running_pnl_today = float(runtime_state.get("running_pnl_today", running_pnl_today))
        wins_today = int(runtime_state.get("wins_today", wins_today))
        losses_today = int(runtime_state.get("losses_today", losses_today))

        # day_risk
        day_R_val = runtime_state.get("day_R", None)
        if day_R_val is not None:
            try:
                day_risk.day_R = float(day_R_val)
            except Exception:
                pass

        consec_val = runtime_state.get("consec_losses", None)
        if consec_val is not None and hasattr(day_risk, "consec_losses"):
            try:
                day_risk.consec_losses = int(consec_val)
            except Exception:
                pass

        # week_state
        week_R_val = runtime_state.get("week_R", None)
        if week_R_val is not None and hasattr(week_state, "week_R"):
            try:
                week_state.week_R = float(week_R_val)
            except Exception:
                pass

        # meta
        meta_ema = runtime_state.get("meta_ema_R", None)
        if meta_ema is not None and hasattr(meta, "ema_R"):
            try:
                meta.ema_R = float(meta_ema)
            except Exception:
                pass

        meta_n = runtime_state.get("meta_n", None)
        if meta_n is not None:
            if hasattr(meta, "n_trades"):
                try:
                    meta.n_trades = int(meta_n)
                except Exception:
                    pass
            elif hasattr(meta, "n"):
                try:
                    meta.n = int(meta_n)
                except Exception:
                    pass

        # equity
        eq_val = runtime_state.get("equity", None)
        if eq_val is not None:
            try:
                equity = float(eq_val)
            except Exception:
                pass

        hwm_val = runtime_state.get("equity_hwm", None)
        if hwm_val is not None:
            try:
                equity_hwm = float(hwm_val)
            except Exception:
                pass

        last_netliq_val = runtime_state.get("last_acct_netliq", None)
        if last_netliq_val is not None:
            try:
                last_acct_netliq = float(last_netliq_val)
            except Exception:
                pass

        # regime
        reg_val = runtime_state.get("last_regime", None)
        if isinstance(reg_val, str) and reg_val:
            last_regime = reg_val

        # bars
        bars_state = runtime_state.get("bars", None)
        if isinstance(bars_state, list):
            for row in bars_state:
                if not isinstance(row, dict):
                    continue
                ts_str = row.get("ts")
                close_val = row.get("close")
                if ts_str is None or close_val is None:
                    continue
                try:
                    ts_obj = dt.datetime.fromisoformat(ts_str)
                    close_f = float(close_val)
                except Exception:
                    continue
                try:
                    bars.add(ts_obj, close_f)
                except Exception:
                    continue

        if logger:
            logger.info("[state_core] runtime state restored from %s", (runtime_state_json or RUNTIME_STATE_DEFAULT))

    except Exception as e:
        if logger:
            logger.error("[state_core] restore_runtime_into_objects failed: %s", e)

    return (
        day_date,
        trades_today,
        running_pnl_today,
        wins_today,
        losses_today,
        equity,
        equity_hwm,
        last_acct_netliq,
        last_regime,
    )


# -------------------------------------------------------------------------
# Throttled saver (loop calls this often)
# -------------------------------------------------------------------------
def save_runtime_state_throttled(
    *,
    runtime_state_json: str,
    logger,
    now_ts: float,
    last_save_ts: float,
    min_interval_sec: float,
    # scalars
    day_date,
    trades_today: int,
    running_pnl_today: float,
    wins_today: int,
    losses_today: int,
    equity: float,
    equity_hwm: float,
    last_acct_netliq: Optional[float],
    last_regime: str,
    # objects
    day_risk,
    week_state,
    meta,
    bars,
    max_bars_to_save: int = 256,
) -> float:
    """
    Writes runtime state json at most once per min_interval_sec.
    Returns updated last_save_ts.
    """
    if (now_ts - last_save_ts) < float(min_interval_sec):
        return last_save_ts

    try:
        # Bars snapshot (tail)
        num_bars = len(getattr(bars, "ts", []) or [])
        start_idx = max(0, num_bars - int(max_bars_to_save))

        bars_payload = []
        for i in range(start_idx, num_bars):
            try:
                ts_i = bars.ts[i]
                close_i = bars.close[i]
                bars_payload.append(
                    {"ts": ts_i.isoformat(timespec="seconds"), "close": float(close_i)}
                )
            except Exception:
                continue

        out = {
            "day_date": day_date.isoformat() if hasattr(day_date, "isoformat") else str(day_date),
            "trades_today": int(trades_today),
            "running_pnl_today": float(running_pnl_today),
            "wins_today": int(wins_today),
            "losses_today": int(losses_today),
            "day_R": float(getattr(day_risk, "day_R", 0.0)),
            "consec_losses": int(getattr(day_risk, "consec_losses", 0)),
            "week_R": float(getattr(week_state, "week_R", 0.0)),
            "meta_ema_R": float(getattr(meta, "ema_R", 0.0)),
            "meta_n": int(getattr(meta, "n_trades", getattr(meta, "n", 0))),
            "equity": float(equity),
            "equity_hwm": float(equity_hwm),
            "last_acct_netliq": float(last_acct_netliq) if last_acct_netliq is not None else None,
            "last_regime": str(last_regime or ""),
            "bars": bars_payload,
        }

        ok = save_runtime_state(runtime_state_json or RUNTIME_STATE_DEFAULT, out, logger=logger)
        if ok:
            return float(now_ts)
        return last_save_ts

    except Exception as e:
        if logger:
            logger.error("[state_core] save_runtime_state_throttled failed: %s", e)
        return last_save_ts
