# pt/ctx_builder.py
# Centralized ctx construction for pt.loop_core.run_loop(ctx)
# Goal: keep paper_trader.py thin and avoid scattering "missing symbol" fixes.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import os
import time

from pt.risk_engine import DayRisk
from pt.week_guard import init_week_state
from pt.heartbeat import init_heartbeat, start_heartbeat_thread

# NOTE: loop_core imports BarBuffer internally; paper_trader historically uses strategy_core.BarBuffer.
try:
    from strategy_core import BarBuffer  # type: ignore
except Exception:  # pragma: no cover
    BarBuffer = None  # type: ignore

def _env_on(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def _default_path(rel: str) -> str:
    # Keep paths consistent with repo scripts; use Windows-friendly relative paths.
    return os.path.join(".", rel).replace("/", os.sep)

@dataclass
class CtxDefaults:
    HB_PATH: str = _default_path(r"run\heartbeat.txt")
    HB_KV_PATH: str = _default_path(r"run\heartbeat_kv.txt")
    TRADE_LOG_CSV: str = _default_path(r"logs\trades.csv")
    RUNTIME_STATE_JSON: str = _default_path(r"run\runtime_state.json")
    DAILY_RESTART_JSON: str = _default_path(r"run\daily_restart.json")

    # Times are Central Time (CT) strings parsed in loop_core/utils.
    SESSION_START_CT: str = "08:30"
    SESSION_END_CT: str = "15:15"
    SHADOW_START_CT: str = "08:30"
    SHADOW_END_CT: str = "15:15"
    DAILY_RESTART_CT: str = "16:00"
    AUTO_FLAT_CT: str = "15:55"
    PRE_CLOSE_SWEEP_CT: str = "15:57"

    # Risk/ops constants
    IB_ERROR_DECAY_SEC: float = 60.0
    ORPHAN_SWEEP_COOLDOWN: float = 30.0
    STATE_SAVE_EVERY_SEC: float = 5.0

    # Learning paths (safe defaults; loop_core may override)
    LEARN_MODEL_PATH: str = _default_path(r"logs\learn\strategy_thompson.json")

class _MarginCompat:
    """Minimal margin manager shim so loop_core can run without hard dependency churn."""
    def __init__(self) -> None:
        self._last = None

    def heartbeat_fields(self) -> Dict[str, Any]:
        return {}

    def update_snapshot(self, snap: Any) -> None:
        self._last = snap

    def clamp_entry_size(
        self,
        *,
        product: str,
        desired_qty_delta: int,
        current_net_qty: int,
        per_contract_init: Optional[float] = None,
    ) -> int:
        # permissive: no clamping in stability mode
        return int(desired_qty_delta)

def build_ctx(
    *,
    args: Any,
    log_fn: Callable[..., Any],
    logger_obj: Any,
    ib: Any,
    con: Any,
    ticker: Any,
    bars: Any,
    day_risk: Optional[DayRisk] = None,
    week_state: Optional[Dict[str, Any]] = None,
    bandit: Any = None,
    meta: Optional[Dict[str, Any]] = None,
    shadow: Optional[Dict[str, Any]] = None,
    margin_mgr: Any = None,
    is_us_market_holiday: Optional[Callable[..., Any]] = None,
    build_and_write_heartbeat: Optional[Callable[..., Any]] = None,
    build_bandit_hb_fields: Optional[Callable[..., Any]] = None,
    roll_week_if_needed: Optional[Callable[..., Any]] = None,
    maybe_daily_restart: Optional[Callable[..., Any]] = None,
    defaults: Optional[CtxDefaults] = None,
) -> Dict[str, Any]:
    """Return a ctx dict compatible with pt.loop_core.run_loop_iteration()."""
    d = defaults or CtxDefaults()

    # Initialize heartbeat paths in centralized module (pt.heartbeat writes heartbeat.txt + kv).
    try:
        init_heartbeat(log_fn, ct_now_fn=None, hb_json_path=d.HB_PATH, hb_kv_path=d.HB_KV_PATH)
        start_heartbeat_thread()
    except Exception:
        pass

    if day_risk is None:
        day_risk = DayRisk()

    if week_state is None:
        try:
            # init_week_state(restored_week_R, restored_week_id, today_date, args)
            restored_week_R = float(getattr(args, "restored_week_R", 0.0) or 0.0)
            restored_week_id = str(getattr(args, "restored_week_id", "") or "")
            today = getattr(args, "today_date", None)
            week_state = init_week_state(restored_week_R, restored_week_id, today, args)
        except Exception:
            week_state = {"week_R": 0.0, "week_halted": False, "last_week_id": "", "weekly_cap_R": 0.0}

    if meta is None:
        meta = {}
    if shadow is None:
        # stability mode: keep default shadow off unless args.learn_mode == "shadow"
        lm = str(getattr(args, "learn_mode", "") or "").lower()
        shadow = {"enabled": (lm == "shadow")}

    if margin_mgr is None:
        margin_mgr = _MarginCompat()

    # lazily bind helper functions from loop_core if not passed
    if any(x is None for x in (build_and_write_heartbeat, build_bandit_hb_fields, roll_week_if_needed, maybe_daily_restart)):
        try:
            from pt import loop_core as _lc  # local import to avoid cycle at module import
            build_and_write_heartbeat = build_and_write_heartbeat or getattr(_lc, "build_and_write_heartbeat", None)
            build_bandit_hb_fields = build_bandit_hb_fields or getattr(_lc, "build_bandit_hb_fields", None)
            roll_week_if_needed = roll_week_if_needed or getattr(_lc, "roll_week_if_needed", None)
            maybe_daily_restart = maybe_daily_restart or getattr(_lc, "maybe_daily_restart", None)
        except Exception:
            pass

    # day/eod policy state placeholders (loop_core will mutate)
    day_policy_state = {}
    eod_state = {}

    ctx: Dict[str, Any] = {
        "args": args,
        "logger": logger_obj,
        "ib": ib,
        "con": con,
        "ticker": ticker,
        "bars": bars,

        "day_risk": day_risk,
        "week_state": week_state,
        "bandit": bandit,
        "meta": meta,
        "shadow": shadow,
        "margin_mgr": margin_mgr,

        "day_policy_state": day_policy_state,
        "eod_state": eod_state,

        # paths / constants loop_core requires
        "HB_PATH": d.HB_PATH,
        "TRADE_LOG_CSV": d.TRADE_LOG_CSV,
        "RUNTIME_STATE_JSON": d.RUNTIME_STATE_JSON,
        "DAILY_RESTART_JSON": d.DAILY_RESTART_JSON,
        "DAILY_RESTART_CT": d.DAILY_RESTART_CT,
        "AUTO_FLAT_CT": d.AUTO_FLAT_CT,
        "SHADOW_START_CT": d.SHADOW_START_CT,
        "SHADOW_END_CT": d.SHADOW_END_CT,
        "SESSION_START_CT": d.SESSION_START_CT,
        "SESSION_END_CT": d.SESSION_END_CT,
        "PRE_CLOSE_SWEEP_CT": d.PRE_CLOSE_SWEEP_CT,
        "IB_ERROR_DECAY_SEC": float(d.IB_ERROR_DECAY_SEC),
        "ORPHAN_SWEEP_COOLDOWN": float(d.ORPHAN_SWEEP_COOLDOWN),
        "STATE_SAVE_EVERY_SEC": float(d.STATE_SAVE_EVERY_SEC),
        "LEARN_MODEL_PATH": d.LEARN_MODEL_PATH,
    }

    if is_us_market_holiday is not None:
        ctx["is_us_market_holiday"] = is_us_market_holiday
    if build_and_write_heartbeat is not None:
        ctx["build_and_write_heartbeat"] = build_and_write_heartbeat
    if build_bandit_hb_fields is not None:
        ctx["build_bandit_hb_fields"] = build_bandit_hb_fields
    if roll_week_if_needed is not None:
        ctx["roll_week_if_needed"] = roll_week_if_needed
    if maybe_daily_restart is not None:
        ctx["maybe_daily_restart"] = maybe_daily_restart

    # A few optional runtime keys loop_core expects to exist if it uses them
    ctx.setdefault("LAST_ORPHAN_SWEEP_TS", 0.0)
    ctx.setdefault("bars15m_seed_last_log_ts", 0.0)
    ctx.setdefault("bars15m_seed_last_try_ts", 0.0)
    ctx.setdefault("bayes_ran_today", False)

    return ctx
