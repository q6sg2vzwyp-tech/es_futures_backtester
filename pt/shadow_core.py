#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shadow_core.py

ShadowSim engine (v1.9) — self-contained (no external shadow model dependencies)

Key guarantees:
- No hard dependency on missing symbols/modules.
- Schema-aligned roundtrip row:
    entry_ts, exit_ts, arm, side, entry_px, exit_px, pnl_usd, R,
    open_gate, close_gate, day, week_R, meta_ema_R, regime
- Robust callback emission:
    * preferred: append_shadow_roundtrip_log(row_dict)
    * supports: append_shadow_roundtrip_log(row=row_dict) and append_shadow_roundtrip_log(**row_dict)
- shadow_enabled=False:
    * if open pos: ONE containment close (close_gate=shadow_mode_off)
    * then NOOP forever until re-enabled
- Stable entry_ts captured at OPEN; fallback to epoch conversion if missing.

NEW in v1.9:
- Shadow overtrading rails:
    * max_roundtrips_per_day / max_roundtrips_per_hour
    * post_close_cooldown_sec
    * post_loss_cooldown_sec
  Enforcement:
    - Entry is blocked when capped or cooling down.
    - Caps/cooldowns appear in open_gate for auditability.
  Counters:
    - shadow_roundtrips_today / shadow_roundtrips_this_hour tracked on CLOSE.
"""

from __future__ import annotations

import os
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


DEFAULT_ROUNDTRIPS_PATH = os.path.join("results", "shadow_roundtrips.csv")
DEFAULT_MODEL_PATH = os.path.join("learn", "shadow_model.json")


@dataclass
class ShadowIndex:
    """
    Compatibility shim expected by shadow_orchestrator/shadow_facade.
    Keep this minimal; orchestrator can store metadata in .meta.
    """
    meta: Dict[str, Any]


def _today_ct() -> dt.date:
    """
    Return today's date in America/Chicago if zoneinfo is available; otherwise local date.
    """
    try:
        from zoneinfo import ZoneInfo  # py3.9+
        return dt.datetime.now(tz=ZoneInfo("America/Chicago")).date()
    except Exception:
        return dt.datetime.now().date()


def _hour_key_ct(now_ct: dt.datetime) -> str:
    """
    Hour bucket key in CT date+hour, stable string for throttling.
    """
    try:
        return now_ct.strftime("%Y-%m-%dT%H")
    except Exception:
        # fall back to epoch hour
        return str(int(time.time() // 3600))


class ShadowSim:
    """
    Lightweight virtual execution engine for shadow learning.

    Behavior:
      - shadow_enabled gating prevents churn/log spam when shadow is OFF
      - Entry is bucket-gated; exits are evaluated on every step while holding.
      - Entry opens 1 contract virtual position (net=+1 long or -1 short)
    """

    def __init__(
        self,
        *,
        roundtrips_csv: str = DEFAULT_ROUNDTRIPS_PATH,
        model_path: str = DEFAULT_MODEL_PATH,
        es_multiplier: float = 50.0,
        model_min_trades: int = 20,
        model_lookback_days: int = 30,
    ) -> None:
        self.roundtrips_csv = str(roundtrips_csv)
        self.model_path = str(model_path)
        self.es_multiplier = float(es_multiplier)

        # compatibility placeholders (unused in self-contained build)
        self.model_min_trades = int(model_min_trades)
        self.model_lookback_days = int(model_lookback_days)
        self.model = None

        # day stats (shadow_trades_today increments on CLOSE; remains for backward-compat)
        self.day_date: dt.date = _today_ct()
        self.shadow_trades_today: int = 0
        self.shadow_R_today: float = 0.0
        self.shadow_pnl_today: float = 0.0

        # NEW: roundtrip counters / throttles
        self.shadow_roundtrips_today: int = 0
        self.shadow_roundtrips_this_hour: int = 0
        self.shadow_hour_key: str = ""
        self.shadow_last_close_ts: float = 0.0
        self.shadow_last_loss_ts: float = 0.0

        # position state
        self.shadow_net: int = 0
        self.shadow_avg_px: float = 0.0
        self.shadow_entry_px: float = 0.0
        self.shadow_entry_epoch: float = 0.0

        self.open_arm: Optional[str] = None
        self.open_side: Optional[str] = None  # "BUY"/"SELL"
        self.open_regime: str = "unknown"
        self.open_gate: str = "none"
        self.open_entry_ts: Optional[str] = None

        # diagnostics
        self.shadow_last_action: str = "INIT"
        self.shadow_last_realized: float = 0.0
        self.shadow_last_eval_ts: float = 0.0
        self.shadow_eval_count_today: int = 0
        self.shadow_step_reason: str = ""

        # bucket gating (entry only)
        self._last_bucket_key: Optional[int] = None

        # prevents repeated “mode off” closes/logs
        self._shadow_off_closed: bool = False

    def reset_day(self) -> None:
        self.day_date = _today_ct()
        self.shadow_trades_today = 0
        self.shadow_R_today = 0.0
        self.shadow_pnl_today = 0.0

        self.shadow_roundtrips_today = 0
        self.shadow_roundtrips_this_hour = 0
        self.shadow_hour_key = ""
        self.shadow_last_close_ts = 0.0
        self.shadow_last_loss_ts = 0.0

        self.shadow_net = 0
        self.shadow_avg_px = 0.0
        self.shadow_entry_px = 0.0
        self.shadow_entry_epoch = 0.0

        self.open_arm = None
        self.open_side = None
        self.open_regime = "unknown"
        self.open_gate = "none"
        self.open_entry_ts = None

        self.shadow_last_action = "RESET"
        self.shadow_last_realized = 0.0
        self.shadow_last_eval_ts = 0.0
        self.shadow_eval_count_today = 0
        self.shadow_step_reason = ""

        self._last_bucket_key = None
        self._shadow_off_closed = False

    def heartbeat_fields(self) -> Dict[str, Any]:
        return {
            "shadow_pnl_today": float(self.shadow_pnl_today),
            "shadow_R_today": float(self.shadow_R_today),
            "shadow_trades_today": int(self.shadow_trades_today),
            "shadow_net": int(self.shadow_net),
            "shadow_avg_px": float(self.shadow_avg_px),
            "shadow_entry_px": float(self.shadow_entry_px),
            "shadow_last_action": str(self.shadow_last_action),
            "shadow_last_realized": float(self.shadow_last_realized),
            "shadow_last_eval_ts": float(self.shadow_last_eval_ts),
            "shadow_eval_count_today": int(self.shadow_eval_count_today),
            "shadow_step_reason": str(self.shadow_step_reason),
            "shadow_roundtrips_csv": self.roundtrips_csv,

            # NEW: throttling introspection
            "shadow_roundtrips_today": int(self.shadow_roundtrips_today),
            "shadow_roundtrips_this_hour": int(self.shadow_roundtrips_this_hour),
            "shadow_hour_key": str(self.shadow_hour_key),
            "shadow_last_close_ts": float(self.shadow_last_close_ts),
            "shadow_last_loss_ts": float(self.shadow_last_loss_ts),
        }

    # Compatibility no-ops
    def maybe_update_model_eod(self, logger=None) -> None:
        return

    # Minimal persistence helpers for compatibility exports
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "shadow_core_v1.9",
            "roundtrips_csv": self.roundtrips_csv,
            "model_path": self.model_path,
            "es_multiplier": self.es_multiplier,
            "model_min_trades": self.model_min_trades,
            "model_lookback_days": self.model_lookback_days,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadowSim":
        return cls(
            roundtrips_csv=data.get("roundtrips_csv", DEFAULT_ROUNDTRIPS_PATH),
            model_path=data.get("model_path", DEFAULT_MODEL_PATH),
            es_multiplier=float(data.get("es_multiplier", 50.0)),
            model_min_trades=int(data.get("model_min_trades", 20)),
            model_lookback_days=int(data.get("model_lookback_days", 30)),
        )

    def entry_multiplier(
        self,
        *,
        regime: str,
        arm: str,
        side: str,
        default: float = 1.0,
        min_trades: int = 10,
    ) -> Tuple[float, Optional[str]]:
        return float(default), None

    def _ensure_day_roll(self, now_ct: dt.datetime) -> None:
        try:
            d = now_ct.date()
        except Exception:
            d = _today_ct()
        if d != self.day_date:
            self.reset_day()

    def _ensure_hour_roll(self, now_ct: dt.datetime) -> None:
        hk = _hour_key_ct(now_ct)
        if hk != self.shadow_hour_key:
            self.shadow_hour_key = hk
            self.shadow_roundtrips_this_hour = 0

    def _emit_roundtrip_row(
        self,
        *,
        row: Dict[str, Any],
        append_shadow_roundtrip_log: Optional[Callable[..., Any]],
    ) -> None:
        """
        Robust emitter supporting:
          - cb(row_dict)
          - cb(row=row_dict)
          - cb(**row_dict)
        """
        if append_shadow_roundtrip_log is None:
            return

        try:
            append_shadow_roundtrip_log(row)
            return
        except TypeError:
            pass
        except Exception:
            return

        try:
            append_shadow_roundtrip_log(row=row)
            return
        except TypeError:
            pass
        except Exception:
            return

        try:
            append_shadow_roundtrip_log(**row)
        except Exception:
            return

    def _entry_ts_for_row(self, *, now_ct: dt.datetime) -> str:
        if self.open_entry_ts:
            return str(self.open_entry_ts)

        try:
            ep = float(self.shadow_entry_epoch or 0.0)
            if ep > 0:
                tz = getattr(now_ct, "tzinfo", None)
                if tz is not None:
                    return dt.datetime.fromtimestamp(ep, tz=tz).isoformat(timespec="seconds")
                return dt.datetime.fromtimestamp(ep).isoformat(timespec="seconds")
        except Exception:
            pass

        return now_ct.isoformat(timespec="seconds")

    def _close_position(
        self,
        *,
        now_ct: dt.datetime,
        last_px: float,
        per_contract_init: float,
        reason: str,
        week_R: float,
        meta_ema_R: float,
        append_shadow_roundtrip_log: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        if self.shadow_net == 0:
            self.shadow_last_action = "NOOP"
            self.shadow_last_realized = 0.0
            return self.heartbeat_fields()

        # ensure hour accounting is correct at close time
        self._ensure_hour_roll(now_ct)

        entry_px = float(self.shadow_entry_px or self.shadow_avg_px or last_px)
        exit_px = float(last_px)

        qty = abs(int(self.shadow_net)) or 1
        side = "BUY" if self.shadow_net > 0 else "SELL"

        pnl_points = (exit_px - entry_px) * float(self.shadow_net)
        pnl_usd = pnl_points * self.es_multiplier * qty

        R = 0.0
        if per_contract_init and per_contract_init > 0:
            R = float(pnl_usd) / float(per_contract_init)

        # Backward-compat counters
        self.shadow_trades_today += 1
        self.shadow_pnl_today += float(pnl_usd)
        self.shadow_R_today += float(R)

        # NEW: roundtrip counters / cooldown stamps
        self.shadow_roundtrips_today += 1
        self.shadow_roundtrips_this_hour += 1
        self.shadow_last_close_ts = float(time.time())
        if float(pnl_usd) < 0:
            self.shadow_last_loss_ts = float(self.shadow_last_close_ts)

        self.shadow_last_action = f"EXIT:{reason}"
        self.shadow_last_realized = float(pnl_usd)

        row = {
            "entry_ts": self._entry_ts_for_row(now_ct=now_ct),
            "exit_ts": now_ct.isoformat(timespec="seconds"),
            "arm": (self.open_arm or "unknown"),
            "side": (self.open_side or side).upper(),
            "entry_px": float(entry_px),
            "exit_px": float(exit_px),
            "pnl_usd": float(pnl_usd),
            "R": float(R),
            "open_gate": str(self.open_gate or "none"),
            "close_gate": str(reason),
            "day": now_ct.date().isoformat(),
            "week_R": float(week_R),
            "meta_ema_R": float(meta_ema_R),
            "regime": str(self.open_regime or "unknown"),
        }
        self._emit_roundtrip_row(row=row, append_shadow_roundtrip_log=append_shadow_roundtrip_log)

        # reset pos
        self.shadow_net = 0
        self.shadow_avg_px = 0.0
        self.shadow_entry_px = 0.0
        self.shadow_entry_epoch = 0.0
        self.open_arm = None
        self.open_side = None
        self.open_regime = "unknown"
        self.open_gate = "none"
        self.open_entry_ts = None

        return self.heartbeat_fields()

    def _entry_block_reason(
        self,
        *,
        now_ct: dt.datetime,
        now_epoch: float,
        max_roundtrips_per_day: int,
        max_roundtrips_per_hour: int,
        post_close_cooldown_sec: int,
        post_loss_cooldown_sec: int,
    ) -> Optional[str]:
        """
        Returns a short reason string if entry should be blocked, else None.
        """
        self._ensure_hour_roll(now_ct)

        # caps
        if max_roundtrips_per_day > 0 and self.shadow_roundtrips_today >= max_roundtrips_per_day:
            return "cap_shadow_rts_day"
        if max_roundtrips_per_hour > 0 and self.shadow_roundtrips_this_hour >= max_roundtrips_per_hour:
            return "cap_shadow_rts_hour"

        # cooldowns
        if post_close_cooldown_sec > 0 and self.shadow_last_close_ts > 0:
            if (now_epoch - float(self.shadow_last_close_ts)) < float(post_close_cooldown_sec):
                return "cooldown_shadow_post_close"

        if post_loss_cooldown_sec > 0 and self.shadow_last_loss_ts > 0:
            if (now_epoch - float(self.shadow_last_loss_ts)) < float(post_loss_cooldown_sec):
                return "cooldown_shadow_post_loss"

        return None

    def step(
        self,
        *,
        now_ct: dt.datetime,
        last_px: float,
        bars: Any = None,
        bars_15m: Any = None,
        in_shadow_window: bool,
        arm: Optional[str],
        side: Optional[str],
        per_contract_init: float,
        last_regime: str,
        week_R: float,
        meta_ema_R: float,
        append_shadow_roundtrip_log: Optional[Callable[..., Any]] = None,
        atr_points: float = 0.0,
        tick_size: float = 0.25,
        decision_bucket_sec: int = 30,
        min_hold_sec: int = 120,
        atr_floor_ticks: float = 2.0,
        max_hold_sec: Optional[int] = None,
        shadow_enabled: bool = True,

        # NEW: shadow overtrading rails (optional; default permissive)
        max_roundtrips_per_day: int = 0,
        max_roundtrips_per_hour: int = 0,
        post_close_cooldown_sec: int = 0,
        post_loss_cooldown_sec: int = 0,
    ) -> Dict[str, Any]:
        """
        Main shadow step called by loop_core.

        shadow_enabled=False behavior:
          - If no open position: NOOP, no logs
          - If open position: ONE containment close (shadow_mode_off) then NOOP thereafter

        Overtrading rails:
          - Caps/cooldowns only block ENTRY (never block EXIT).
          - Reasons are surfaced via open_gate when an OPEN is attempted.
        """
        self._ensure_day_roll(now_ct)
        self._ensure_hour_roll(now_ct)

        now_epoch = time.time()
        self.shadow_last_eval_ts = float(now_epoch)
        self.shadow_eval_count_today += 1

        # ---- HARD GATE: shadow disabled ----
        if not bool(shadow_enabled):
            self.shadow_step_reason = "shadow_mode_off"

            if self.shadow_net != 0 and not self._shadow_off_closed:
                self._shadow_off_closed = True
                return self._close_position(
                    now_ct=now_ct,
                    last_px=float(last_px),
                    per_contract_init=float(per_contract_init),
                    reason="shadow_mode_off",
                    week_R=float(week_R),
                    meta_ema_R=float(meta_ema_R),
                    append_shadow_roundtrip_log=append_shadow_roundtrip_log,
                )

            self.shadow_last_action = "NOOP"
            self.shadow_last_realized = 0.0
            return self.heartbeat_fields()

        # shadow enabled again -> release latch
        self._shadow_off_closed = False

        # default max hold
        if max_hold_sec is None:
            max_hold_sec = max(600, int(5 * max(1, int(min_hold_sec))))

        regime = (last_regime or "unknown").strip() or "unknown"

        # ---- Outside shadow window -> force-flat only ----
        if not bool(in_shadow_window):
            self.shadow_step_reason = "force_flat"
            if self.shadow_net != 0:
                return self._close_position(
                    now_ct=now_ct,
                    last_px=float(last_px),
                    per_contract_init=float(per_contract_init),
                    reason="force_flat",
                    week_R=float(week_R),
                    meta_ema_R=float(meta_ema_R),
                    append_shadow_roundtrip_log=append_shadow_roundtrip_log,
                )
            self.shadow_last_action = "NOOP"
            self.shadow_last_realized = 0.0
            return self.heartbeat_fields()

        # classify this call (for open_gate)
        call_gate = "signal" if (arm and side) else "bucket"
        self.shadow_step_reason = call_gate

        # ---- EXIT LOGIC (while holding) ----
        if self.shadow_net != 0:
            hold_sec = 0.0
            if self.shadow_entry_epoch and self.shadow_entry_epoch > 0:
                hold_sec = float(now_epoch - float(self.shadow_entry_epoch))

            atr_points_f = float(atr_points or 0.0)
            tick_size_f = float(tick_size or 0.25)
            atr_floor_points = float(atr_floor_ticks) * tick_size_f

            stop_dist = max(atr_points_f, atr_floor_points)
            target_dist = stop_dist

            entry_px = float(self.shadow_entry_px or self.shadow_avg_px or last_px)

            if hold_sec >= float(min_hold_sec):
                if self.shadow_net > 0:
                    if float(last_px) <= (entry_px - stop_dist):
                        return self._close_position(
                            now_ct=now_ct,
                            last_px=float(last_px),
                            per_contract_init=float(per_contract_init),
                            reason="stop",
                            week_R=float(week_R),
                            meta_ema_R=float(meta_ema_R),
                            append_shadow_roundtrip_log=append_shadow_roundtrip_log,
                        )
                    if float(last_px) >= (entry_px + target_dist):
                        return self._close_position(
                            now_ct=now_ct,
                            last_px=float(last_px),
                            per_contract_init=float(per_contract_init),
                            reason="target",
                            week_R=float(week_R),
                            meta_ema_R=float(meta_ema_R),
                            append_shadow_roundtrip_log=append_shadow_roundtrip_log,
                        )
                else:
                    if float(last_px) >= (entry_px + stop_dist):
                        return self._close_position(
                            now_ct=now_ct,
                            last_px=float(last_px),
                            per_contract_init=float(per_contract_init),
                            reason="stop",
                            week_R=float(week_R),
                            meta_ema_R=float(meta_ema_R),
                            append_shadow_roundtrip_log=append_shadow_roundtrip_log,
                        )
                    if float(last_px) <= (entry_px - target_dist):
                        return self._close_position(
                            now_ct=now_ct,
                            last_px=float(last_px),
                            per_contract_init=float(per_contract_init),
                            reason="target",
                            week_R=float(week_R),
                            meta_ema_R=float(meta_ema_R),
                            append_shadow_roundtrip_log=append_shadow_roundtrip_log,
                        )

                if hold_sec >= float(max_hold_sec):
                    return self._close_position(
                        now_ct=now_ct,
                        last_px=float(last_px),
                        per_contract_init=float(per_contract_init),
                        reason="time",
                        week_R=float(week_R),
                        meta_ema_R=float(meta_ema_R),
                        append_shadow_roundtrip_log=append_shadow_roundtrip_log,
                    )

            self.shadow_last_action = "NOOP"
            self.shadow_last_realized = 0.0
            return self.heartbeat_fields()

        # ---- ENTRY LOGIC (bucket-gated) ----
        try:
            bucket_key = int(now_epoch // max(1, int(decision_bucket_sec)))
        except Exception:
            bucket_key = None

        if bucket_key is not None and bucket_key == self._last_bucket_key:
            self.shadow_last_action = "NOOP"
            self.shadow_last_realized = 0.0
            return self.heartbeat_fields()

        self._last_bucket_key = bucket_key

        # Only enter on concrete (arm, side)
        if not arm or not side:
            self.shadow_last_action = "NOOP"
            self.shadow_last_realized = 0.0
            return self.heartbeat_fields()

        side_u = str(side).upper().strip()
        if side_u not in ("BUY", "SELL"):
            self.shadow_last_action = "NOOP"
            self.shadow_last_realized = 0.0
            return self.heartbeat_fields()

        # ---- Overtrading rails (ENTRY ONLY) ----
        # If loop_core/paper_trader sets these on the object, allow those values as fallback.
        try:
            if not max_roundtrips_per_day:
                max_roundtrips_per_day = int(getattr(self, "max_roundtrips_per_day", 0) or 0)
            if not max_roundtrips_per_hour:
                max_roundtrips_per_hour = int(getattr(self, "max_roundtrips_per_hour", 0) or 0)
            if not post_close_cooldown_sec:
                post_close_cooldown_sec = int(getattr(self, "post_close_cooldown_sec", 0) or 0)
            if not post_loss_cooldown_sec:
                post_loss_cooldown_sec = int(getattr(self, "post_loss_cooldown_sec", 0) or 0)
        except Exception:
            pass

        block = self._entry_block_reason(
            now_ct=now_ct,
            now_epoch=float(now_epoch),
            max_roundtrips_per_day=int(max_roundtrips_per_day or 0),
            max_roundtrips_per_hour=int(max_roundtrips_per_hour or 0),
            post_close_cooldown_sec=int(post_close_cooldown_sec or 0),
            post_loss_cooldown_sec=int(post_loss_cooldown_sec or 0),
        )
        if block:
            self.shadow_step_reason = block
            self.shadow_last_action = f"BLOCK:{block}"  # instead of "NOOP"
            self.shadow_last_realized = 0.0
            return self.heartbeat_fields()

        # Open 1 contract virtual position
        self.shadow_net = 1 if side_u == "BUY" else -1
        self.shadow_entry_px = float(last_px)
        self.shadow_avg_px = float(last_px)
        self.shadow_entry_epoch = float(now_epoch)

        self.open_arm = str(arm).strip()
        self.open_side = side_u
        self.open_regime = regime
        # Record why this OPEN was permitted (useful for audit)
        self.open_gate = str(call_gate or "none")
        self.open_entry_ts = now_ct.isoformat(timespec="seconds")

        self.shadow_last_action = "OPEN"
        self.shadow_last_realized = 0.0
        return self.heartbeat_fields()


# -----------------------------
# Compatibility exports (module)
# -----------------------------

def load_shadow_model(
    model_json_path: str = DEFAULT_MODEL_PATH,
    logger=None
) -> Tuple[Optional[ShadowSim], ShadowIndex]:
    path = str(model_json_path or "").strip() or DEFAULT_MODEL_PATH
    abspath = os.path.abspath(path)

    if not os.path.exists(abspath):
        if logger:
            logger.warning("[shadow] load_shadow_model: missing %s (starting fresh).", abspath)
        return None, ShadowIndex(meta={"path": abspath, "loaded": False, "reason": "missing"})

    try:
        with open(abspath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        if logger:
            logger.exception("[shadow] load_shadow_model: read failed %s: %s", abspath, e)
        return None, ShadowIndex(meta={"path": abspath, "loaded": False, "reason": f"read_error:{e}"})

    try:
        if isinstance(data, dict) and hasattr(ShadowSim, "from_dict"):
            sim = ShadowSim.from_dict(data)
            return sim, ShadowIndex(meta={"path": abspath, "loaded": True})
        return None, ShadowIndex(meta={"path": abspath, "loaded": False, "reason": "no_from_dict"})
    except Exception as e:
        if logger:
            logger.exception("[shadow] load_shadow_model: hydrate failed: %s", e)
        return None, ShadowIndex(meta={"path": abspath, "loaded": False, "reason": f"hydrate_error:{e}"})


def update_shadow_model(
    model_json_path: str = DEFAULT_MODEL_PATH,
    shadow_sim: Optional[ShadowSim] = None,
    shadow_index: Optional[ShadowIndex] = None,
    logger=None,
) -> ShadowIndex:
    if shadow_index is None:
        shadow_index = ShadowIndex(meta={})

    path = str(model_json_path or "").strip() or DEFAULT_MODEL_PATH
    abspath = os.path.abspath(path)
    os.makedirs(os.path.dirname(abspath), exist_ok=True)

    payload: Dict[str, Any] = {"version": "shadow_core_v1.9", "_note": "no shadow_sim provided"}
    if shadow_sim is not None and hasattr(shadow_sim, "to_dict"):
        try:
            payload = shadow_sim.to_dict()
        except Exception as e:
            if logger:
                logger.exception("[shadow] update_shadow_model: export failed: %s", e)
            shadow_index.meta.update({"path": abspath, "saved": False, "reason": f"export_error:{e}"})
            return shadow_index

    try:
        with open(abspath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        shadow_index.meta.update({"path": abspath, "saved": True})
    except Exception as e:
        if logger:
            logger.exception("[shadow] update_shadow_model: write failed %s: %s", abspath, e)
        shadow_index.meta.update({"path": abspath, "saved": False, "reason": f"write_error:{e}"})

    return shadow_index
