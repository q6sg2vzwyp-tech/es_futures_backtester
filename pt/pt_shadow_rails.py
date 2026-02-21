#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pt_shadow_rails.py

Shadow stepping + rails pass-through extracted from loop_core.py.

Best-effort: never raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time
import datetime as dt

SHADOW_DECISION_BUCKET_SEC_DEFAULT = 60
SHADOW_MIN_HOLD_SEC_DEFAULT = 300


@dataclass(frozen=True)
class ShadowRails:
    decision_bucket_sec: int = SHADOW_DECISION_BUCKET_SEC_DEFAULT
    min_hold_sec: int = SHADOW_MIN_HOLD_SEC_DEFAULT
    max_hold_sec: Optional[int] = None
    max_roundtrips_per_day: int = 0
    max_roundtrips_per_hour: int = 0
    post_close_cooldown_sec: int = 0
    post_loss_cooldown_sec: int = 0

    @staticmethod
    def from_args(args: Any) -> "ShadowRails":
        def _i(v, default: int) -> int:
            try:
                return int(v)
            except Exception:
                return int(default)

        decision_bucket_sec = _i(getattr(args, "shadow_decision_bucket_sec", SHADOW_DECISION_BUCKET_SEC_DEFAULT),
                                 SHADOW_DECISION_BUCKET_SEC_DEFAULT) or SHADOW_DECISION_BUCKET_SEC_DEFAULT
        min_hold_sec = _i(getattr(args, "shadow_min_hold_sec", SHADOW_MIN_HOLD_SEC_DEFAULT),
                          SHADOW_MIN_HOLD_SEC_DEFAULT) or SHADOW_MIN_HOLD_SEC_DEFAULT

        max_hold = getattr(args, "shadow_max_hold_sec", None)
        try:
            max_hold_sec = int(max_hold) if max_hold is not None else None
        except Exception:
            max_hold_sec = None

        return ShadowRails(
            decision_bucket_sec=decision_bucket_sec,
            min_hold_sec=min_hold_sec,
            max_hold_sec=max_hold_sec,
            max_roundtrips_per_day=_i(getattr(args, "shadow_max_roundtrips_per_day", 0), 0),
            max_roundtrips_per_hour=_i(getattr(args, "shadow_max_roundtrips_per_hour", 0), 0),
            post_close_cooldown_sec=_i(getattr(args, "shadow_post_close_cooldown_sec", 0), 0),
            post_loss_cooldown_sec=_i(getattr(args, "shadow_post_loss_cooldown_sec", 0), 0),
        )

    def apply_to_shadow(self, shadow: Any) -> None:
        for k, v in (
            ("max_roundtrips_per_day", self.max_roundtrips_per_day),
            ("max_roundtrips_per_hour", self.max_roundtrips_per_hour),
            ("post_close_cooldown_sec", self.post_close_cooldown_sec),
            ("post_loss_cooldown_sec", self.post_loss_cooldown_sec),
        ):
            try:
                setattr(shadow, k, v)
            except Exception:
                pass

    def step(self, *, shadow: Any, now_ct: dt.datetime, last_px: float, in_shadow_window: bool,
             arm: Optional[str], side: Optional[str], per_contract_init: float, last_regime: str,
             week_R: float, meta_ema_R: float, append_shadow_roundtrip_log, atr_points: float,
             tick_size: float, atr_floor_ticks: float = 2.0, shadow_enabled: bool) -> Dict[str, Any]:
        return dict(
            shadow.step(
                now_ct=now_ct,
                last_px=last_px,
                in_shadow_window=bool(in_shadow_window),
                arm=arm,
                side=side,
                per_contract_init=float(per_contract_init),
                last_regime=last_regime,
                week_R=float(week_R),
                meta_ema_R=float(meta_ema_R),
                append_shadow_roundtrip_log=append_shadow_roundtrip_log,
                atr_points=float(atr_points),
                tick_size=float(tick_size),
                decision_bucket_sec=int(self.decision_bucket_sec),
                min_hold_sec=int(self.min_hold_sec),
                atr_floor_ticks=float(atr_floor_ticks),
                max_hold_sec=self.max_hold_sec,
                shadow_enabled=bool(shadow_enabled),
                max_roundtrips_per_day=int(self.max_roundtrips_per_day),
                max_roundtrips_per_hour=int(self.max_roundtrips_per_hour),
                post_close_cooldown_sec=int(self.post_close_cooldown_sec),
                post_loss_cooldown_sec=int(self.post_loss_cooldown_sec),
            )
        )


def _merge(ctx: Dict[str, Any], sh_status: Any) -> None:
    try:
        ctx["shadow_last_status"] = dict(sh_status) if isinstance(sh_status, dict) else {}
    except Exception:
        ctx["shadow_last_status"] = {}


def step_force_path(*, ctx: Dict[str, Any], now_ct: dt.datetime, last_px: float, per_contract_init: float,
                    last_regime: str, shadow_enabled: bool, rails: ShadowRails, logger: Any) -> None:
    try:
        shadow = ctx["shadow"]
        week_state = ctx["week_state"]
        meta = ctx["meta"]
        sh_status = rails.step(
            shadow=shadow,
            now_ct=now_ct,
            last_px=last_px,
            in_shadow_window=False,
            arm=None,
            side=None,
            per_contract_init=per_contract_init,
            last_regime=last_regime,
            week_R=float(getattr(week_state, "week_R", 0.0) or 0.0),
            meta_ema_R=float(getattr(meta, "ema_R", 0.0) or 0.0),
            append_shadow_roundtrip_log=ctx["append_shadow_roundtrip_log"],
            atr_points=float(ctx.get("last_atr_points", 0.0) or 0.0),
            tick_size=float(getattr(ctx["args"], "tick_size", 0.25) or 0.25),
            shadow_enabled=shadow_enabled,
        )
        _merge(ctx, sh_status)
    except Exception as e:
        try:
            logger.error("[shadow] step (force-path) failed: %s", e)
        except Exception:
            pass


def step_in_window_if_needed(*, ctx: Dict[str, Any], now_ct: dt.datetime, last_px: float,
                             arm: Optional[str], side: Optional[str], per_contract_init: float,
                             last_regime: str, shadow_enabled: bool, rails: ShadowRails, logger: Any) -> None:
    try:
        shadow = ctx["shadow"]
        week_state = ctx["week_state"]
        meta = ctx["meta"]

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
            if (time.time() - last_eval) >= float(rails.decision_bucket_sec):
                should_step = True
                step_reason = "bucket"

        if not should_step:
            return

        reg = ("chop" if (not last_regime or last_regime == "unknown") else last_regime)

        sh_status = rails.step(
            shadow=shadow,
            now_ct=now_ct,
            last_px=last_px,
            in_shadow_window=True,
            arm=arm,
            side=side,
            per_contract_init=per_contract_init,
            last_regime=reg,
            week_R=float(getattr(week_state, "week_R", 0.0) or 0.0),
            meta_ema_R=float(getattr(meta, "ema_R", 0.0) or 0.0),
            append_shadow_roundtrip_log=ctx["append_shadow_roundtrip_log"],
            atr_points=float(ctx.get("last_atr_points", 0.0) or 0.0),
            tick_size=float(getattr(ctx["args"], "tick_size", 0.25) or 0.25),
            shadow_enabled=shadow_enabled,
        )
        _merge(ctx, sh_status)
        try:
            if isinstance(ctx.get("shadow_last_status"), dict):
                ctx["shadow_last_status"]["shadow_step_reason"] = step_reason
                ctx["shadow_last_status"]["shadow_enabled"] = bool(shadow_enabled)
        except Exception:
            pass

    except Exception as e:
        try:
            logger.error("[shadow] step failed: %s", e)
        except Exception:
            pass
