#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shadow_facade.py

High-level wrapper around shadow_core.

Goals:
- No dependency on shadow_runner / shadow_engine modules (prevents import drift).
- Stable API for loop_core:
    * controller.step(...) -> Dict[str, Any] (heartbeat-ish fields)
    * controller.entry_multiplier(...) -> (multiplier, veto_or_None)
    * controller.maybe_update_model_eod(...) safe to call repeatedly
- Compatible with two possible shadow_core load contracts:
    A) load_shadow_model(...) -> (ShadowSim|None, ShadowIndex)
    B) load_shadow_model(...) -> legacy_model_object
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Tuple
import datetime as dt

from shadow_core import load_shadow_model, update_shadow_model

AppendRoundtripFn = Callable[[Dict[str, Any]], None]


def _normalize_loaded_model(x: Any) -> Tuple[Optional[Any], Optional[Any], Any]:
    """
    Returns (shadow_sim, shadow_index, raw_model)

    Supports:
      - (sim, index) tuple
      - sim only
      - legacy dict/obj
    """
    if isinstance(x, tuple) and len(x) == 2:
        return x[0], x[1], x
    # If caller returned a ShadowSim-like object, treat it as sim.
    if hasattr(x, "step") and hasattr(x, "heartbeat_fields"):
        return x, None, x
    # Legacy model object
    return None, None, x


@dataclass
class ShadowController:
    # Paths
    model_json: str
    roundtrip_csv: str

    # Core engine (preferred)
    shadow: Any  # ShadowSim-like (has step(), heartbeat_fields())

    # Optional index/meta object (ShadowIndex-like)
    index: Optional[Any] = None

    # For legacy scoring-only models
    model: Optional[Any] = None

    # EOD guard
    last_model_update_day: Optional[dt.date] = None

    def reset_day(self) -> None:
        """
        Resets intraday counters/position state in shadow engine.
        """
        if self.shadow is not None and hasattr(self.shadow, "reset_day"):
            self.shadow.reset_day()

    def step(
        self,
        *,
        now_ct: dt.datetime,
        last_px: float,
        in_shadow_window: bool,
        arm: Optional[str],
        side: Optional[str],
        per_contract_init: float,
        last_regime: str,
        week_R: float,
        meta_ema_R: float,
        append_shadow_roundtrip_log: Optional[AppendRoundtripFn] = None,
        atr_points: float = 0.0,
        tick_size: float = 0.25,
        decision_bucket_sec: int = 30,
        min_hold_sec: int = 120,
        atr_floor_ticks: float = 2.0,
        max_hold_sec: Optional[int] = None,
        shadow_enabled: bool = True,
        # Optional overtrading rails (pass through if your ShadowSim supports them)
        max_roundtrips_per_day: int = 0,
        max_roundtrips_per_hour: int = 0,
        post_close_cooldown_sec: int = 0,
        post_loss_cooldown_sec: int = 0,
    ) -> Dict[str, Any]:
        """
        Delegates to shadow_core.ShadowSim.step() (or compatible engine).

        This replaces the prior shadow_runner_step dependency.
        """
        if self.shadow is None or not hasattr(self.shadow, "step"):
            # Fail safe: return minimal heartbeat fields
            return {
                "shadow_last_action": "NO_SHADOW_ENGINE",
                "shadow_step_reason": "no_engine",
                "shadow_pnl_today": 0.0,
                "shadow_R_today": 0.0,
                "shadow_trades_today": 0,
            }

        # Call shadow.step with a superset of args; ShadowSim is expected to accept these.
        return self.shadow.step(
            now_ct=now_ct,
            last_px=float(last_px),
            in_shadow_window=bool(in_shadow_window),
            arm=arm,
            side=side,
            per_contract_init=float(per_contract_init),
            last_regime=str(last_regime or "unknown"),
            week_R=float(week_R),
            meta_ema_R=float(meta_ema_R),
            append_shadow_roundtrip_log=append_shadow_roundtrip_log,
            atr_points=float(atr_points or 0.0),
            tick_size=float(tick_size or 0.25),
            decision_bucket_sec=int(decision_bucket_sec),
            min_hold_sec=int(min_hold_sec),
            atr_floor_ticks=float(atr_floor_ticks),
            max_hold_sec=max_hold_sec,
            shadow_enabled=bool(shadow_enabled),
            max_roundtrips_per_day=int(max_roundtrips_per_day or 0),
            max_roundtrips_per_hour=int(max_roundtrips_per_hour or 0),
            post_close_cooldown_sec=int(post_close_cooldown_sec or 0),
            post_loss_cooldown_sec=int(post_loss_cooldown_sec or 0),
        )

    def heartbeat_fields(self) -> Dict[str, Any]:
        """
        Prefer engine heartbeat_fields() when available.
        """
        if self.shadow is not None and hasattr(self.shadow, "heartbeat_fields"):
            try:
                return dict(self.shadow.heartbeat_fields())
            except Exception:
                pass

        return {
            "shadow_pnl_today": 0.0,
            "shadow_R_today": 0.0,
            "shadow_trades_today": 0,
            "shadow_last_action": "NOOP",
        }

    def entry_multiplier(
        self,
        *,
        regime: str,
        arm: str,
        side: str,
        default: float = 1.0,
        min_trades: int = 10,
    ) -> Tuple[float, Optional[str]]:
        """
        Returns (multiplier, veto_string_or_None)

        Primary behavior:
          - If engine provides entry_multiplier(), use it.
          - Else, fall back to legacy bucket model if present.
        """
        # Preferred: engine-native scoring
        if self.shadow is not None and hasattr(self.shadow, "entry_multiplier"):
            try:
                return self.shadow.entry_multiplier(
                    regime=regime, arm=arm, side=side, default=default, min_trades=min_trades
                )
            except TypeError:
                # older signature without min_trades
                try:
                    return self.shadow.entry_multiplier(
                        regime=regime, arm=arm, side=side, default=default
                    )
                except Exception:
                    pass
            except Exception:
                pass

        # Legacy fallback: dict-like bucket model with .get(key)->bucket(mean_R)
        m = self.model
        if not m or not hasattr(m, "get"):
            return float(default), None

        key = (regime or "unknown", arm or "unknown", (side or "").upper())
        bucket = m.get(key)
        if bucket is None:
            return float(default), None

        score = float(getattr(bucket, "mean_R", 0.0) or 0.0)
        # Keep the same thresholds you had (block/weak/good).
        if score <= -0.20:
            veto = f"shadow_block arm={arm} side={side.upper()} reg={regime} score={score:.3f}"
            return 0.0, veto
        if score <= -0.10:
            veto = f"shadow_weak arm={arm} side={side.upper()} reg={regime} score={score:.3f}"
            return 0.50, veto
        if score >= 0.05:
            return 1.25, None
        return 1.0, None

    def maybe_update_model_eod(
        self,
        *,
        now_ct: Optional[dt.datetime] = None,
        logger=None,
    ) -> None:
        """
        Safe to call repeatedly; will only refresh once per CT date.

        Behavior:
        - If engine supports maybe_update_model_eod(): call it.
        - Persist state via update_shadow_model(...) when possible.
        - Reload from disk to keep controller fields aligned with on-disk JSON.
        """
        now_ct = now_ct or dt.datetime.now()
        day = now_ct.date()
        if self.last_model_update_day == day:
            return

        # 1) Let engine update its own model/index if it supports it.
        if self.shadow is not None and hasattr(self.shadow, "maybe_update_model_eod"):
            try:
                # Support both signatures (some versions accept now_ct)
                try:
                    self.shadow.maybe_update_model_eod(now_ct=now_ct, logger=logger)
                except TypeError:
                    self.shadow.maybe_update_model_eod(logger=logger)
            except Exception as e:
                if logger:
                    logger.warning(f"[shadow] maybe_update_model_eod failed: {e}")

        # 2) Persist to JSON using shadow_core.update_shadow_model if it supports shadow_sim.
        try:
            update_shadow_model(
                model_json_path=self.model_json,
                shadow_sim=self.shadow if self.shadow is not None else None,
                shadow_index=self.index,
                logger=logger,
            )
        except TypeError:
            # Older update_shadow_model signature; try simplest.
            try:
                update_shadow_model(self.model_json, logger=logger)
            except Exception as e:
                if logger:
                    logger.warning(f"[shadow] update_shadow_model failed: {e}")
        except Exception as e:
            if logger:
                logger.warning(f"[shadow] update_shadow_model failed: {e}")

        # 3) Reload to align controller state with whatever was written.
        try:
            loaded = load_shadow_model(self.model_json, logger=logger)
            sim, idx, raw = _normalize_loaded_model(loaded)
            if sim is not None:
                self.shadow = sim
            if idx is not None:
                self.index = idx
            # raw legacy model object
            if sim is None:
                self.model = raw
        except Exception as e:
            if logger:
                logger.warning(f"[shadow] load_shadow_model failed after update: {e}")

        self.last_model_update_day = day


def new_shadow_controller(*, model_json: str, roundtrip_csv: str, logger=None) -> ShadowController:
    """
    Construct controller from on-disk model, with a safe fallback engine.

    If load_shadow_model returns (ShadowSim|None, ShadowIndex):
      - use returned sim if present
      - else create a fresh ShadowSim via shadow_core.ShadowSim()
    """
    loaded = load_shadow_model(model_json, logger=logger)
    sim, idx, raw = _normalize_loaded_model(loaded)

    shadow_engine = sim
    legacy_model = None

    if shadow_engine is None:
        # Try to construct ShadowSim directly if available in shadow_core
        try:
            from shadow_core import ShadowSim
            shadow_engine = ShadowSim(roundtrips_csv=roundtrip_csv, model_path=model_json)
        except Exception:
            shadow_engine = None
            legacy_model = raw
    else:
        # If engine exists, keep legacy model empty
        legacy_model = None

    ctl = ShadowController(
        model_json=str(model_json),
        roundtrip_csv=str(roundtrip_csv),
        shadow=shadow_engine,
        index=idx,
        model=legacy_model,
    )
    return ctl
