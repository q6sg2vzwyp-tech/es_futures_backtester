#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shadow_orchestrator.py

One-stop wrapper that wires:
- preferred: shadow_core.ShadowSim (self-contained step + logging)
- optional: shadow_engine + shadow_runner (if those modules exist in your repo)
- shadow_core model load/update (compatible with multiple signatures)
- optional real-entry veto/multiplier (via pt_utils.shadow_score_combo if available)

Design goals:
- Avoid import-time crashes if shadow_engine/shadow_runner/pt_utils are missing.
- Tolerate shadow_core contract drift (tuple vs single return; update signature changes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Tuple
import datetime as dt

from shadow_core import load_shadow_model, update_shadow_model

AppendRoundtripFn = Callable[[Dict[str, Any]], None]


# -----------------------------
# Optional dependencies (best-effort)
# -----------------------------
_HAS_RUNNER = False
ShadowEngineState = None
ShadowRunnerState = None
shadow_runner_step = None
shadow_runner_reset_day = None

try:
    from shadow_engine import ShadowEngineState as _ShadowEngineState
    from shadow_runner import ShadowRunnerState as _ShadowRunnerState
    from shadow_runner import shadow_runner_step as _shadow_runner_step
    from shadow_runner import shadow_runner_reset_day as _shadow_runner_reset_day

    ShadowEngineState = _ShadowEngineState
    ShadowRunnerState = _ShadowRunnerState
    shadow_runner_step = _shadow_runner_step
    shadow_runner_reset_day = _shadow_runner_reset_day
    _HAS_RUNNER = True
except Exception:
    _HAS_RUNNER = False

_shadow_score_combo = None
try:
    from pt_utils import shadow_score_combo as _ssc
    _shadow_score_combo = _ssc
except Exception:
    _shadow_score_combo = None


def _normalize_loaded_model(x: Any) -> Tuple[Optional[Any], Optional[Any], Any]:
    """
    Returns (shadow_sim, shadow_index, raw_model)

    Supports:
      - (sim, index) tuple
      - sim only (ShadowSim-like)
      - legacy model object
    """
    if isinstance(x, tuple) and len(x) == 2:
        return x[0], x[1], x
    if hasattr(x, "step") and hasattr(x, "heartbeat_fields"):
        return x, None, x
    return None, None, x


@dataclass
class ShadowOrchestrator:
    model_json: str
    roundtrip_csv: str

    # Either runner-based or engine-based
    runner: Optional[Any] = None         # ShadowRunnerState-like
    shadow: Optional[Any] = None         # ShadowSim-like

    # Model/index (shape depends on shadow_core)
    model: Optional[Any] = None          # ShadowIndex-like or legacy model
    index: Optional[Any] = None          # ShadowIndex-like (optional)

    bad_threshold: float = -0.20
    weak_threshold: float = -0.10
    good_threshold: float = 0.05

    min_trades: int = 20
    lookback_days: int = 5

    last_model_update_day: Optional[dt.date] = None

    @classmethod
    def new(
        cls,
        *,
        model_json: str,
        roundtrip_csv: str,
        logger=None,
        bad_threshold: float = -0.20,
        weak_threshold: float = -0.10,
        good_threshold: float = 0.05,
        min_trades: int = 20,
        lookback_days: int = 5,
        prefer_runner: bool = False,
    ) -> "ShadowOrchestrator":
        loaded = load_shadow_model(model_json, logger=logger)
        sim, idx, raw = _normalize_loaded_model(loaded)

        runner = None
        shadow = None
        model = None
        index = None

        # If we have a ShadowSim-like engine, prefer it for stability unless caller explicitly prefers runner.
        if sim is not None and not prefer_runner:
            shadow = sim
            index = idx
            model = None
        else:
            # If runner modules exist and caller prefers runner, use them.
            if _HAS_RUNNER and prefer_runner:
                try:
                    runner = ShadowRunnerState(engine=ShadowEngineState())
                except Exception:
                    runner = None
                # model/index could still be useful for scoring
                model = idx if idx is not None else raw
                index = idx
            else:
                # Fall back: create ShadowSim directly if available
                try:
                    from shadow_core import ShadowSim
                    shadow = ShadowSim(roundtrips_csv=roundtrip_csv, model_path=model_json)
                    index = idx
                    model = None
                except Exception:
                    shadow = None
                    model = raw
                    index = idx

        return cls(
            model_json=str(model_json),
            roundtrip_csv=str(roundtrip_csv),
            runner=runner,
            shadow=shadow,
            model=model,
            index=index,
            bad_threshold=bad_threshold,
            weak_threshold=weak_threshold,
            good_threshold=good_threshold,
            min_trades=int(min_trades),
            lookback_days=int(lookback_days),
        )

    def reset_day(self) -> None:
        # Runner path
        if self.runner is not None and shadow_runner_reset_day is not None:
            try:
                shadow_runner_reset_day(self.runner)
                return
            except Exception:
                pass

        # ShadowSim path
        if self.shadow is not None and hasattr(self.shadow, "reset_day"):
            try:
                self.shadow.reset_day()
            except Exception:
                pass

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
        qty: int = 1,
        point_value: float = 50.0,
        force_flat_outside_window: bool = True,

        # anti-overtrading / context
        decision_bucket_sec: int = 30,
        min_hold_sec: int = 120,
        min_roundtrip_gap_sec: float = 10.0,  # runner-only
        atr_points: float = 0.0,
        tick_size: float = 0.25,
        atr_floor_ticks: float = 2.0,
        max_hold_sec: Optional[int] = None,
        shadow_enabled: bool = True,

        # ShadowSim-only (optional rails)
        max_roundtrips_per_day: int = 0,
        max_roundtrips_per_hour: int = 0,
        post_close_cooldown_sec: int = 0,
        post_loss_cooldown_sec: int = 0,
    ) -> Dict[str, Any]:
        # Prefer runner if it exists AND is configured
        if self.runner is not None and shadow_runner_step is not None:
            return shadow_runner_step(
                self.runner,
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
                qty=int(qty),
                point_value=float(point_value),
                force_flat_outside_window=bool(force_flat_outside_window),

                decision_bucket_sec=int(decision_bucket_sec),
                min_hold_sec=int(min_hold_sec),
                min_roundtrip_gap_sec=float(min_roundtrip_gap_sec or 0.0),
                atr_points=float(atr_points or 0.0),
                tick_size=float(tick_size or 0.25),
                atr_floor_ticks=float(atr_floor_ticks),
            )

        # Otherwise, use ShadowSim-like engine
        if self.shadow is not None and hasattr(self.shadow, "step"):
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

        # Fail-safe minimal heartbeat
        return {
            "shadow_last_action": "NO_SHADOW",
            "shadow_step_reason": "no_engine",
            "shadow_pnl_today": 0.0,
            "shadow_R_today": 0.0,
            "shadow_trades_today": 0,
        }

    def heartbeat_fields(self) -> Dict[str, Any]:
        # Runner fields
        if self.runner is not None:
            try:
                return {
                    "shadow_pnl_today": float(getattr(self.runner, "pnl_today", 0.0)),
                    "shadow_R_today": float(getattr(self.runner, "R_today", 0.0)),
                    "shadow_trades_today": int(getattr(self.runner, "trades_today", 0)),
                }
            except Exception:
                pass

        # ShadowSim fields
        if self.shadow is not None and hasattr(self.shadow, "heartbeat_fields"):
            try:
                return dict(self.shadow.heartbeat_fields())
            except Exception:
                pass

        return {"shadow_pnl_today": 0.0, "shadow_R_today": 0.0, "shadow_trades_today": 0}

    def entry_multiplier(
        self,
        *,
        regime: str,
        arm: str,
        side: str,
        default: float = 1.0,
        min_trades: int = 10,
    ) -> Tuple[float, Optional[str]]:
        # Preferred: engine-native scoring
        if self.shadow is not None and hasattr(self.shadow, "entry_multiplier"):
            try:
                return self.shadow.entry_multiplier(
                    regime=regime, arm=arm, side=side, default=default, min_trades=min_trades
                )
            except TypeError:
                try:
                    return self.shadow.entry_multiplier(regime=regime, arm=arm, side=side, default=default)
                except Exception:
                    pass
            except Exception:
                pass

        # Optional: pt_utils scoring over a ShadowIndex-like model
        if _shadow_score_combo is not None and self.model is not None:
            try:
                score = float(
                    _shadow_score_combo(
                        shadow_model=self.model,
                        regime=regime,
                        arm=arm,
                        side=side,
                        default=0.0,
                    )
                )
                if score <= self.bad_threshold:
                    veto = f"shadow_block arm={arm} side={side.upper()} reg={regime} score={score:.3f}"
                    return 0.0, veto
                if score <= self.weak_threshold:
                    veto = f"shadow_weak arm={arm} side={side.upper()} reg={regime} score={score:.3f}"
                    return 0.50, veto
                if score >= self.good_threshold:
                    return 1.25, None
                return 1.0, None
            except Exception:
                pass

        # Default: no veto
        return float(default), None

    def maybe_update_model_eod(self, *, logger=None, now_ct: Optional[dt.datetime] = None) -> Any:
        """
        Update/persist model once per CT date.
        Works with multiple update_shadow_model() signatures.
        """
        now_ct = now_ct or dt.datetime.now()
        day = now_ct.date()
        if self.last_model_update_day == day:
            return self.model

        # 1) If the engine has its own EOD updater, call it first
        if self.shadow is not None and hasattr(self.shadow, "maybe_update_model_eod"):
            try:
                try:
                    self.shadow.maybe_update_model_eod(now_ct=now_ct, logger=logger)
                except TypeError:
                    self.shadow.maybe_update_model_eod(logger=logger)
            except Exception:
                pass

        # 2) Persist using whatever update_shadow_model signature exists
        updated = None
        try:
            # Newer signature (shadow_sim/index persistence)
            updated = update_shadow_model(
                model_json_path=self.model_json,
                shadow_sim=self.shadow if self.shadow is not None else None,
                shadow_index=self.index,
                logger=logger,
            )
        except TypeError:
            # Older signature (roundtrips -> json rebuild)
            try:
                updated = update_shadow_model(
                    roundtrip_csv=self.roundtrip_csv,
                    out_json=self.model_json,
                    min_trades=int(self.min_trades),
                    lookback_days=int(self.lookback_days),
                    logger=logger,
                )
            except Exception as e:
                if logger:
                    logger.warning(f"[shadow] update_shadow_model failed: {e}")
        except Exception as e:
            if logger:
                logger.warning(f"[shadow] update_shadow_model failed: {e}")

        # 3) Reload to align orchestrator state
        try:
            loaded = load_shadow_model(self.model_json, logger=logger)
            sim, idx, raw = _normalize_loaded_model(loaded)
            if sim is not None:
                self.shadow = sim
            if idx is not None:
                self.index = idx
            # keep a model handle for pt_utils scoring if applicable
            if sim is None:
                self.model = raw
            elif updated is not None:
                self.model = updated
        except Exception as e:
            if logger:
                logger.warning(f"[shadow] reload after update failed: {e}")

        self.last_model_update_day = day
        return self.model
