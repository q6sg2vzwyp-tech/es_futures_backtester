#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
strategy_core.py
Strategy-related helpers for ES Paper Trader:

- BarBuffer (bars & indicators)
- ATR / ADX proxy calculations
- Signal selection via bandit arms
- Dynamic stop_dist / tp_dist based on ATR & ADX
- Regime-aware arm selection (trend vs chop)
- SHORT "parole" gating:
    * Only allow shorts in clear downtrend with sufficient ADX
    * Only during a safe time window (09:00–11:30 CT)

NEW (2025-12-21):
- Real-arm selection can be overridden at runtime (real_arms_override).
  This enables shadow→real promotion without editing code.
"""

from typing import List, Optional, Tuple, Dict
import datetime as dt


class BarBuffer:
    """
    Lightweight bar buffer built from last-trade prices.

    NOTE:
    - Close-only, so ATR/ADX are proxies.
    """

    def __init__(self, maxlen: int = 512):
        self.maxlen = maxlen
        self.ts: List[dt.datetime] = []
        self.close: List[float] = []
        self.total_bars: int = 0

    def add(self, ts: dt.datetime, close: float) -> None:
        self.ts.append(ts)
        self.close.append(close)
        self.total_bars += 1
        if len(self.ts) > self.maxlen:
            self.ts = self.ts[-self.maxlen:]
            self.close = self.close[-self.maxlen:]

    def ready(self, lookback: int) -> bool:
        return len(self.close) >= lookback

    def count(self) -> int:
        return len(self.close)

    def ema(self, length: int) -> Optional[float]:
        if not self.ready(length):
            return None
        alpha = 2.0 / (length + 1.0)
        v = self.close[-length]
        for x in self.close[-length + 1:]:
            v = alpha * x + (1.0 - alpha) * v
        return v

    def sma(self, length: int) -> Optional[float]:
        if not self.ready(length):
            return None
        return sum(self.close[-length:]) / length

    def atr_proxy(self, length: int = 14) -> Optional[float]:
        if len(self.close) <= length:
            return None
        diffs = [abs(self.close[i] - self.close[i - 1]) for i in range(1, len(self.close))]
        if len(diffs) < length:
            return None
        return sum(diffs[-length:]) / length

    def adx_proxy(self, length: int = 14) -> Optional[float]:
        if len(self.close) <= length + 1:
            return None

        closes = self.close
        trs: List[float] = []
        plus_dm: List[float] = []
        minus_dm: List[float] = []

        for i in range(1, len(closes)):
            up_move = closes[i] - closes[i - 1]
            down_move = closes[i - 1] - closes[i]

            plus = max(up_move, 0.0) if up_move > down_move else 0.0
            minus = max(down_move, 0.0) if down_move > up_move else 0.0

            tr = abs(closes[i] - closes[i - 1])
            trs.append(tr)
            plus_dm.append(plus)
            minus_dm.append(minus)

        if len(trs) < length:
            return None

        tr_n = sum(trs[-length:])
        plus_n = sum(plus_dm[-length:])
        minus_n = sum(minus_dm[-length:])

        if tr_n <= 0:
            return None

        plus_di = 100.0 * (plus_n / tr_n)
        minus_di = 100.0 * (minus_n / tr_n)
        denom = max(plus_di + minus_di, 1e-9)
        dx = abs(plus_di - minus_di) / denom * 100.0

        return dx


# ---------------------------------------------------------------------------
# Arm sets and regime groups
# ---------------------------------------------------------------------------

DEFAULT_ARMS = [
    "trend_ema",
    "trend_sma",
    "breakout_atr",
    "pullback_vwap",
    "momentum_rsi",
    "range_fade",
    "trend_pullback",

    "trend_ema2",
    "breakout_adx",
    "range_fade_strict",
    "mean_revert_ema",
    "momentum_pullback",
    "ma50_intraday",
]

REAL_ARMS = [
    "trend_ema",
    "pullback_vwap",
    "trend_pullback",
    "range_fade",
    "ma50_intraday",
]

SHADOW_ARMS = DEFAULT_ARMS.copy()

TREND_ARMS = {
    "trend_ema",
    "trend_sma",
    "breakout_atr",
    "momentum_rsi",
    "trend_pullback",
    "trend_ema2",
    "breakout_adx",
    "momentum_pullback",
    "ma50_intraday",
}

CHOP_ARMS = {
    "range_fade",
    "pullback_vwap",
    "range_fade_strict",
    "mean_revert_ema",
}


def _regime_from_adx(adx: Optional[float]) -> str:
    if adx is None:
        return "unknown"
    if adx < 18.0:
        return "chop"
    if adx > 25.0:
        return "trend"
    return "unknown"


def regime_from_adx_value(adx_val: float) -> str:
    if adx_val <= 0.0:
        return "unknown"
    return _regime_from_adx(adx_val)


def _pick_arm_with_bandit(
    bandit,
    regime_hint: str = "unknown",
    is_shadow: bool = False,
    real_arms_override: Optional[List[str]] = None,
) -> str:
    """
    Arm selection:

    - Shadow uses SHADOW_ARMS.
    - Real uses REAL_ARMS *unless* real_arms_override is provided.
    """

    if is_shadow:
        allowed = SHADOW_ARMS.copy()
    else:
        if real_arms_override and isinstance(real_arms_override, list) and len(real_arms_override) > 0:
            allowed = [str(a).strip() for a in real_arms_override if str(a).strip()]
        else:
            allowed = REAL_ARMS.copy()

    if not allowed:
        allowed = DEFAULT_ARMS.copy()

    # Regime-aware filtering
    if regime_hint == "chop":
        chop_candidates = [a for a in allowed if a in CHOP_ARMS]
        if chop_candidates:
            allowed = chop_candidates
    elif regime_hint == "trend":
        trend_candidates = [a for a in allowed if a in TREND_ARMS]
        if trend_candidates:
            allowed = trend_candidates

    if not allowed:
        allowed = DEFAULT_ARMS.copy()

    chosen: Optional[str] = None

    # Try bandit methods, respecting allowed arms
    if bandit is not None:
        if hasattr(bandit, "pick_arm_from"):
            try:
                arm = bandit.pick_arm_from(allowed)
                if arm in allowed:
                    chosen = arm
            except Exception:
                chosen = None

        if chosen is None and hasattr(bandit, "pick_arm"):
            try:
                arm = bandit.pick_arm()
                if arm in allowed:
                    chosen = arm
            except Exception:
                chosen = None

        if chosen is None and hasattr(bandit, "sample_arm"):
            try:
                arm = bandit.sample_arm()
                if arm in allowed:
                    chosen = arm
            except Exception:
                chosen = None

    if chosen is None:
        return allowed[0]

    # Light regime nudge
    if regime_hint == "chop" and chosen in TREND_ARMS:
        for a in allowed:
            if a in CHOP_ARMS:
                return a
        return chosen

    if regime_hint == "trend" and chosen in CHOP_ARMS:
        for a in allowed:
            if a in TREND_ARMS:
                return a
        return chosen

    return chosen


def _snap_dist_to_tick(dist: float, tick_size: float) -> float:
    if tick_size <= 0.0:
        return dist
    ticks = dist / tick_size
    ticks_rounded = max(1, int(round(ticks)))
    return ticks_rounded * tick_size


# ---------------------------------------------------------------------------
# SHORT "parole" gating helpers
# ---------------------------------------------------------------------------

def _ok_to_short_env(
    last_px: float,
    ema_fast: Optional[float],
    ema_slow: Optional[float],
    adx_val: float,
) -> bool:
    if ema_fast is None or ema_slow is None:
        return False
    if not (last_px < ema_slow):
        return False
    if not (ema_fast < ema_slow):
        return False
    if adx_val < 25.0:
        return False
    return True


def _ok_to_short_time(now_ct: Optional[dt.datetime] = None) -> bool:
    if now_ct is None:
        now_ct = dt.datetime.now()
    t = now_ct.time()
    if not (dt.time(9, 0) <= t < dt.time(11, 30)):
        return False
    return True


def _gate_short_side(
    side: Optional[str],
    last_px: float,
    ema_fast: Optional[float],
    ema_slow: Optional[float],
    adx_val: float,
) -> Optional[str]:
    if side != "SELL":
        return side
    now_ct = dt.datetime.now()
    if not _ok_to_short_env(last_px, ema_fast, ema_slow, adx_val):
        return None
    if not _ok_to_short_time(now_ct):
        return None
    return side


# ---------------------------------------------------------------------------
# Main signal builder
# ---------------------------------------------------------------------------

def build_signal_and_bands(
    bars: BarBuffer,
    last_px: float,
    bandit,
    risk_ticks: int,
    tick_size: float,
    base_tp_R: float,
    bars_15m: Optional[BarBuffer] = None,
    is_shadow: bool = False,
    forced_arm: Optional[str] = None,
    real_arms_override: Optional[List[str]] = None,
) -> Tuple[
    Optional[str],
    Optional[str],
    float,
    float,
    float,
    float,
    float,
]:
    if not bars.ready(20):
        return None, None, 0.0, 0.0, base_tp_R, 0.0, 0.0

    ema_fast = bars.ema(9)
    ema_slow = bars.ema(21)
    sma_fast = bars.sma(20)
    sma_slow = bars.sma(50)
    atr = bars.atr_proxy(14)
    adx = bars.adx_proxy(14)

    atr_points = float(atr or 0.0)
    adx_val = float(adx or 0.0)

    regime = _regime_from_adx(adx)

    if forced_arm is not None:
        arm = forced_arm
    else:
        arm = _pick_arm_with_bandit(
            bandit,
            regime_hint=regime,
            is_shadow=is_shadow,
            real_arms_override=real_arms_override,
        )

    side: Optional[str] = None

    if arm == "trend_ema" and ema_fast and ema_slow:
        if ema_fast > ema_slow:
            side = "BUY"
        elif ema_fast < ema_slow:
            side = "SELL"

    elif arm == "trend_ema2" and ema_fast and ema_slow:
        if ema_fast > ema_slow * 1.001 and last_px > ema_fast:
            side = "BUY"
        elif ema_fast < ema_slow * 0.999 and last_px < ema_fast:
            side = "SELL"

    elif arm == "trend_sma" and sma_fast and sma_slow:
        if sma_fast > sma_slow:
            side = "BUY"
        elif sma_fast < sma_slow:
            side = "SELL"

    elif arm == "breakout_atr" and atr is not None:
        mean_price = bars.sma(20)
        if mean_price and last_px > mean_price + 2 * atr:
            side = "BUY"
        elif mean_price and last_px < mean_price - 2 * atr:
            side = "SELL"

    elif arm == "breakout_adx" and atr is not None and adx is not None:
        mean_price = bars.sma(20)
        if mean_price and adx >= 28.0:
            if last_px > mean_price + 1.8 * atr:
                side = "BUY"
            elif last_px < mean_price - 1.8 * atr:
                side = "SELL"

    elif arm == "pullback_vwap":
        vwap_proxy = sma_slow
        if vwap_proxy:
            if last_px < vwap_proxy * 0.997:
                side = "BUY"
            elif last_px > vwap_proxy * 1.003:
                side = "SELL"

    elif arm == "momentum_rsi":
        if ema_fast:
            if last_px > ema_fast * 1.002:
                side = "BUY"
            elif last_px < ema_fast * 0.998:
                side = "SELL"

    elif arm == "momentum_pullback" and ema_fast and ema_slow:
        if ema_fast > ema_slow:
            if last_px < ema_fast * 0.999:
                side = "BUY"
        elif ema_fast < ema_slow:
            if last_px > ema_fast * 1.001:
                side = "SELL"

    elif arm == "range_fade":
        mean_price = sma_fast
        if atr is not None and mean_price is not None and adx is not None:
            if adx < 20.0:
                upper = mean_price + 1.5 * atr
                lower = mean_price - 1.5 * atr
                if last_px > upper:
                    side = "SELL"
                elif last_px < lower:
                    side = "BUY"

    elif arm == "range_fade_strict":
        mean_price = sma_fast
        if atr is not None and mean_price is not None and adx is not None:
            if adx < 15.0:
                upper = mean_price + 2.0 * atr
                lower = mean_price - 2.0 * atr
                if last_px > upper:
                    side = "SELL"
                elif last_px < lower:
                    side = "BUY"

    elif arm == "trend_pullback" and ema_fast and ema_slow:
        if ema_fast > ema_slow:
            if last_px < ema_fast * 0.998:
                side = "BUY"
        elif ema_fast < ema_slow:
            if last_px > ema_fast * 1.002:
                side = "SELL"

    elif arm == "mean_revert_ema" and ema_slow:
        if atr is not None:
            upper = ema_slow + 1.5 * atr
            lower = ema_slow - 1.5 * atr
            if last_px > upper:
                side = "SELL"
            elif last_px < lower:
                side = "BUY"

    elif arm == "ma50_intraday":
        # Hybrid execution arm (true 15m if bars_15m is provided):
        # - Intraday reference: EMA50
        # - Entry style: pullback toward EMA50 + rejection away (close-only proxy)
        src = bars_15m if (bars_15m is not None and getattr(bars_15m, 'ready', None) and bars_15m.ready(52)) else bars
        ema_50 = src.ema(50)
        atr_src = src.atr_proxy(14)
        if ema_50 is not None and atr_src is not None and len(src.close) >= 2:
            prev_close = float(src.close[-2])
            last_close = float(src.close[-1])

            # Anti-chop buffers in points (ATR proxy is close-to-close)
            buf = 0.25 * float(atr_src)   # must be meaningfully above/below EMA50
            pull = 0.60 * float(atr_src)  # prior close must have pulled back near EMA50

            # Long: above EMA50, then pull back near it, then reject back up
            if last_close > ema_50 + buf and prev_close <= ema_50 + pull:
                side = "BUY"

            # Short: below EMA50, then pull back near it, then reject back down
            elif last_close < ema_50 - buf and prev_close >= ema_50 - pull:
                side = "SELL"

            # If we are trading this arm, prefer 15m ATR/ADX for sizing/TP dynamics
            if side in ("BUY", "SELL"):
                atr = atr_src
                try:
                    adx_src = src.adx_proxy(14)
                except Exception:
                    adx_src = None
                if adx_src is not None:
                    adx = adx_src

        # Refresh telemetry for this arm if we swapped ATR/ADX
        atr_points = float(atr or 0.0)
        adx_val = float(adx or 0.0)

    side = _gate_short_side(
        side=side,
        last_px=last_px,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        adx_val=adx_val,
    )

    if side is None:
        return arm, None, 0.0, 0.0, base_tp_R, atr_points, adx_val

    base_stop_ticks = risk_ticks
    if atr is not None and tick_size > 0.0:
        atr_ticks = atr / tick_size
        lo = base_stop_ticks * 0.75
        hi = base_stop_ticks * 1.75
        dyn_ticks = max(lo, min(hi, atr_ticks))
        dyn_ticks_rounded = max(1, int(round(dyn_ticks)))
        stop_dist = dyn_ticks_rounded * tick_size
    else:
        stop_dist = base_stop_ticks * tick_size

    dyn_tp_R = base_tp_R
    if adx is not None:
        if adx < 15.0:
            dyn_tp_R = max(0.7, base_tp_R * 0.8)
        elif adx < 25.0:
            dyn_tp_R = base_tp_R
        elif adx < 35.0:
            dyn_tp_R = base_tp_R * 1.3
        else:
            dyn_tp_R = base_tp_R * 1.7

    tp_dist = dyn_tp_R * stop_dist

    stop_dist = _snap_dist_to_tick(stop_dist, tick_size)
    tp_dist = _snap_dist_to_tick(tp_dist, tick_size)

    return arm, side, stop_dist, tp_dist, dyn_tp_R, atr_points, adx_val


def build_all_signals(
    bars: BarBuffer,
    last_px: float,
    risk_ticks: int,
    tick_size: float,
    base_tp_R: float,
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    if not bars.ready(20):
        return out

    for arm_name in SHADOW_ARMS:
        (
            _chosen_arm,
            side,
            stop_dist,
            tp_dist,
            dyn_tp_R,
            atr_points,
            adx_val,
        ) = build_signal_and_bands(
            bars=bars,
            bars_15m=bars_15m,
            last_px=last_px,
            bandit=None,
            risk_ticks=risk_ticks,
            tick_size=tick_size,
            base_tp_R=base_tp_R,
            is_shadow=True,
            forced_arm=arm_name,
        )

        out[arm_name] = {
            "side": side,
            "stop_dist": float(stop_dist),
            "tp_dist": float(tp_dist),
            "dyn_tp_R": float(dyn_tp_R),
            "atr_points": float(atr_points),
            "adx_val": float(adx_val),
        }

    return out
