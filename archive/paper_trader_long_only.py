r"""
paper_trader_long_only.py — ES long-only paper trader (LIVE on IB paper)
- Long-only: BUY to open; SELL only to reduce/exit (no shorts, ever)
- Entry: stop-entry above signal bar (last high + k*ATR)
- Protection: stop-loss (STOP) based on ATR from entry
- Scaling IN: add buy units every step_atr * ATR progress (up to N adds)
- Scaling OUT: sequential take-profit ladder by R-multiples (1R, 2R, ...)
    * After first TP fill → move SL to breakeven
    * After each TP fill → place next TP (until ladder done)
- Time stop: exit if progress < X·R after N bars
- RTH/spread/ATR gates; warm-up 1m history; real-time 5s→1m aggregation
- LIVE market data by default (don’t pass --use-delayed)

Prereqs:
  pip install ib-insync pandas numpy
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, Contract, Future, Order, Trade, util
except Exception:
    IB = None
    util = None
    Contract = Future = Order = Trade = object  # type: ignore

# ----- Constants -----
RESULTS_DIR = "results"
TICK_SIZE = 0.25
TICK_VALUE = 12.5
CHICAGO_TZ = "America/Chicago"


# ===================== Indicators =====================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    gain = up.ewm(alpha=1 / length, min_periods=length).mean()
    loss = dn.ewm(alpha=1 / length, min_periods=length).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    c1 = df["Close"].shift()
    tr1 = (df["High"] - df["Low"]).abs()
    tr2 = (df["High"] - c1).abs()
    tr3 = (df["Low"] - c1).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return _true_range(df).rolling(length, min_periods=1).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plusDM = np.where((up > dn) & (up > 0), up, 0.0)
    minusDM = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _true_range(df)
    atrN = tr.ewm(alpha=1 / length, adjust=False).mean()
    plusDI = (
        100
        * pd.Series(plusDM, index=df.index).ewm(alpha=1 / length, adjust=False).mean()
        / (atrN + 1e-12)
    )
    minusDI = (
        100
        * pd.Series(minusDM, index=df.index).ewm(alpha=1 / length, adjust=False).mean()
        / (atrN + 1e-12)
    )
    dx = ((plusDI - minusDI).abs() / (plusDI + minusDI + 1e-12)) * 100
    return dx.ewm(alpha=1 / length, adjust=False).mean()


# ===================== Strategy: Long-only EMA + optional RSI =====================
class StrategyLongOnly:
    def __init__(self, fast=8, slow=20, rsi_len=14, use_rsi=True, rsi_buy=55.0):
        self.fast = int(fast)
        self.slow = int(slow)
        self.rsi_len = int(rsi_len)
        self.use_rsi = bool(use_rsi)
        self.rsi_buy = float(rsi_buy)

    def indicators(self, df: pd.DataFrame) -> dict[str, float]:
        c = df["Close"]
        return {
            "ema_f": float(ema(c, self.fast).iloc[-1]) if len(c) else np.nan,
            "ema_s": float(ema(c, self.slow).iloc[-1]) if len(c) else np.nan,
            "rsi": float(rsi(c, self.rsi_len).iloc[-1]) if len(c) else 50.0,
        }

    def buy_signal(self, df: pd.DataFrame) -> bool:
        if len(df) < max(self.slow, self.rsi_len) + 2:
            return False
        c = df["Close"]
        ef, es = ema(c, self.fast), ema(c, self.slow)
        cross_up = ef.iloc[-2] <= es.iloc[-2] and ef.iloc[-1] > es.iloc[-1]
        if not cross_up:
            return False
        if self.use_rsi:
            r = rsi(c, self.rsi_len).iloc[-1]
            return r >= self.rsi_buy
        return True


# ===================== Logging =====================
@dataclass
class TradeRec:
    EntryTime: pd.Timestamp
    ExitTime: pd.Timestamp
    Side: str
    EntryPrice: float
    ExitPrice: float
    Qty: int


class Bookkeeper:
    def __init__(self, session_dir: str, strategy_tag: str = "LIVE_LONG_ONLY"):
        self.dir = session_dir
        self.tag = strategy_tag
        os.makedirs(self.dir, exist_ok=True)
        self.paths = {
            "bars": os.path.join(self.dir, "bars_1m.csv"),
            "signals": os.path.join(self.dir, "signals.csv"),
            "orders": os.path.join(self.dir, "orders.csv"),
            "meta": os.path.join(self.dir, "session.json"),
        }

    @staticmethod
    def _append_row(path: str, header: list[str], row: dict[str, object]):
        first = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if first:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in header})

    def log_bar(self, ts, o, h, l, c):
        self._append_row(
            self.paths["bars"],
            ["Time", "Open", "High", "Low", "Close"],
            {"Time": str(pd.Timestamp(ts)), "Open": o, "High": h, "Low": l, "Close": c},
        )

    def log_signal(self, ts, inds, signal_txt: str):
        self._append_row(
            self.paths["signals"],
            ["Time", "EMA_Fast", "EMA_Slow", "RSI", "Note", "StrategyTag"],
            {
                "Time": str(pd.Timestamp(ts)),
                "EMA_Fast": inds.get("ema_f", np.nan),
                "EMA_Slow": inds.get("ema_s", np.nan),
                "RSI": inds.get("rsi", np.nan),
                "Note": signal_txt,
                "StrategyTag": self.tag,
            },
        )

    def log_order(self, ts, action, order_type, qty, ref_px=None, extra=""):
        self._append_row(
            self.paths["orders"],
            ["Time", "Action", "OrderType", "Qty", "RefPx", "Extra", "StrategyTag"],
            {
                "Time": str(pd.Timestamp(ts)),
                "Action": action,
                "OrderType": order_type,
                "Qty": qty,
                "RefPx": "" if ref_px is None else ref_px,
                "Extra": extra,
                "StrategyTag": self.tag,
            },
        )

    def write_meta(self, meta: dict[str, object]):
        with open(self.paths["meta"], "w") as f:
            json.dump(meta, f, indent=2, default=str)


# ===================== Helpers =====================
def _now_ct():
    # Chicago time (DST-safe if system tz data available)
    try:
        from zoneinfo import ZoneInfo

        return dt.datetime.now(dt.UTC).astimezone(ZoneInfo(CHICAGO_TZ))
    except Exception:
        return dt.datetime.now(dt.timezone(dt.timedelta(hours=-5)))


def is_rth(ts_utc: pd.Timestamp) -> bool:
    loc = (
        ts_utc.tz_convert(CHICAGO_TZ)
        if ts_utc.tzinfo is not None
        else ts_utc.tz_localize("UTC").tz_convert(CHICAGO_TZ)
    )
    start = loc.replace(hour=8, minute=30, second=0, microsecond=0)
    end = loc.replace(hour=15, minute=0, second=0, microsecond=0)
    return start <= loc <= end


def qualify_front_two_es(ib: IB) -> list[Contract]:
    cds = ib.reqContractDetails(Future(symbol="ES", exchange="CME", currency="USD"))
    futs = [cd.contract for cd in cds if getattr(cd.contract, "lastTradeDateOrContractMonth", None)]
    today = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    futs = [f for f in futs if f.lastTradeDateOrContractMonth >= today]
    futs.sort(key=lambda c: c.lastTradeDateOrContractMonth)
    return futs[:2] if len(futs) >= 2 else futs


def pick_active_contract(ib: IB) -> Contract:
    futs = qualify_front_two_es(ib)
    if not futs:
        raise RuntimeError("No ES futures returned from IB.")
    return ib.qualifyContracts(futs[0])[0]


def spread_ticks_now(ib: IB, contract: Contract) -> float | None:
    try:
        t = ib.reqMktData(contract, "", False, False)
        ib.sleep(0.5)
        b, a = float(getattr(t, "bid", np.nan)), float(getattr(t, "ask", np.nan))
        if np.isnan(b) or np.isnan(a) or a <= 0 or b <= 0:
            return None
        return max(0.0, (a - b) / TICK_SIZE)
    except Exception:
        return None


# ===================== Long-only Live OMS =====================
@dataclass
class PositionState:
    in_pos: bool = False
    entry_px: float | None = None
    total_qty: int = 0
    bars_in_pos: int = 0
    r_val: float | None = None
    last_add_trigger_px: float | None = None
    tp_index: int = 0
    parent: Trade | None = None
    sl: Trade | None = None
    tp_working: Trade | None = None


class LongOnlyOMS:
    def __init__(self, ib: IB, contract: Contract, book: Bookkeeper | None, place_orders: bool):
        self.ib = ib
        self.contract = contract
        self.book = book
        self.place_orders = bool(place_orders)
        self.state = PositionState()

    # --- order helpers ---
    def _place(self, order: Order) -> Trade | None:
        if not self.place_orders:
            if self.book:
                self.book.log_order(
                    pd.Timestamp.utcnow(),
                    order.action,
                    order.orderType,
                    int(order.totalQuantity),
                    ref_px=getattr(order, "lmtPrice", None) or getattr(order, "auxPrice", None),
                    extra="(not placed)",
                )
            return None
        tr = self.ib.placeOrder(self.contract, order)
        return tr

    def place_parent_buy_stop(self, qty: int, trigger_px: float) -> Trade | None:
        o = Order(
            action="BUY",
            orderType="STP",
            totalQuantity=qty,
            auxPrice=trigger_px,
            tif="DAY",
            transmit=True,
        )
        if self.book:
            self.book.log_order(pd.Timestamp.utcnow(), "BUY", "STP(entry)", qty, ref_px=trigger_px)
        tr = self._place(o)
        self.state.parent = tr
        return tr

    def _cancel_trade(self, tr: Trade | None):
        if not tr or not self.place_orders:
            return
        try:
            if tr.orderStatus.status not in ("Cancelled", "Filled", "Inactive"):
                self.ib.cancelOrder(tr.order)
        except Exception as e:

            logger.debug("Swallowed exception in archive\paper_trader_long_only.py: %s", e)

    def replace_stop_loss(self, new_px: float, new_qty: int):
        # cancel old SL, place new STOP (SELL) for full qty
        self._cancel_trade(self.state.sl)
        o = Order(
            action="SELL",
            orderType="STP",
            totalQuantity=new_qty,
            auxPrice=new_px,
            tif="DAY",
            transmit=True,
        )
        if self.book:
            self.book.log_order(
                pd.Timestamp.utcnow(), "SELL", "SL(STP)", new_qty, ref_px=new_px, extra="replace"
            )
        self.state.sl = self._place(o)

    def place_take_profit(self, limit_px: float, qty: int):
        # cancel any working TP and place new one (sequential ladder)
        self._cancel_trade(self.state.tp_working)
        o = Order(
            action="SELL",
            orderType="LMT",
            totalQuantity=qty,
            lmtPrice=limit_px,
            tif="DAY",
            transmit=True,
        )
        if self.book:
            self.book.log_order(
                pd.Timestamp.utcnow(), "SELL", "TP(LMT)", qty, ref_px=limit_px, extra="ladder step"
            )
        self.state.tp_working = self._place(o)

    def cancel_all(self):
        self._cancel_trade(self.state.tp_working)
        self._cancel_trade(self.state.sl)
        self._cancel_trade(self.state.parent)
        self.state = PositionState()


# ===================== Controller (signals → orders) =====================
@dataclass
class Params:
    stop_entry_atr_mult: float = 0.5
    atr_sl_mult: float = 1.8
    scale_in_max: int = 2
    scale_in_step_atr: float = 0.5
    add_qty: int = 1
    tp_r_list: list[float] = None  # e.g., [1.0, 2.0]
    tp_frac_list: list[float] = None  # e.g., [0.5, 0.5]
    time_stop_bars: int = 30
    min_progress_r: float = 0.5


class LongOnlyController:
    def __init__(
        self,
        p: Params,
        ib: IB,
        contract: Contract,
        book: Bookkeeper | None,
        place_orders: bool,
        default_qty: int,
    ):
        self.p = p
        self.ib = ib
        self.contract = contract
        self.book = book
        self.default_qty = int(default_qty)
        self.oms = LongOnlyOMS(ib, contract, book, place_orders)

    def _ensure_lists(self):
        if not self.p.tp_r_list:
            self.p.tp_r_list = [1.0]
        if not self.p.tp_frac_list:
            self.p.tp_frac_list = [1.0]
        # normalize lens
        n = min(len(self.p.tp_r_list), len(self.p.tp_frac_list))
        self.p.tp_r_list = self.p.tp_r_list[:n]
        self.p.tp_frac_list = self.p.tp_frac_list[:n]

    def maybe_enter(self, ts: pd.Timestamp, df: pd.DataFrame, qty: int):
        if self.oms.state.in_pos or self.oms.state.parent:
            return
        atr14 = float(atr(df, 14).iloc[-1])
        last_high = float(df["High"].iloc[-1])
        trigger = last_high + self.p.stop_entry_atr_mult * atr14
        self.oms.place_parent_buy_stop(qty, trigger)
        if self.book:
            self.book.log_signal(
                ts, {"ema_f": np.nan, "ema_s": np.nan, "rsi": np.nan}, f"BUY stop {trigger:.2f}"
            )

    def manage_after_fill(self, ts: pd.Timestamp, df: pd.DataFrame):
        s = self.oms.state
        # detect parent fill → initialize position state, SL and first TP
        if s.parent and not s.in_pos:
            try:
                if s.parent.filled() and s.parent.fills:
                    s.in_pos = True
                    s.entry_px = float(s.parent.fills[-1].execution.avgPrice)
                    s.total_qty = int(s.parent.order.totalQuantity)
                    s.bars_in_pos = 0
                    # initial R
                    atr14 = float(atr(df, 14).iloc[-1])
                    sl_px = s.entry_px - self.p.atr_sl_mult * atr14
                    s.r_val = s.entry_px - sl_px
                    self.oms.replace_stop_loss(sl_px, s.total_qty)
                    # first TP
                    self._ensure_lists()
                    tp_px = s.entry_px + self.p.tp_r_list[0] * s.r_val
                    qty_tp = max(1, int(round(self.p.tp_frac_list[0] * s.total_qty)))
                    self.oms.place_take_profit(tp_px, qty_tp)
                    s.tp_index = 0
                    s.last_add_trigger_px = s.entry_px
                    print(
                        f"[LONG] Filled {s.total_qty} @ {s.entry_px:.2f} | SL {sl_px:.2f} | TP1 {tp_px:.2f} qty={qty_tp}"
                    )
            except Exception as e:

                logger.debug("Swallowed exception in archive\paper_trader_long_only.py: %s", e)
        if not s.in_pos or s.entry_px is None or s.r_val is None:
            return

        # time stop
        s.bars_in_pos += 1
        if s.bars_in_pos >= self.p.time_stop_bars:
            progress = float(df["Close"].iloc[-1]) - s.entry_px
            if progress < self.p.min_progress_r * s.r_val:
                # exit market: cancel SL/TP first
                self.oms.cancel_all()
                if self.oms.place_orders:
                    m = Order(
                        action="SELL",
                        orderType="MKT",
                        totalQuantity=s.total_qty,
                        tif="DAY",
                        transmit=True,
                    )
                    self.ib.placeOrder(self.contract, m)
                    if self.book:
                        self.book.log_order(ts, "SELL", "MKT", s.total_qty, extra="time_stop")
                print(f"[EXIT] Time stop @ {float(df['Close'].iloc[-1]):.2f}")
                return  # state reset already in cancel_all()

        # scaling IN (add) on progress
        atr14 = float(atr(df, 14).iloc[-1])
        step = self.p.scale_in_step_atr * atr14
        c = float(df["Close"].iloc[-1])
        adds_done = max(0, (s.total_qty - self.default_qty) // max(1, self.p.add_qty))
        if (
            self.p.scale_in_max > 0
            and adds_done < self.p.scale_in_max
            and s.last_add_trigger_px is not None
        ):
            if c >= s.last_add_trigger_px + step:
                # add buy MKT
                qty_add = self.p.add_qty
                if self.oms.place_orders:
                    o = Order(
                        action="BUY",
                        orderType="MKT",
                        totalQuantity=qty_add,
                        tif="DAY",
                        transmit=True,
                    )
                    self.ib.placeOrder(self.contract, o)
                    if self.book:
                        self.book.log_order(ts, "BUY", "MKT(add)", qty_add, extra="scale-in")
                s.total_qty += qty_add
                s.last_add_trigger_px = c
                # refresh SL qty (keep price; adjust qty)
                if s.r_val is not None and s.entry_px is not None:
                    sl_px = s.entry_px - (s.r_val)  # keep original SL price relative to entry
                    self.oms.replace_stop_loss(sl_px, s.total_qty)
                # (optional) grow current TP qty proportionally
                if self.oms.state.tp_working and self.p.tp_frac_list:
                    cur_frac = self.p.tp_frac_list[min(s.tp_index, len(self.p.tp_frac_list) - 1)]
                    new_tp_qty = max(1, int(round(cur_frac * s.total_qty)))
                    # replace current TP with updated quantity (same price)
                    try:
                        tp_px = float(self.oms.state.tp_working.order.lmtPrice)
                    except Exception:
                        tp_px = (
                            s.entry_px
                            + self.p.tp_r_list[min(s.tp_index, len(self.p.tp_r_list) - 1)] * s.r_val
                        )
                    self.oms.place_take_profit(tp_px, new_tp_qty)
                print(f"[ADD] +{qty_add} @ ~{c:.2f} → total {s.total_qty}")

        # TP filled? then advance ladder and adjust SL
        try:
            if self.oms.state.tp_working and self.oms.state.tp_working.filled():
                s.tp_index += 1
                # after first TP → move SL to BE
                if s.tp_index == 1 and s.entry_px is not None:
                    self.oms.replace_stop_loss(s.entry_px, s.total_qty)
                    print(f"[BE] SL moved to breakeven {s.entry_px:.2f}")
                # place next TP if available
                self._ensure_lists()
                if s.tp_index < len(self.p.tp_r_list):
                    next_r = self.p.tp_r_list[s.tp_index]
                    tp_px = s.entry_px + next_r * s.r_val
                    frac = self.p.tp_frac_list[min(s.tp_index, len(self.p.tp_frac_list) - 1)]
                    qty_tp = max(1, int(round(frac * s.total_qty)))
                    self.oms.place_take_profit(tp_px, qty_tp)
                    print(f"[TP] Next TP @{tp_px:.2f} qty={qty_tp}")
                else:
                    # all TPs done: keep trailing via SL only
                    self.oms._cancel_trade(self.oms.state.tp_working)
                    self.oms.state.tp_working = None
                    print("[TP] Ladder complete; managing with SL only")
        except Exception as e:

            logger.debug("Swallowed exception in archive\paper_trader_long_only.py: %s", e)
        # If SL filled, reset state
        try:
            if self.oms.state.sl and self.oms.state.sl.filled():
                print("[EXIT] SL filled")
                self.oms.cancel_all()
        except Exception as e:

            logger.debug("Swallowed exception in archive\paper_trader_long_only.py: %s", e)


# ===================== Engine (on completed 1m bar) =====================
def on_completed_bar(
    ts: pd.Timestamp,
    o: float,
    h: float,
    l: float,
    c: float,
    strat: StrategyLongOnly,
    hist_df: pd.DataFrame,
    rth_only: bool,
    ib: IB,
    contract: Contract,
    verbose: bool,
    flat_at_local: str | None,
    book: Bookkeeper,
    ctrl: LongOnlyController,
    min_atr_ticks: float,
    max_spread_ticks: float,
    regime_off: bool,
    htf_off: bool,
):
    # hard flat guards
    def is_flat_at_now() -> bool:
        if not flat_at_local:
            return False
        try:
            now_ct = _now_ct()
            hh, mm = [int(x) for x in flat_at_local.split(":")]
            return (now_ct.hour, now_ct.minute) >= (hh, mm)
        except Exception:
            return False

    def is_rth_auto_flat() -> bool:
        now_ct = _now_ct()
        return (now_ct.hour == 14 and now_ct.minute >= 57) or (now_ct.hour >= 15)

    if is_flat_at_now() or is_rth_auto_flat():
        ctrl.oms.cancel_all()
        return

    # gates
    if rth_only and not is_rth(ts.tz_localize("UTC") if ts.tzinfo is None else ts):
        if verbose:
            print(f"[Gate] Outside RTH {ts}")
        return
    atr_now = float(atr(hist_df, 14).iloc[-1]) if len(hist_df) else None
    if atr_now and min_atr_ticks > 0 and (atr_now / TICK_SIZE) < min_atr_ticks:
        if verbose:
            print(f"[Gate] ATR {(atr_now/TICK_SIZE):.2f} < {min_atr_ticks}")
        return
    sp_ticks = spread_ticks_now(ib, contract)
    if sp_ticks is not None and max_spread_ticks > 0 and sp_ticks > max_spread_ticks:
        if verbose:
            print(f"[Gate] Spread {sp_ticks:.2f} > {max_spread_ticks}")
        return

    # regime/HTF (optional)
    ok_regime = True
    if not regime_off:
        a = float(adx(hist_df, 14).iloc[-1])
        ok_regime = a >= 10.0  # modest default for testing
        if verbose and not ok_regime:
            print(f"[Gate] Regime ADX={a:.1f}<10")
    ok_htf = True
    if not htf_off:
        htf = (
            hist_df.resample("60T")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna()
        )
        if len(htf) > 205:
            e200 = ema(htf["Close"], 200)
            slope = float(e200.iloc[-1] - e200.iloc[-5])
            ok_htf = slope > 0
            if verbose and not ok_htf:
                print("[Gate] HTF downtrend")
        else:
            ok_htf = True

    inds = strat.indicators(hist_df)
    if book:
        book.log_bar(ts, o, h, l, c)

    # manage any open/working orders first
    ctrl.manage_after_fill(ts, hist_df)

    # entry condition
    if ok_regime and ok_htf and strat.buy_signal(hist_df):
        ctrl.maybe_enter(ts, hist_df, qty=ctrl.default_qty)


# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)  # Paper default
    ap.add_argument("--clientId", type=int, default=9)
    ap.add_argument("--use-delayed", action="store_true")

    ap.add_argument("--fast", type=int, default=8)
    ap.add_argument("--slow", type=int, default=20)
    ap.add_argument("--rsi", type=int, default=14)
    ap.add_argument("--no-rsi", dest="use_rsi", action="store_false")
    ap.set_defaults(use_rsi=True)
    ap.add_argument("--qty", type=int, default=1)

    ap.add_argument("--place-orders", action="store_true")
    ap.add_argument("--rth-only", action="store_true")
    ap.add_argument("--min-atr-ticks", type=float, default=0.0)
    ap.add_argument("--max-spread-ticks", type=float, default=0.0)
    ap.add_argument("--flat-at", default=None, help="HH:MM Chicago")

    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--regime-off", action="store_true")
    ap.add_argument("--htf-off", action="store_true")

    # long-only params
    ap.add_argument("--stop-entry-atr-mult", type=float, default=0.5)
    ap.add_argument("--atr-sl-mult", type=float, default=1.8)
    ap.add_argument("--scale-in-max", type=int, default=2)
    ap.add_argument("--scale-in-step-atr", type=float, default=0.5)
    ap.add_argument("--add-qty", type=int, default=1)
    ap.add_argument("--tp-rs", type=str, default="1.0,2.0")
    ap.add_argument("--tp-fracs", type=str, default="0.5,0.5")
    ap.add_argument("--time-stop-bars", type=int, default=30)
    ap.add_argument("--min-progress-r", type=float, default=0.5)

    args = ap.parse_args()

    if IB is None:
        raise RuntimeError("ib-insync is required: pip install ib-insync")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    session_dir = os.path.join(RESULTS_DIR, dt.datetime.now().strftime("ib_session_%Y%m%d_%H%M%S"))
    book = Bookkeeper(session_dir)

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.clientId)
    ib.reqMarketDataType(4 if args.use_delayed else 1)
    md_type = "DELAYED (4)" if args.use_delayed else "LIVE (1)"
    print(
        f"Connected {args.host}:{args.port} clientId={args.clientId} | Market data type: {md_type}"
    )

    contract = pick_active_contract(ib)
    print(
        f"Using contract: {getattr(contract,'localSymbol','')}  (conId={getattr(contract,'conId',0)})"
    )

    # strategy + controller
    strat = StrategyLongOnly(args.fast, args.slow, args.rsi, args.use_rsi, rsi_buy=55.0)
    tp_r_list = [float(x) for x in args.tp_rs.split(",") if x.strip() != ""]
    tp_frac_list = [float(x) for x in args.tp_fracs.split(",") if x.strip() != ""]
    params = Params(
        stop_entry_atr_mult=args.stop_entry_atr_mult,
        atr_sl_mult=args.atr_sl_mult,
        scale_in_max=args.scale_in_max,
        scale_in_step_atr=args.scale_in_step_atr,
        add_qty=args.add_qty,
        tp_r_list=tp_r_list,
        tp_frac_list=tp_frac_list,
        time_stop_bars=args.time_stop_bars,
        min_progress_r=args.min_progress_r,
    )
    ctrl = LongOnlyController(
        params, ib, contract, book, place_orders=bool(args.place_orders), default_qty=args.qty
    )

    book.write_meta(
        {
            "contract": {
                "symbol": getattr(contract, "symbol", "ES"),
                "localSymbol": getattr(contract, "localSymbol", ""),
                "conId": getattr(contract, "conId", 0),
            },
            "market_data_type": md_type,
            "params": params.__dict__,
            "rth_only": args.rth_only,
            "gates": {
                "min_atr_ticks": args.min_atr_ticks,
                "max_spread_ticks": args.max_spread_ticks,
            },
        }
    )

    # warm-up ~400 bars of 1m history so filters have context
    completed: list[tuple] = []
    try:
        pre = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="2 D",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=False,
            keepUpToDate=False,
        )
        dfp = util.df(pre)
        if not dfp.empty:
            dfp.rename(
                columns={
                    "date": "Time",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                },
                inplace=True,
            )
            dfp = dfp[["Time", "Open", "High", "Low", "Close"]].tail(400).copy()
            completed = [tuple(row) for row in dfp.itertuples(index=False, name=None)]
            for row in completed:
                book.log_bar(*row)
            print(f"[WARMUP] Loaded {len(completed)} historical 1m bars.")
    except Exception as e:
        print(f"[WARMUP] Skipped: {e}")

    # real-time 5s → 1m aggregation; fallback streaming 1m
    rt = ib.reqRealTimeBars(contract, 5, whatToShow="TRADES", useRTH=False)
    last_rt_stamp = None
    rt_stale_loops = 0
    minute_bucket = None
    cur_bar = None
    last_processed_bucket = None
    fallback_hist = None

    print("▶ Long-only LIVE paper trading (buy to open; sell only to close). Ctrl+C to stop.")
    try:
        while True:
            ib.sleep(1.0)

            if rt and (fallback_hist is None) and len(rt) > 0:
                new = rt[-1]
                ts = pd.to_datetime(new.time)
                o, h, l, c = float(new.open_), float(new.high), float(new.low), float(new.close)
                if last_rt_stamp == ts:
                    rt_stale_loops += 1
                else:
                    rt_stale_loops = 0
                last_rt_stamp = ts
                if rt_stale_loops >= 25:
                    print("[INFO] RT bars stale → switching to streaming 1m")
                    fallback_hist = ib.reqHistoricalData(
                        contract,
                        endDateTime="",
                        durationStr="1800 S",
                        barSizeSetting="1 min",
                        whatToShow="TRADES",
                        useRTH=False,
                        keepUpToDate=True,
                    )
                    continue

                bucket = ts.floor("min")
                if minute_bucket is None:
                    minute_bucket = bucket
                    cur_bar = [bucket, o, h, l, c]
                elif bucket == minute_bucket:
                    cur_bar[2] = max(cur_bar[2], h)
                    cur_bar[3] = min(cur_bar[3], l)
                    cur_bar[4] = c
                else:
                    completed.append(tuple(cur_bar))
                    book.log_bar(*cur_bar)
                    minute_bucket = bucket
                    cur_bar = [bucket, o, h, l, c]
                    if completed and (last_processed_bucket != completed[-1][0]):
                        last_ts, oo, hh, ll, cc = completed[-1]
                        hist_df = pd.DataFrame(
                            completed, columns=["Time", "Open", "High", "Low", "Close"]
                        ).set_index("Time")
                        on_completed_bar(
                            pd.Timestamp(last_ts),
                            oo,
                            hh,
                            ll,
                            cc,
                            strat,
                            hist_df,
                            args.rth_only,
                            ib,
                            contract,
                            args.verbose,
                            args.flat_at,
                            book,
                            ctrl,
                            args.min_atr_ticks,
                            args.max_spread_ticks,
                            args.regime_off,
                            args.htf_off,
                        )
                        last_processed_bucket = completed[-1][0]

            elif fallback_hist is not None and len(fallback_hist) > 0:
                try:
                    bars = list(fallback_hist)
                    df = util.df(bars)
                    if df.empty:
                        continue
                    df.rename(
                        columns={
                            "date": "Time",
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                        },
                        inplace=True,
                    )
                    df = df[["Time", "Open", "High", "Low", "Close"]].copy()
                    df.set_index("Time", inplace=True)
                    last_ts = df.index[-1].to_pydatetime()
                    if (last_processed_bucket is None) or (last_ts != last_processed_bucket):
                        oo, hh, ll, cc = map(float, df.iloc[-1][["Open", "High", "Low", "Close"]])
                        hist_df = df.copy()
                        book.log_bar(last_ts, oo, hh, ll, cc)
                        on_completed_bar(
                            pd.Timestamp(last_ts),
                            oo,
                            hh,
                            ll,
                            cc,
                            strat,
                            hist_df,
                            args.rth_only,
                            ib,
                            contract,
                            args.verbose,
                            args.flat_at,
                            book,
                            ctrl,
                            args.min_atr_ticks,
                            args.max_spread_ticks,
                            args.regime_off,
                            args.htf_off,
                        )
                        last_processed_bucket = last_ts
                except Exception as e:
                    print(f"[WARN] Fallback stream error: {e}")

    except KeyboardInterrupt:
        pass
    finally:
        ctrl.oms.cancel_all()
        try:
            ib.disconnect()
        except Exception as e:

            logger.debug("Swallowed exception in archive\paper_trader_long_only.py: %s", e)


if __name__ == "__main__":
    main()
