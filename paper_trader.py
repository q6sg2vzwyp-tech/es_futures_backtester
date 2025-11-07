#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ES Paper Trader (IBKR + ib_insync)
- Paper-only safety by default
- Thompson Sampling learner (shadow/advisory/control)
- Risk profiles: balanced/aggressive/conservative
- OCO protection builder & audit (orphan sweeps, sibling cancel)
- VWAP session + optional short guards (VWAP buffer + lower-high)
- Session cutovers (multi) + persistent day guard
- Version-safe IB PnL subscription + NetLiq via accountValueEvent
- 24/5 trading window by default (no TOD blackouts unless provided)
- News kill switch: file flag + optional TOD windows + IBKR news bulletins
- 1-second JSON heartbeats with explicit idle reasons & RT status/age/queue
- Robust RTâ†’Polling fallback (5s bars via historical polling)
- **NEW**:
  - Error logging, market-data warmup, RT-starvedâ‡’poll fallback, MIDPOINT auto-resubscribe
  - Parameter meta-learning (Thompson) over per-trade parameter sets
  - Persistent save/load of learners to JSON (auto-save after each flat)
  - Auto self-backup of this script into .\backups\ on startup
"""

from __future__ import annotations
import sys, os, time, json, math, random, argparse, datetime as dt, re, traceback, threading, shutil
from typing import Optional, List, Dict, Any, Tuple
from ib_insync import IB, Future, Contract, LimitOrder, StopOrder, MarketOrder, Trade

# ---------- Utilities ----------
def utc_now_str(): return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
def ct_now() -> dt.datetime: return dt.datetime.now()
def parse_hhmm(s: str) -> dt.time: h,m = s.split(":"); return dt.time(int(h), int(m))
def clamp(x, lo, hi): return max(lo, min(hi, x))
def ticks_to_price_delta(ticks: int, tick_size: float) -> float: return float(ticks) * float(tick_size)
def round_to_tick(p: float, tick: float) -> float: return round(p / tick) * tick if tick > 0 else p

def log(evt: str, **fields):
    payload = {"ts": utc_now_str(), "evt": evt}; payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False), flush=True)

# ---------- Self-backup on start ----------
def self_backup():
    try:
        src = os.path.abspath(__file__)
        bdir = os.path.join(os.path.dirname(src), "backups")
        os.makedirs(bdir, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(bdir, f"paper_trader_{ts}.py")
        shutil.copy2(src, dst)
        log("self_backup_ok", path=dst)
    except Exception as e:
        log("self_backup_warn", err=str(e))

# ---------- Math / indicators ----------
def ema(vals: List[float], span: int) -> float:
    if not vals: return float("nan")
    k = 2/(span+1); s = vals[0]
    for v in vals[1:]: s = v*k + s*(1-k)
    return s

def atr(H: List[float], L: List[float], C: List[float], n: int = 14) -> float:
    if len(C) < n+1: return float("nan")
    trs = []
    for i in range(1, len(C)):
        hl = H[i] - L[i]; hc = abs(H[i] - C[i-1]); lc = abs(L[i] - C[i-1])
        trs.append(max(hl, hc, lc))
    if len(trs) < n: return float("nan")
    k = 2/(n+1)
    s = trs[-n]
    for v in trs[-n+1:]:
        s = v*k + s*(1-k)
    return s

# ---------- Session helpers ----------
def parse_ct_list(spec: str) -> List[dt.time]:
    spec = (spec or "").strip()
    if not spec: return [parse_hhmm("16:10")]
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk: continue
        try: out.append(parse_hhmm(chunk))
        except Exception: pass
    out = sorted(list({t for t in out}))
    return out or [parse_hhmm("16:10")]

def within_session(now: dt.datetime, start_ct: str, end_ct: str) -> bool:
    t = now.time(); a = parse_hhmm(start_ct); b = parse_hhmm(end_ct)
    if a <= b: return a <= t <= b
    return (t >= a) or (t <= b)

def parse_blackouts(spec: str) -> List[Tuple[dt.time, dt.time]]:
    out = []
    spec = (spec or "").strip()
    if not spec: return out
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk: continue
        try:
            a, b = chunk.split("-")
            out.append((parse_hhmm(a), parse_hhmm(b)))
        except Exception:
            pass
    return out

def in_tod_blackout(now: dt.datetime, blackouts: List[Tuple[dt.time, dt.time]]) -> bool:
    if not blackouts: return False
    t = now.time()
    for a, b in blackouts:
        if a <= b:
            if a <= t <= b: return True
        else:  # crosses midnight
            if (t >= a) or (t <= b): return True
    return False

def session_key_multi(now: dt.datetime, reset_times: List[dt.time]) -> str:
    t = now.time()
    idx_today = -1
    for i, ct in enumerate(reset_times):
        if t >= ct: idx_today = i
        else: break
    if idx_today >= 0:
        base_date = now.date(); seg = idx_today
    else:
        base_date = (now - dt.timedelta(days=1)).date(); seg = len(reset_times) - 1
    return f"{base_date.strftime('%Y-%m-%d')}-S{seg}"

def reset_due_multi(now: dt.datetime, reset_times: List[dt.time], last_reset_marks: Dict[str, str]) -> Optional[str]:
    today = now.date().strftime("%Y-%m-%d")
    for ct in reset_times:
        label = ct.strftime("%H:%M")
        if last_reset_marks.get(label) == today:
            continue
        if now.time() >= ct:
            last_reset_marks[label] = today
            return f"{today}#{label}"
    return None

# ---------- Heartbeat (thread) ----------
_hb_lock = threading.Lock()
_hb_state: Dict[str, Any] = {
    "state": "-",
    "idle_reason": "starting_or_quiet",
    "net_qty": 0,
    "bars": 0,
    "rt_enabled": False,
    "rt_status": "disabled",
    "rt_age_sec": None,
    "rt_queue_len": 0,
    "in_session_window": False,
    "caps": [],
    "news_kill": False,
    "dayR": 0.0,
    "trades_today": 0,
    "cool_until": None,
    "orders_disabled_paper_safety": False,
}
def hb_update(**kv):
    with _hb_lock:
        _hb_state.update(kv)
def _hb_loop():
    while True:
        with _hb_lock:
            payload = dict(_hb_state)
        log("hb", **payload)
        time.sleep(1.0)
def start_heartbeat_thread():
    t = threading.Thread(target=_hb_loop, daemon=True)
    t.start()

# ---------- Thompson learner ----------
class ThompsonGaussian:
    def __init__(self, arms: List[str], decay_gamma: float, prior_mean=0.0, prior_var=0.25):
        self.arms = arms[:]
        self.gamma = decay_gamma
        self.m = {a: prior_mean for a in arms}
        self.s2 = {a: prior_var for a in arms}
        self.w = {a: 1e-6 for a in arms}
        self.last_arm: Optional[str] = None
    def choose(self, cand_arms: List[str], sample: bool) -> Tuple[str, Dict[str, float]]:
        scores = {}
        for a in cand_arms:
            std = math.sqrt(max(1e-6, self.s2[a] / (self.w[a] + 1.0)))
            scores[a] = random.gauss(self.m[a], std)
        m = max(scores.values()) if scores else 0.0
        exps = {a: math.exp(scores[a] - m) for a in cand_arms}
        s = sum(exps.values()) or 1.0
        probs = {a: exps[a]/s for a in cand_arms}
        choice = max(probs.items(), key=lambda kv: kv[1])[0]
        if sample:
            r, cum = random.random(), 0.0
            for a in cand_arms:
                cum += probs[a]
                if r <= cum:
                    choice = a; break
        return choice, probs
    def update(self, arm: str, reward_R: float):
        g = self.gamma
        w_old = self.w[arm]
        self.w[arm] = g*w_old + 1.0
        m_old = self.m[arm]
        m_new = m_old + (reward_R - m_old) / self.w[arm]
        s2_old = self.s2[arm]
        s2_new = g*s2_old + (reward_R - m_old)*(reward_R - m_new)
        self.m[arm] = m_new
        self.s2[arm] = max(1e-6, s2_new)
        self.last_arm = arm
    # ---- persistence ----
    def to_dict(self) -> Dict[str, Any]:
        return {"arms": self.arms, "gamma": self.gamma, "m": self.m, "s2": self.s2, "w": self.w, "last_arm": self.last_arm}
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ThompsonGaussian":
        obj = ThompsonGaussian(d["arms"], float(d["gamma"]))
        obj.m = {k: float(v) for k, v in d["m"].items()}
        obj.s2 = {k: float(v) for k, v in d["s2"].items()}
        obj.w = {k: float(v) for k, v in d["w"].items()}
        obj.last_arm = d.get("last_arm")
        return obj

# ---------- VWAP ----------
class SessionVWAP:
    def __init__(self): self.pv=0.0; self.v=0.0; self.vwap=float("nan")
    def reset(self): self.pv=0.0; self.v=0.0; self.vwap=float("nan")
    def update_bar(self, close_px: Optional[float], volume: Optional[float]):
        try:
            if close_px is None or volume is None: return
            vol = max(0.0, float(volume))
            if vol <= 0: return
            px = float(close_px)
            self.pv += px * vol; self.v += vol
            if self.v > 0: self.vwap = self.pv / self.v
        except Exception:
            pass

# ---------- IB helpers ----------
ACTIVE_STATUSES = {"Submitted","PreSubmitted","ApiPending","PendingSubmit","PendingCancel","Inactive"}
CANCELLABLE_STATUSES = {"Submitted","PreSubmitted","ApiPending","PendingSubmit"}

def _parse_ib_date(s: str) -> Optional[dt.date]:
    try: return dt.datetime.strptime(s, "%Y%m%d").date()
    except Exception:
        try: return dt.datetime.strptime(s, "%Y%m%d%H:%M:%S").date()
        except Exception: return None

def qualify_local_symbol(ib: IB, local_symbol: str, exchange="CME"):
    cds = ib.reqContractDetails(Future(localSymbol=local_symbol, exchange=exchange))
    if not cds: raise RuntimeError(f"Local symbol {local_symbol} not found on {exchange}")
    con = cds[0].contract
    ib.qualifyContracts(con)
    return con

def mk_contract(ib: IB, args) -> Contract:
    if getattr(args, "local_symbol", None):
        con = qualify_local_symbol(ib, args.local_symbol, "CME")
        print(f"[CONTRACT] Using {con.localSymbol} conId={con.conId} exp={con.lastTradeDateOrContractMonth}")
        return con
    cds = ib.reqContractDetails(Future(symbol=args.symbol, exchange="CME", currency="USD"))
    if not cds: raise RuntimeError(f"Symbol {args.symbol} not found on CME; supply --local-symbol")
    best = None; best_date = None
    for cd in cds:
        d = _parse_ib_date(cd.contract.lastTradeDateOrContractMonth)
        if not d: continue
        if best is None or d < best_date:
            best = cd.contract; best_date = d
    if best is None: raise RuntimeError("Could not resolve front contract; supply --local-symbol")
    ib.qualifyContracts(best)
    print(f"[CONTRACT] Using {best.localSymbol} conId={best.conId} exp={best.lastTradeDateOrContractMonth}")
    return best

def contract_multiplier(ib: IB, con: Contract) -> float:
    try:
        cds = ib.reqContractDetails(con)
        mul = cds[0].contract.multiplier
        m = float(mul) if mul is not None else 1.0
        return m if m > 0 else 1.0
    except Exception:
        return 1.0

def ib_position_truth(ib: IB, con: Contract) -> Tuple[int, Optional[float]]:
    try:
        qty = 0; avg = None
        for p in ib.positions():
            if getattr(p.contract, "conId", None) == con.conId:
                qty += int(round(p.position)); avg = float(p.avgCost)
        return qty, avg
    except Exception:
        return 0, None

def has_active_parent_entry(ib: IB, con: Contract) -> bool:
    try:
        for t in ib.openTrades():
            if getattr(t.contract, "conId", None) != con.conId: continue
            st = (getattr(t.orderStatus, "status", "") or "").strip()
            if st not in ACTIVE_STATUSES: continue
            if (t.order.orderType or "").upper() == "LMT" and (t.order.parentId in (None, 0)):
                act = (t.order.action or "").upper()
                if act in ("BUY","SELL"): return True
    except Exception: pass
    return False

def list_open_orders_for_contract(ib: IB, con: Contract) -> List[Trade]:
    trades = []
    try:
        for t in ib.openTrades():
            if getattr(t.contract, "conId", None) == con.conId:
                st = (t.orderStatus.status or "").strip()
                if st in ACTIVE_STATUSES: trades.append(t)
    except Exception: pass
    return trades

def safe_cancel(ib: IB, order_or_trade, note: str = ""):
    try:
        if hasattr(order_or_trade, 'orderStatus'):
            st = (order_or_trade.orderStatus.status or "").strip()
            oid = order_or_trade.order.orderId
            tgt = order_or_trade.order
        else:
            st = (getattr(order_or_trade, 'status', None) or "").strip()
            oid = getattr(order_or_trade, 'orderId', None)
            tgt = order_or_trade
        if st and st not in CANCELLABLE_STATUSES: return
        ib.cancelOrder(tgt)
        log("cancel_sent", orderId=oid, note=note)
    except Exception as e:
        msg = str(e)
        if "Error 161" in msg:
            log("cancel_ignored_161", orderId=oid, note=note)
        else:
            log("cancel_warn", orderId=oid, err=msg, note=note)

def cancel_siblings_for_trade(ib: IB, trade: Trade, con: Contract):
    try:
        ocag = (trade.order.ocaGroup or "").strip()
        if ocag:
            for t in ib.openTrades():
                if getattr(t.contract, "conId", None) != con.conId: continue
                if (t.order.ocaGroup or "").strip() == ocag:
                    if t.order.orderId != trade.order.orderId:
                        safe_cancel(ib, t, note=f"[sibling OCA={ocag}]")
            return
        my_side = (trade.order.action or "").upper()
        opp = "SELL" if my_side == "BUY" else "BUY"
        for t in ib.openTrades():
            if getattr(t.contract, "conId", None) != con.conId: continue
            st = (t.orderStatus.status or "").strip()
            if st not in ACTIVE_STATUSES: continue
            ot = (t.order.orderType or "").upper()
            if ot in {"LMT", "STP", "STP LMT"} and (t.order.action or "").upper() == opp:
                safe_cancel(ib, t, note="[sibling fallback]")
    except Exception as e:
        log("sibling_cancel_err", err=str(e))

def reconcile_orphans(ib: IB, account: str, con: Contract):
    try:
        qty, _ = ib_position_truth(ib, con)
        if qty != 0: return
        for t in ib.openTrades():
            if getattr(t.contract, "conId", None) != con.conId: continue
            st = (t.orderStatus.status or "").strip()
            if st in {"Submitted", "PreSubmitted"}:
                ot = (t.order.orderType or "").upper()
                if ot in {"LMT", "STP", "STP LMT"} and (t.order.parentId not in (None, 0) or t.order.ocaGroup):
                    safe_cancel(ib, t, note="[ORPHAN SWEEP]")
    except Exception as e:
        log("orphan_sweep_err", err=str(e))

# ---------- Risk & sizing ----------
class DayRisk:
    def __init__(self, loss_cap_R: float, max_trades: int, max_consec_losses: int):
        self.loss_cap_R = float(loss_cap_R); self.max_trades = int(max_trades)
        self.max_consec_losses = int(max_consec_losses)
        self.reset()
    def reset(self):
        self.day_R = 0.0; self.trades = 0
        self.cool_until: Optional[dt.datetime] = None
        self.halted = False
        self.last_entry_time: Optional[float] = None
        self.consec_losses = 0
    def can_trade(self, now: dt.datetime, min_gap_s:int) -> bool:
        if self.halted: return False
        if self.cool_until and now < self.cool_until: return False
        if self.trades >= self.max_trades: return False
        if self.day_R <= -abs(self.loss_cap_R): return False
        if self.consec_losses >= self.max_consec_losses: return False
        if self.last_entry_time and (time.time() - self.last_entry_time) < max(0, min_gap_s): return False
        return True

def iso_week_id(d: dt.date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{int(w):02d}"

def decay_factor_from_half_life(hl_trades: float) -> float:
    hl = max(1.0, float(hl_trades))
    return math.exp(math.log(0.5)/hl)

# ---------- News helpers ----------
def parse_news_blackouts(spec: str) -> List[Tuple[dt.time, dt.time]]:
    return parse_blackouts(spec)

def read_news_file_flag(path: str) -> Tuple[bool, Optional[dt.datetime]]:
    try:
        if not path: return (False, None)
        path = path.strip().strip('"').strip("'")
        if not os.path.exists(path): return (False, None)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        on = bool(obj.get("on", False))
        until_s = obj.get("until")
        until = None
        if until_s:
            try:
                until = dt.datetime.strptime(until_s, "%Y-%m-%d %H:%M")
            except Exception:
                pass
        return (on, until)
    except Exception as e:
        log("news_file_read_err", err=str(e))
        return (False, None)

# ---------- Param arms parsing ----------
def parse_param_arms(spec: str) -> Dict[str, Dict[str, float]]:
    out = {}
    spec = (spec or "").strip()
    if not spec: return out
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk: continue
        try:
            name, rest = chunk.split(":", 1)
            kv = {}
            for part in rest.split(","):
                part = part.strip()
                if not part: continue
                k, v = part.split("=", 1)
                kv[k.strip()] = float(v.strip())
            out[name.strip()] = kv
        except Exception:
            pass
    return out

# ---------- Persistence for learners ----------
def mkdirs(p): 
    if p: os.makedirs(p, exist_ok=True)

def learner_paths(base_dir: str, stem: str) -> str:
    mkdirs(base_dir)
    return os.path.join(base_dir, f"{stem}.json")

def load_thompson(path: str) -> Optional[ThompsonGaussian]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return ThompsonGaussian.from_dict(d)
    except Exception as e:
        log("learner_load_err", path=path, err=str(e))
    return None

def save_thompson(path: str, learner: ThompsonGaussian):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(learner.to_dict(), f)
        os.replace(tmp, path)
        log("learner_saved", path=path)
    except Exception as e:
        log("learner_save_err", path=path, err=str(e))

# ---------- CLI ----------
def build_argparser():
    ap = argparse.ArgumentParser(description="ES Paper Trader (session-aware + Thompson learner + rails)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--clientId", type=int, default=111)
    ap.add_argument("--symbol", default="ES")
    ap.add_argument("--local-symbol", dest="local_symbol", default="")
    ap.add_argument("--place-orders", action="store_true")
    ap.add_argument("--tif", default="GTC")
    ap.add_argument("--outsideRth", action="store_true")

    # Sizing & Risk
    ap.add_argument("--acct-base", type=float, default=30000.0)
    ap.add_argument("--risk-pct", type=float, default=0.01)
    ap.add_argument("--scale-step", type=float, default=10000.0)
    ap.add_argument("--start-contracts", type=int, default=2)
    ap.add_argument("--max-contracts", type=int, default=6)
    ap.add_argument("--static-size", action="store_true")
    ap.add_argument("--qty", type=float, default=2.0)
    ap.add_argument("--risk-ticks", type=int, default=12)
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--tp-R", type=float, default=1.0)

    # Margin awareness
    ap.add_argument("--margin-per-contract", type=float, default=13200.0)
    ap.add_argument("--margin-reserve-pct", type=float, default=0.10)

    # Strategy gates
    ap.add_argument("--enable-arms", default="trend,breakout")
    ap.add_argument("--gate-adx", type=float, default=19.0)
    ap.add_argument("--gate-atrp", type=float, default=0.000055)
    ap.add_argument("--gate-bbbw", type=float, default=0.0)  # 0 disables

    # Anti-burst & day/session rails
    ap.add_argument("--min-seconds-between-entries", type=int, default=20)
    ap.add_argument("--max-trades-per-day", type=int, default=12)
    ap.add_argument("--day-loss-cap-R", type=float, default=3.0)
    ap.add_argument("--max-consec-losses", type=int, default=3)
    ap.add_argument("--strategy-cooldown-sec", type=int, default=150)

    # Risk governance extras
    ap.add_argument("--pos-age-cap-sec", type=int, default=1200)
    ap.add_argument("--pos-age-minR", type=float, default=0.5)
    ap.add_argument("--hwm-stepdown", action="store_true")
    ap.add_argument("--hwm-stepdown-dollars", type=float, default=5000.0)

    # Trading window (24/5) + optional TOD blackouts
    ap.add_argument("--trade-start-ct", default="00:00")
    ap.add_argument("--trade-end-ct", default="23:59")
    ap.add_argument("--tod-blackouts", default="")

    # Order behavior
    ap.add_argument("--entry-slippage-ticks", type=int, default=2)
    ap.add_argument("--require-new-bar-after-start", action="store_true")
    ap.add_argument("--startup-delay-sec", type=int, default=0)
    ap.add_argument("--debounce-one-bar", action="store_true")

    # Session resets (AM/PM logging only)
    ap.add_argument("--session-reset-cts", default="08:30,16:00,17:00")
    ap.add_argument("--daily-reset-ct", default="16:10")  # legacy

    # Connectivity & data
    ap.add_argument("--connect-timeout-sec", type=int, default=60)
    ap.add_argument("--timeout-sec", type=int, default=60)
    ap.add_argument("--connect-attempts", type=int, default=10)
    ap.add_argument("--force-delayed", action="store_true")
    ap.add_argument("--poll-hist-when-no-rt", action="store_true")
    ap.add_argument("--poll-interval-sec", type=int, default=10)
ap.add_argument("--no-poll-fallback", action="store_true",
                help="Disable historical polling fallback when RT is starved/stale")
    ap.add_argument("--require-rt-before-trading", action="store_true")
    ap.add_argument("--rt-staleness-sec", type=int, default=45)

    # Learning
    ap.add_argument("--bandit", choices=["thompson"], default="thompson")
    ap.add_argument("--learn-mode", choices=["shadow","advisory","control"], default="advisory")
    ap.add_argument("--decay-half-life-trades", type=float, default=200.0)
    ap.add_argument("--learn-log", action="store_true")
    ap.add_argument("--learn-log-dir", default=r".\logs\learn")
    # Parameter meta-learning arms (Bayesian-ish tuning)
    ap.add_argument("--param-arms", default="",
                    help="Semicolon-separated list like 'A:risk_ticks=10,tp_R=1.0,entry_slippage_ticks=1; B:risk_ticks=12,tp_R=1.2'")

    # PnL & equity sync
    ap.add_argument("--use-ib-pnl", action="store_true")
    ap.add_argument("--peak-dd-guard-pct", type=float, default=0.60)
    ap.add_argument("--day-guard-pct", type=float, default=0.025)
    ap.add_argument("--peak-dd-min-profit", type=float, default=1500.0)

    # Short guard rails & VWAP control
    ap.add_argument("--short-guard-vwap-buffer-ticks", type=int, default=4)
    ap.add_argument("--short-guard-min-pullback-ticks", type=int, default=6)
    ap.add_argument("--short-guard-lookback-bars", type=int, default=60)
    ap.add_argument("--short-guard-vwap", action="store_true", default=True)
    ap.add_argument("--no-short-guard-vwap", dest="short_guard_vwap", action="store_false")
    ap.add_argument("--short-guard-lower-high", action="store_true", default=True)
    ap.add_argument("--no-short-guard-lower-high", dest="short_guard_lower_high", action="store_false")
    ap.add_argument("--vwap-reset-on-session", action="store_true", default=True)
    ap.add_argument("--no-vwap-reset-on-session", dest="vwap_reset_on_session", action="store_false")

    # Safety
    ap.add_argument("--allow_live", action="store_true")

    # Risk profile selector
    ap.add_argument("--risk-profile", choices=["balanced","aggressive","conservative"], default="balanced")

    # News kill
    ap.add_argument("--news-file-kill", default=r".\data\kill\news_kill.json")
    ap.add_argument("--news-flatten-on-kill", action="store_true")
    ap.add_argument("--news-cancel-only", action="store_true")
    ap.add_argument("--news-blackouts", default="")
    ap.add_argument("--news-bulletin-listen", action="store_true")
    ap.add_argument("--news-keywords", default="FOMC,rate,nonfarm,employment,inflation,CPI,PPI,ISM,PMI,Jerome Powell,press conference")
    ap.add_argument("--news-kill-minutes", type=int, default=15)

    # Optional CSV
    ap.add_argument("--segment-trade-csv", default=r".\logs\trades_segmented.csv")
    return ap

# ---------- Main ----------
def main():
    self_backup()  # snapshot the file to .\backups\ at each start

    args = build_argparser().parse_args()

    # Start HB immediately
    start_heartbeat_thread()
    hb_update(state="-", idle_reason="booting", rt_enabled=False, rt_status="disabled",
              rt_age_sec=None, rt_queue_len=0, bars=0, net_qty=0)

    # Connect (with retries)
    ib = IB()

    # ---- Error logging so entitlement/contract issues are visible ----
    def _on_err(reqId, code, msg, misc):
        try:
            log("ib_error", reqId=reqId, code=int(code), msg=str(msg))
        except Exception:
            print(f"[IB-ERR] {code} {msg}")
    ib.errorEvent += _on_err

    def connect_with_retries():
        base = int(args.clientId)
        for i in range(max(1, int(args.connect_attempts))):
            cid = base + i
            try:
                print(f"[CONNECT] Attempt {i+1}/{args.connect_attempts} -> clientId={cid}")
                log("boot_progress", step="connecting")
                ib.connect(args.host, args.port, clientId=cid, timeout=args.connect_timeout_sec)
                ib.sleep(0.6)
                if ib.isConnected():
                    print(f"Connected (clientId={cid})")
                    print(f"[POST-CONNECT] isConnected=True host={args.host} port={args.port} clientId={cid}")
                    try:
                        accts = ib.managedAccounts()
                        print(f"[POST-CONNECT] managedAccounts: {accts}")
                    except Exception: pass
                    return cid
            except Exception as e:
                print(f"[CONNECT] Failed: {repr(e)}")
                try: ib.disconnect()
                except Exception: pass
                ib.sleep(0.5 + 0.25*i)
        return None

    cid = connect_with_retries()
    if cid is None:
        print("ERROR [CONNECT] Could not establish connection."); return

    # Paper-only safety
    if not args.allow_live:
        try:
            accts = ib.managedAccounts(); acct = accts[0] if accts else None
        except Exception:
            acct = None
        bad_port = (getattr(args, 'port', None) != 7497)
        bad_acct = (acct is not None and not str(acct).upper().startswith("DU"))
        orders_disabled = bad_port or bad_acct
        hb_update(orders_disabled_paper_safety=orders_disabled)
        if orders_disabled:
            print(f"[SAFE] Paper-only: refusing to trade (port={getattr(args,'port',None)}, account={acct}).")
            return

    # Market data type
    try:
        ib.reqMarketDataType(3 if args.force_delayed else 1)
        print("[MD] DELAYED (3)." if args.force_delayed else "[MD] LIVE (1).")
        hb_update(rt_enabled=not args.force_delayed, rt_status=("ok" if not args.force_delayed else "disabled"))
    except Exception as e:
        print("[MD] marketDataType failed:", repr(e))

    # Contract
    try:
        con = mk_contract(ib, args)
    except Exception as e:
        print("[CONTRACT] Error:", repr(e)); return
    px_mult = contract_multiplier(ib, con)
    print(f"[CONTRACT] Multiplier detected: {px_mult:g}")

    # Warm up L1 quotes
    try:
        tkr = ib.reqMktData(con, genericTickList="", snapshot=False, regulatorySnapshot=False)
        ib.sleep(1.0)
        ib.cancelMktData(con)
        log("md_warmup", ok=True)
    except Exception as e:
        log("md_warmup_warn", err=str(e))

    # -------- PnL & NetLiq (version-safe) --------
    ib_acct = None
    ib_daily_pnl: Optional[float] = None
    ib_netliq: Optional[float] = None

    if args.use_ib_pnl:
        try:
            accts = ib.managedAccounts()
            if accts:
                ib_acct = accts[0]
                def _on_pnl(p):
                    nonlocal ib_daily_pnl
                    try:
                        ib_daily_pnl = float(getattr(p, "dailyPnL", 0.0) or 0.0)
                    except Exception:
                        pass
                ib.pnlEvent += _on_pnl
                ib.reqPnL(ib_acct, "")

                def _on_account_value(v):
                    nonlocal ib_netliq
                    try:
                        if v.tag == "NetLiquidation" and (not ib_acct or v.account == ib_acct):
                            ib_netliq = float(v.value)
                    except Exception:
                        pass
                ib.accountValueEvent += _on_account_value

                try:
                    ib.reqAccountUpdates(account=ib_acct)
                except TypeError:
                    try: ib.reqAccountUpdates(True)
                    except Exception: pass

                print(f"[IB PNL] Sync enabled for account={ib_acct}")
            else:
                print("[IB PNL] No managed accounts found; IB P&L sync disabled.")
        except Exception as e:
            print(f"[IB PNL] Failed to subscribe: {e}")

    # Seed history + RT bars
    H: List[float] = []; L: List[float] = []; C: List[float] = []; V: List[float] = []
    last_bar_ts = None
    vwap = SessionVWAP()

    def _bar_ts(b):
        return getattr(b, "time", None) or getattr(b, "date", None)

    try:
        hist = ib.reqHistoricalData(con, endDateTime='', durationStr='1800 S', barSizeSetting='5 secs',
                                    whatToShow='TRADES', useRTH=False, keepUpToDate=False)
        for b in hist:
            H.append(b.high); L.append(b.low); C.append(b.close)
            vol = getattr(b, "volume", None); V.append(vol if vol is not None else 0.0)
            vwap.update_bar(b.close, vol)
            last_bar_ts = _bar_ts(b)
        print(f"[BOOT] Hist seed: {len(hist)} bars")
        hb_update(bars=len(C))
    except Exception as e:
        log("hist_seed_err", err=repr(e))

    # Subscribe RT bars
    rt = None
    rt_mode = "TRADES"
    try:
        rt = ib.reqRealTimeBars(con, 5, 'TRADES', False)
        if rt is None:
            log("rt_subscribe_warn", msg="reqRealTimeBars returned None; polling may be used")
        else:
            hb_update(rt_enabled=True, rt_status="ok")
    except Exception as e:
        rt = None
        log("rt_subscribe_err", err=repr(e))

    _rt_last_seen = None
    poll_enabled = bool(args.poll_hist_when_no_rt)
    poll_iv = max(5, int(args.poll_interval_sec)) if poll_enabled else 999999
    _last_poll = 0.0

    # Risk state
    risk = DayRisk(args.day_loss_cap_R, args.max_trades_per_day, args.max_consec_losses)
    last_entry_bar_ts = None
    current_arm: Optional[str] = None
    entry_price: Optional[float] = None
    prev_net_qty = 0
    last_exec_price: Optional[float] = None
    cycle_commission = 0.0
    in_trade_cycle = False
    cycle_entry_qty = 0
    realized_pnl_total = 0.0
    equity = float(args.acct_base)
    equity_hwm = equity
    realized_pnl_day = 0.0
    cycle_entry_time: Optional[dt.datetime] = None
    age_forced_flat_done = False
    week_R = 0.0
    week_halted = False
    last_week_id = iso_week_id(ct_now().date())
    weekly_cap_R = -min(5.0, 2.0 * abs(args.day_loss_cap_R))

    # Sessions (AM/PM segmentation only)
    session_cutovers = parse_ct_list(getattr(args, "session_reset_cts", "08:30,16:00,17:00"))
    last_reset_marks: Dict[str, str] = {}

    # News kill state
    news_kill_until: Optional[dt.datetime] = None
    news_blackouts = parse_news_blackouts(args.news_blackouts)
    news_keywords = [s.strip().lower() for s in (args.news_keywords or "").split(",") if s.strip()]

    # Learners (with persistence)
    gamma = decay_factor_from_half_life(args.decay_half_life_trades)
    arms_enabled = [a.strip() for a in args.enable_arms.split(",") if a.strip()] or ["trend","breakout"]

    learn_dir = os.path.abspath(getattr(args, "learn_log_dir", r".\logs\learn"))
    strat_path = learner_paths(learn_dir, "strategy_thompson")
    param_path = learner_paths(learn_dir, "param_thompson")

    learner = load_thompson(strat_path)
    if learner is None or set(learner.arms) != set(arms_enabled):
        learner = ThompsonGaussian(arms_enabled, gamma)

    param_arms = parse_param_arms(getattr(args, "param_arms", ""))
    param_learner: Optional[ThompsonGaussian] = None
    if param_arms:
        existing = load_thompson(param_path)
        if existing and set(existing.arms) == set(param_arms.keys()):
            param_learner = existing
        else:
            param_learner = ThompsonGaussian(list(param_arms.keys()), gamma)
    current_param_arm: Optional[str] = None

    # Safe, deferred sibling/orphan triggers
    sibling_cancel_needed = False
    orphan_sweep_needed = False

    def _on_exec(trade, fill):
        nonlocal last_exec_price, sibling_cancel_needed, orphan_sweep_needed
        try: last_exec_price = float(fill.price)
        except Exception: pass
        try:
            st = (trade.orderStatus.status or "").strip()
            if st in ("Filled", "PartiallyFilled"):
                sibling_cancel_needed = True
                orphan_sweep_needed = True
        except Exception: pass
    ib.execDetailsEvent += _on_exec

    def _on_status(trade):
        nonlocal sibling_cancel_needed, orphan_sweep_needed
        try:
            if (trade.orderStatus.status or "").strip() == "Filled":
                sibling_cancel_needed = True
                orphan_sweep_needed = True
        except Exception: pass
    ib.orderStatusEvent += _on_status

    # Day guard persistence
    DAY_STATE_PATH = r".\data\state\day_guard.json"
    def load_day_state() -> Dict[str, Any]:
        try:
            if os.path.exists(DAY_STATE_PATH):
                with open(DAY_STATE_PATH, "r", encoding="utf-8") as f: return json.load(f)
        except Exception as e: log("day_state_load_err", err=str(e))
        return {}
    def save_day_state(data: Dict[str, Any]):
        try:
            mkdirs(os.path.dirname(DAY_STATE_PATH))
            tmp = DAY_STATE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f: json.dump(data, f)
            os.replace(tmp, DAY_STATE_PATH)
        except Exception as e: log("day_state_save_err", err=str(e))

    day_state = load_day_state()
    k = session_key_multi(ct_now(), session_cutovers)
    if k not in day_state:
        seed = float(args.acct_base)
        if args.use_ib_pnl and (ib_netliq is not None): seed = float(ib_netliq)
        day_state[k] = {"start_equity": seed, "day_realized": 0.0, "day_peak_realized": 0.0}
        save_day_state(day_state)

    day_realized = float(day_state[k]["day_realized"])
    day_peak_realized = float(day_state[k]["day_peak_realized"])
    start_of_day_equity = float(day_state[k]["start_equity"])
    day_guard_dollars = -float(args.day_guard_pct) * start_of_day_equity

    # Sizing helpers
    def per_contract_risk_dollars() -> float:
        return float(args.risk_ticks) * float(args.tick_size) * float(px_mult)

    def equity_ladder_size(eq: float) -> int:
        steps = 0 if eq < args.acct_base else math.floor((eq - args.acct_base) / max(1.0, args.scale_step))
        return int(clamp(args.start_contracts + steps, 1, args.max_contracts))

    def risk_budget_size(eq: float) -> int:
        risk_budget = float(eq) * float(args.risk_pct)
        pc_risk = per_contract_risk_dollars()
        if pc_risk <= 0: return 1
        return int(clamp(math.floor(risk_budget / pc_risk), 1, args.max_contracts))

    def apply_hwm_stepdown(qty_suggested: int) -> int:
        if not args.hwm_stepdown: return qty_suggested
        dd = max(0.0, equity_hwm - equity)
        if args.hwm_stepdown_dollars <= 0: return qty_suggested
        steps_down = int(math.floor(dd / float(args.hwm_stepdown_dollars)))
        if steps_down <= 0: return qty_suggested
        return max(1, qty_suggested - steps_down)

    def margin_cap_qty(eq_now: float) -> int:
        reserve = max(0.0, float(args.margin_reserve_pct))
        eff_eq = max(0.0, float(eq_now) * (1.0 - reserve))
        per = max(1.0, float(args.margin_per_contract))
        return int(clamp(math.floor(eff_eq / per), 0, args.max_contracts))

    def determine_order_qty(current_net_qty: int) -> int:
        if args.use_ib_pnl and (ib_netliq is not None):
            eq_now = float(ib_netliq)
        else:
            eq_now = float(equity)
        if args.static_size:
            final_qty = int(max(1, round(float(args.qty))))
            return int(max(0, min(final_qty, args.max_contracts - abs(current_net_qty))))
        eq_size  = equity_ladder_size(eq_now)
        rb_size  = risk_budget_size(eq_now)
        base_qty = int(clamp(min(eq_size, rb_size), 1, args.max_contracts))
        stepdown_qty = apply_hwm_stepdown(base_qty)
        mcap_total = margin_cap_qty(eq_now)
        desired_total = min(int(clamp(stepdown_qty, 0, args.max_contracts)),
                            int(clamp(mcap_total, 0, args.max_contracts)))
        if abs(current_net_qty) >= desired_total: return 0
        addable = desired_total - abs(int(current_net_qty))
        final_qty = int(max(0, addable))
        if (abs(current_net_qty) + final_qty) > args.max_contracts:
            final_qty = max(0, args.max_contracts - abs(current_net_qty))
        return int(final_qty)

    # Entry / place
    def place_bracket(go_long: bool, last_price: float, last_bar_ts_local, net_qty_now: int):
        nonlocal entry_price, current_arm, last_entry_bar_ts, current_param_arm
        if not args.place_orders:
            log("sim_no_place", reason="--place-orders not set"); return
        if has_active_parent_entry(ib, con):
            log("gate_skip", reason="active_parent_entry"); return
        if args.debounce_one_bar and last_entry_bar_ts is not None and last_bar_ts_local == last_entry_bar_ts:
            log("gate_skip", reason="debounce_one_bar"); return

        qty = determine_order_qty(net_qty_now)
        if qty <= 0:
            log("gate_skip", reason="qty_le_0"); return

        # ---- parameter meta-learning override (per-entry) ----
        chosen_params = {}
        if param_learner:
            arm, _ = param_learner.choose(list(param_arms.keys()), sample=(args.learn_mode!="shadow"))
            chosen_params = param_arms.get(arm, {})
            current_param_arm = arm
            risk_ticks_local = int(chosen_params.get("risk_ticks", args.risk_ticks))
            tp_R_local = float(chosen_params.get("tp_R", args.tp_R))
            entry_slip_local = int(chosen_params.get("entry_slippage_ticks", args.entry_slippage_ticks))
        else:
            current_param_arm = None
            risk_ticks_local = args.risk_ticks
            tp_R_local = args.tp_R
            entry_slip_local = args.entry_slippage_ticks

        tick = float(args.tick_size)
        slippage = ticks_to_price_delta(entry_slip_local, tick)
        risk_px = ticks_to_price_delta(risk_ticks_local, tick)
        if go_long:
            entry = round_to_tick(last_price + slippage, tick)
            sl = round_to_tick(entry - risk_px, tick)
            tp = round_to_tick(entry + risk_px * float(tp_R_local), tick)
            action = "BUY"
        else:
            entry = round_to_tick(last_price - slippage, tick)
            sl = round_to_tick(entry + risk_px, tick)
            tp = round_to_tick(entry - risk_px * float(tp_R_local), tick)
            action = "SELL"

        parent = LimitOrder(action=action, totalQuantity=qty, lmtPrice=entry, tif=args.tif, outsideRth=bool(args.outsideRth))
        parent.transmit = False
        parent_trade = ib.placeOrder(con, parent)
        parent_id = parent_trade.order.orderId
        ib.sleep(0.02)

        exit_action = "SELL" if action=="BUY" else "BUY"
        oca = f"OCO-{int(time.time())}"

        stop_loss = StopOrder(action=exit_action, totalQuantity=qty, stopPrice=sl, tif=args.tif, outsideRth=bool(args.outsideRth))
        try: stop_loss.triggerMethod = 2
        except Exception: pass
        stop_loss.parentId = parent_id
        stop_loss.ocaGroup = oca
        stop_loss.transmit = False
        try: stop_loss.ocaType = 1
        except Exception: pass

        take_profit = LimitOrder(action=exit_action, totalQuantity=qty, lmtPrice=tp, tif=args.tif, outsideRth=bool(args.outsideRth))
        take_profit.parentId = parent_id
        take_profit.ocaGroup = oca
        take_profit.transmit = True
        try: take_profit.ocaType = 1
        except Exception: pass

        try:
            ib.placeOrder(con, stop_loss)
            ib.placeOrder(con, take_profit)
            log("bracket_submitted", side=action, qty=qty, entry=entry, stop=sl, tp=tp,
                param_arm=current_param_arm, params=chosen_params or None)
        except Exception as e:
            log("bracket_err", err=str(e)); return

        entry_price = entry
        last_entry_bar_ts = last_bar_ts_local
        risk.last_entry_time = time.time()
        risk.cool_until = ct_now() + dt.timedelta(seconds=int(args.strategy_cooldown_sec))
        risk.trades += 1

    def flatten_market(net_qty: int):
        if not args.place_orders: return
        side = "SELL" if net_qty > 0 else "BUY"
        qty = abs(int(net_qty))
        try:
            mo = MarketOrder(side, qty)
            ib.placeOrder(con, mo)
            log("flatten_market", side=side, qty=qty)
        except Exception as e:
            log("flatten_err", err=str(e))

    # Heartbeat/timing vars
    start_ts = time.time()
    startup_bar_ts = None
    startup_bar_seen = not args.require_new_bar_after_start
    in_caps = False
    tod_blackouts = parse_blackouts(args.tod_blackouts)

    # Preload last RT bar ts (for debounce)
    try:
        hist_last = ib.reqHistoricalData(con, endDateTime='', durationStr='60 S', barSizeSetting='5 secs',
                                         whatToShow='TRADES', useRTH=False, keepUpToDate=False)
        if hist_last:
            startup_bar_ts = getattr(hist_last[-1], "time", None) or getattr(hist_last[-1], "date", None)
    except Exception: pass

    # --- IBKR News Bulletins (optional) ---
    news_kill_until_ref: Optional[dt.datetime] = None
    def _arm_news_kill(reason: str, minutes: float):
        nonlocal news_kill_until_ref
        news_kill_until_ref = ct_now() + dt.timedelta(minutes=max(1.0, float(minutes)))
        log("news_kill_armed", reason=reason, until=str(news_kill_until_ref))
    if args.news_bulletin_listen:
        try:
            def _on_news_bulletin(msgId, newsType, message, exchange):
                try:
                    text = f"{message or ''}".lower()
                    hit_kw = any(k in text for k in news_keywords) if news_keywords else False
                    if int(newsType) == 1 or hit_kw:
                        _arm_news_kill(reason=f"bulletin(type={newsType})", minutes=args.news_kill_minutes)
                except Exception as e:
                    log("news_bulletin_err", err=str(e))
            ib.newsBulletinEvent += _on_news_bulletin
            ib.reqNewsBulletins(True)
            log("news_bulletins_on", msg="Subscribed to IBKR news bulletins")
        except Exception as e:
            log("news_bulletins_unavailable", err=str(e))

    # MAIN LOOP
    try:
        while True:
            now = ct_now()

            # Startup delay for stability
            if (time.time() - start_ts) < args.startup_delay_sec:
                hb_update(idle_reason="booting", bars=len(C))
                ib.sleep(0.5); continue

            # Session cutover logging / segmentation
            cutover_hit = reset_due_multi(now, session_cutovers, last_reset_marks)
            if cutover_hit:
                risk.reset()
                realized_pnl_day = 0.0
                nk = session_key_multi(now, session_cutovers)
                day_state.setdefault(nk, {"start_equity": float(ib_netliq or args.acct_base), "day_realized": 0.0, "day_peak_realized": 0.0})
                keys = sorted(day_state.keys())
                for old in keys[:-8]:
                    try: del day_state[old]
                    except: pass
                save_day_state(day_state)
                k = nk
                start_of_day_equity = float(day_state[k]["start_equity"])
                day_guard_dollars = -float(args.day_guard_pct) * start_of_day_equity
                day_realized = float(day_state[k]["day_realized"])
                day_peak_realized = float(day_state[k]["day_peak_realized"])
                in_caps = False
                if args.vwap_reset_on_session: vwap.reset()
                log("daily_reset", cutover=cutover_hit)

            # ---- News kill checks ----
            file_on, file_until = read_news_file_flag(args.news_file_kill)
            if file_on:
                if (file_until is None) or (now <= file_until):
                    tgt_until = file_until or (now + dt.timedelta(minutes=args.news_kill_minutes))
                    news_kill_until = max(news_kill_until or now, tgt_until)
            # TOD news blackouts
            if in_tod_blackout(now, news_blackouts):
                news_kill_until = max(news_kill_until or now, now + dt.timedelta(minutes=1))
            # Merge with bulletin kill
            if news_kill_until_ref:
                news_kill_until = max(news_kill_until or now, news_kill_until_ref)

            news_kill_active = (news_kill_until is not None) and (now <= news_kill_until)
            if news_kill_active:
                try:
                    if args.news_cancel_only or args.news_flatten_on_kill:
                        c = 0
                        for t in ib.openTrades():
                            st = (getattr(t.orderStatus, "status", "") or "").strip()
                            if st in ACTIVE_STATUSES:
                                safe_cancel(ib, t, note="[news_kill]")
                                c += 1
                        if c: log("news_kill_cancel_working", count=c)
                    if args.news_flatten_on_kill:
                        qty_now, _ = ib_position_truth(ib, con)
                        if qty_now != 0:
                            flatten_market(qty_now)
                except Exception as e:
                    log("news_kill_action_err", err=str(e))
                hb_update(news_kill=True)
            else:
                hb_update(news_kill=False)

            # RT / polling ingestion
            try:
                if rt and len(rt) > 0:
                    b = rt[-1]; ts = getattr(b, "time", None) or getattr(b, "date", None)
                    if last_bar_ts is None or (ts and ts != last_bar_ts):
                        H.append(b.high); L.append(b.low); C.append(b.close)
                        vol = getattr(b, "volume", None); V.append(vol if vol is not None else 0.0)
                        vwap.update_bar(b.close, vol)
                        last_bar_ts = ts; _rt_last_seen = time.time()
                        hb_update(bars=len(C))
                        if startup_bar_ts is not None and ts != startup_bar_ts: startup_bar_seen = True
            except Exception: pass

            # --- Determine RT freshness / starvation ---
            _now = time.time()
            rt_seen_age = ((_now - _rt_last_seen) if _rt_last_seen else None)

            # If we subscribed but have never seen a bar for >8s, treat as starved
            rt_starved = (rt is not None) and (_rt_last_seen is None) and ((_now - start_ts) > 8.0)

            # Fresh if we've seen a bar within staleness window
            rt_fresh = (_rt_last_seen is not None) and (rt_seen_age is not None) and (rt_seen_age <= max(5, int(args.rt_staleness_sec)))

            # Heartbeat update about RT state
            if rt is not None:
                hb_update(rt_enabled=True,
                          rt_status=("ok" if rt_fresh else ("starved" if rt_starved else "stale")),
                          rt_age_sec=(rt_seen_age if _rt_last_seen else None),
                          rt_queue_len=(len(rt) if rt else 0))
            else:
                hb_update(rt_enabled=False, rt_status="disabled", rt_age_sec=None, rt_queue_len=0)

            # --- If TRADES stream is starved, try resubscribing as MIDPOINT once ---
            if rt_starved and rt_mode == "TRADES":
                try:
                    try:
                        ib.cancelRealTimeBars(rt)
                    except Exception:
                        pass
                    ib.sleep(0.2)
                    rt = ib.reqRealTimeBars(con, 5, 'MIDPOINT', False)
                    rt_mode = "MIDPOINT"
                    _rt_last_seen = None
                    log("rt_resubscribe", what="MIDPOINT")
                except Exception as e:
                    log("rt_resubscribe_err", err=str(e))

            # --- Polling fallback: also poll if RT is starved or stale ---
            poll_active = bool(poll_enabled) or rt_starved or (not rt_fresh)
            if poll_active and ((_now - _last_poll) >= poll_iv):
                _last_poll = _now
                try:
                    hist = ib.reqHistoricalData(con, endDateTime='', durationStr='60 S', barSizeSetting='5 secs',
                                                whatToShow='TRADES', useRTH=False, keepUpToDate=False)
                    if not hist and rt_mode == "MIDPOINT":
                        hist = ib.reqHistoricalData(con, endDateTime='', durationStr='60 S', barSizeSetting='5 secs',
                                                    whatToShow='MIDPOINT', useRTH=False, keepUpToDate=False)
                    if hist:
                        last = hist[-1]; ts = getattr(last, "time", None) or getattr(last, "date", None)
                        if last_bar_ts is None or (ts and ts > last_bar_ts):
                            H.append(last.high); L.append(last.low); C.append(last.close)
                            vol = getattr(last, "volume", None); V.append(vol if vol is not None else 0.0)
                            vwap.update_bar(last.close, vol)
                            last_bar_ts = ts
                            hb_update(bars=len(C))
                            if startup_bar_ts is not None and ts != startup_bar_ts: startup_bar_seen = True
                            log("poll_bar", time=str(ts), close=last.close, total=len(C), mode=rt_mode)
                    else:
                        log("poll_bar_empty", note="historical returned 0 bars", mode=rt_mode)
                except Exception as e:
                    log("poll_bar_err", err=str(e))

            # Position truth
            net_qty, avg_cost_raw = ib_position_truth(ib, con)
            hb_update(net_qty=net_qty)

            # Execute deferred sibling/orphan actions
            if sibling_cancel_needed:
                sibling_cancel_needed = False
                try:
                    for t in ib.openTrades():
                        if (t.orderStatus.status or "").strip() in ("Filled","PartiallyFilled"):
                            cancel_siblings_for_trade(ib, t, con)
                except Exception as e:
                    log("sibling_cancel_deferred_err", err=str(e))
            if orphan_sweep_needed:
                orphan_sweep_needed = False
                try:
                    reconcile_orphans(ib, ib_acct or "", con)
                except Exception as e:
                    log("orphan_sweep_deferred_err", err=str(e))

            # Trade-cycle transitions
            if prev_net_qty == 0 and net_qty != 0:
                in_trade_cycle = True
                cycle_commission = 0.0
                cycle_entry_qty = abs(net_qty)
                entry_price = (avg_cost_raw / px_mult) if (avg_cost_raw is not None and px_mult > 0) else (C[-1] if C else None)
                cycle_entry_time = now
                age_forced_flat_done = False

            if prev_net_qty != 0 and net_qty == 0 and entry_price is not None:
                exit_px = last_exec_price if last_exec_price is not None else (C[-1] if C else entry_price)
                signed = 1 if prev_net_qty > 0 else -1
                pc_risk = float(args.risk_ticks) * float(args.tick_size) * float(px_mult)
                risk_dollars_total = pc_risk * max(1, cycle_entry_qty)
                pnl_dollars = (exit_px - entry_price) * px_mult * signed * max(1, cycle_entry_qty) - cycle_commission
                reward_R = (pnl_dollars / risk_dollars_total) if risk_dollars_total > 0 else 0.0

                realized_pnl_total += pnl_dollars
                realized_pnl_day += pnl_dollars
                if ib_netliq is not None and args.use_ib_pnl:
                    equity = float(ib_netliq)
                else:
                    equity = float(args.acct_base) + realized_pnl_total
                equity_hwm = max(equity_hwm, equity)

                risk.day_R += reward_R
                risk.consec_losses = (risk.consec_losses + 1) if (reward_R < 0) else 0
                if risk.consec_losses >= args.max_consec_losses: risk.halted = True
                if (reward_R < 0) and not risk.halted:
                    base = int(args.strategy_cooldown_sec)
                    mult = 2.0 if risk.consec_losses >= 2 else 1.0
                    tgt = ct_now() + dt.timedelta(seconds=int(base*mult))
                    if (risk.cool_until is None) or (tgt > risk.cool_until):
                        risk.cool_until = tgt
                    log("graded_cooldown", base=base, mult=mult, consec=risk.consec_losses, until=str(risk.cool_until))

                week_R += reward_R
                day_realized += pnl_dollars
                day_peak_realized = max(day_peak_realized, day_realized)
                day_state[k]["day_realized"] = day_realized
                day_state[k]["day_peak_realized"] = day_peak_realized
                save_day_state(day_state)

                try:
                    if args.learn_mode in ("advisory", "control"):
                        learner.update(current_arm or "trend", reward_R)
                        save_thompson(strat_path, learner)
                except Exception as e:
                    log("learn_update_err", err=str(e))

                try:
                    if param_learner and current_param_arm:
                        param_learner.update(current_param_arm, reward_R)
                        save_thompson(param_path, param_learner)
                except Exception as e:
                    log("param_learn_update_err", err=str(e))

                log("flat_cycle", pnl=round(pnl_dollars,2), R=round(reward_R,3),
                    comm=round(cycle_commission,2), qty=max(1, cycle_entry_qty),
                    dayR=round(risk.day_R,3), consec=risk.consec_losses, equity=round(equity,2),
                    param_arm=current_param_arm)

                in_trade_cycle = False
                entry_price = None
                current_arm = None
                current_param_arm = None
                cycle_entry_qty = 0
                cycle_commission = 0.0
                cycle_entry_time = None
                age_forced_flat_done = False

            prev_net_qty = net_qty

            # IB PnL overwrite
            if args.use_ib_pnl:
                if ib_daily_pnl is not None:
                    day_realized = float(ib_daily_pnl)
                    day_peak_realized = max(day_peak_realized, day_realized)
                    day_state[k]["day_realized"] = day_realized
                    day_state[k]["day_peak_realized"] = day_peak_realized
                    save_day_state(day_state)
                if ib_netliq is not None:
                    equity = float(ib_netliq); equity_hwm = max(equity_hwm, equity)

            # OCO rescue (ensure both stop+tp exist and correct side)
            last_px = C[-1] if C else None
            try:
                if net_qty != 0 and args.place_orders:
                    trades = list_open_orders_for_contract(ib, con)
                    exit_action = "SELL" if net_qty > 0 else "BUY"
                    has_stop = any((t.order.orderType or "").upper().startswith("STP") and (t.order.action or "").upper()==exit_action for t in trades)
                    has_tp   = any((t.order.orderType or "").upper()=="LMT" and (t.order.action or "").upper()==exit_action for t in trades)
                    if (not has_stop) or (not has_tp):
                        for t in trades:
                            ot = (t.order.orderType or "").upper(); act = (t.order.action or "").upper()
                            if ot in {"LMT","STP","STP LMT"} and act != exit_action:
                                safe_cancel(ib, t, note="[prot wrong-side]")
                            elif ot in {"LMT","STP","STP LMT"} and act == exit_action:
                                safe_cancel(ib, t, note="[prot refresh]")
                        if last_px is not None and not math.isnan(last_px):
                            qty_abs = abs(int(net_qty))
                            tick = float(args.tick_size)
                            risk_px = ticks_to_price_delta(args.risk_ticks, tick)
                            tp_px = risk_px * float(args.tp_R)
                            if net_qty > 0:
                                stop_price = round_to_tick(last_px - risk_px - tick, tick)
                                targ_price = round_to_tick(last_px + tp_px + tick, tick)
                            else:
                                stop_price = round_to_tick(last_px + risk_px + tick, tick)
                                targ_price = round_to_tick(last_px - tp_px - tick, tick)
                            oca = f"OCO-PROT-{int(time.time())}"
                            stp = StopOrder(action=exit_action, totalQuantity=qty_abs, stopPrice=stop_price, tif=args.tif, outsideRth=bool(args.outsideRth))
                            try: stp.triggerMethod = 2
                            except Exception: pass
                            stp.ocaGroup = oca; stp.transmit = False
                            try: stp.ocaType = 1
                            except Exception: pass
                            lmt = LimitOrder(action=exit_action, totalQuantity=qty_abs, lmtPrice=targ_price, tif=args.tif, outsideRth=bool(args.outsideRth))
                            lmt.ocaGroup = oca; lmt.transmit = True
                            try: lmt.ocaType = 1
                            except Exception: pass
                            ib.placeOrder(con, stp); ib.placeOrder(con, lmt)
                            log("oco_rebuilt", side=exit_action, stp=stop_price, lmt=targ_price, qty=qty_abs)
            except Exception as e:
                log("oco_rescue_err", err=str(e))

            # Position-age cap
            if (net_qty != 0) and (entry_price is not None) and (cycle_entry_time is not None) and (not args.pos_age_minR is None):
                elapsed = (now - cycle_entry_time).total_seconds()
                if elapsed >= max(1, int(args.pos_age_cap_sec)) and not age_forced_flat_done:
                    if last_px is not None and not math.isnan(last_px):
                        side = 1 if net_qty > 0 else -1
                        pc_risk = float(args.risk_ticks) * float(args.tick_size) * float(px_mult)
                        uR = 0.0
                        if pc_risk > 0:
                            uR = ((last_px - entry_price) * px_mult * side * max(1,cycle_entry_qty)) / (pc_risk * max(1,cycle_entry_qty))
                        if uR < float(args.pos_age_minR):
                            flatten_market(net_qty)
                            age_forced_flat_done = True
                            risk.cool_until = ct_now() + dt.timedelta(seconds=int(args.strategy_cooldown_sec))
                            log("pos_age_forced_flat", uR=round(uR,3), elapsed=int(elapsed))

            # Weekly cap gating
            if week_R <= weekly_cap_R:
                week_halted = True

            # Peak DD guard + caps
            caps_reasons = []
            if risk.day_R <= -abs(args.day_loss_cap_R): caps_reasons.append("dayR_cap")
            if risk.trades >= args.max_trades_per_day: caps_reasons.append("max_trades")
            if risk.halted: caps_reasons.append("risk_halted")
            if day_realized <= day_guard_dollars: caps_reasons.append("minus_pct_guard")
            if args.peak_dd_guard_pct > 0 and (day_peak_realized >= float(args.peak_dd_min_profit)):
                if day_realized <= day_peak_realized * (1.0 - float(args.peak_dd_guard_pct)):
                    caps_reasons.append("peak_dd_guard")
            if week_halted: caps_reasons.append("weekly_cap_R")
            if news_kill_active: caps_reasons.append("news_kill")

            # Determine state
            if caps_reasons:
                state = "caps"
            elif (args.require_rt_before_trading and not rt_fresh):
                state = "wait_rt"
            else:
                state = "active" if within_session(now, args.trade_start_ct, args.trade_end_ct) else "sleep"
            if in_tod_blackout(now, parse_blackouts(args.tod_blackouts)):
                state = "sleep"

            # Heartbeat
            if state == "caps" and not in_caps:
                in_caps = True
                log("caps_on", reasons=caps_reasons,
                    day_realized=round(day_realized,2), day_peak=round(day_peak_realized,2), dayR=round(risk.day_R,3))
            if state != "caps" and in_caps:
                in_caps = False

            if state == "caps" and net_qty == 0:
                try:
                    c = 0
                    for t in ib.openTrades():
                        st = (getattr(t.orderStatus, "status", "") or "").strip()
                        if st in ACTIVE_STATUSES:
                            safe_cancel(ib, t, note="[cap_sweep]")
                            c += 1
                    if c: log("cap_sweep_all_orders", count=c)
                except Exception as e:
                    log("cap_sweep_err", err=str(e))

            # Idle reason for HB
            if state == "caps":
                idle_reason = "capped:" + ",".join(caps_reasons)
            elif state == "wait_rt":
                idle_reason = "waiting_for_rt_fresh"
            elif state == "sleep":
                idle_reason = "outside_trade_window"
            else:
                if has_active_parent_entry(ib, con):
                    idle_reason = "parent_entry_working"
                elif not risk.can_trade(now, int(args.min_seconds_between_entries)):
                    if risk.cool_until and now < risk.cool_until:
                        idle_reason = "gated:cooldown"
                    elif risk.trades >= args.max_trades_per_day:
                        idle_reason = "gated:max_trades"
                    elif risk.day_R <= -abs(args.day_loss_cap_R):
                        idle_reason = "gated:dayR_cap"
                    elif risk.consec_losses >= args.max_consec_losses:
                        idle_reason = "gated:consec_losses"
                    else:
                        idle_reason = "gated:min_gap_between_entries"
                elif args.require_new_bar_after_start and not startup_bar_seen:
                    idle_reason = "waiting_for_first_new_bar"
                elif len(C) < 60:
                    idle_reason = "waiting_for_bars"
                else:
                    idle_reason = "active_waiting"

            hb_update(state=state,
                      idle_reason=idle_reason,
                      in_session_window=within_session(now, args.trade_start_ct, args.trade_end_ct),
                      caps=caps_reasons,
                      dayR=round(risk.day_R, 3),
                      trades_today=int(risk.trades),
                      cool_until=(str(risk.cool_until) if risk.cool_until else None))

            # ENTRY LOGIC (simple placeholder)
            if state == "active" and len(C) >= 60 and not has_active_parent_entry(ib, con) and net_qty == 0:
                close = C[-1]
                _atrv = atr(H, L, C, 14)
                _atrp = (_atrv / close) if (not math.isnan(_atrv) and close>0) else float("nan")
                fast = ema(C[-20:], 20); slow = ema(C[-50:], 50)
                is_trend = (fast > slow) and (not math.isnan(_atrp)) and _atrp >= args.gate_atrp
                is_breakout = (len(C) >= 30) and (close >= max(C[-20:]) or close <= min(C[-20:]))

                # Optional BBBW gate (disabled by default via --gate-bbbw 0)
                bbbw_ok = True
                if args.gate_bbbw > 0:
                    win = C[-20:]
                    m = sum(win)/20.0
                    var = sum((x-m)*(x-m) for x in win)/20.0
                    sd = math.sqrt(max(0.0, var))
                    if m > 0:
                        bbbw = (2*2.0*sd)/m
                        bbbw_ok = bbbw >= float(args.gate_bbbw)
                    else:
                        bbbw_ok = False
                    if not bbbw_ok:
                        hb_update(idle_reason="gated:bbbw_low")

                cand = []
                if "trend" in arms_enabled and is_trend and bbbw_ok: cand.append("trend")
                if "breakout" in arms_enabled and is_breakout and bbbw_ok: cand.append("breakout")
                if cand:
                    if len(cand) == 1:
                        chosen = cand[0]
                    else:
                        chosen, probs = learner.choose(cand, sample=(args.learn_mode!="shadow"))
                        if args.learn_mode in ("shadow","advisory"):
                            log("learn_decision", cand=cand, probs={k: round(v,3) for k,v in probs.items()}, chosen=chosen)
                    go_long = True if (chosen == "trend" and fast >= slow) else (close >= max(C[-20:]))
                    if args.learn_mode != "shadow":
                        place_bracket(go_long, close, last_bar_ts, net_qty)
                        current_arm = chosen

            # Sweep orphans regularly
            reconcile_orphans(ib, ib_acct or "", con)

            ib.sleep(1.0)

    except KeyboardInterrupt:
        print("INFO [CTRL-C] Shutting down...")
    finally:
        try:
            day_state[k]["day_realized"] = day_realized
            day_state[k]["day_peak_realized"] = day_peak_realized
            save_day_state(day_state)
        except Exception: pass
        # Final learner save
        try:
            save_thompson(strat_path, learner)
            if param_learner: save_thompson(param_path, param_learner)
        except Exception: pass
        try:
            sent = getattr(getattr(ib, "client", None), "bytesSent", 0) or 0
            recv = getattr(getattr(ib, "client", None), "bytesReceived", 0) or 0
            print(f"Disconnecting from {args.host}:{args.port}, {sent/1024:.0f} kB sent, {recv/1024:.0f} kB received.")
        except Exception: pass
        try: ib.disconnect(); print("Disconnected.")
        except Exception: pass

if __name__ == "__main__":
    main()

