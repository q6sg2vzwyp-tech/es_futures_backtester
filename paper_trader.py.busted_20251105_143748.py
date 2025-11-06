#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, time, json, math, random, argparse, datetime as dt, re, traceback
from typing import Optional, List, Dict, Any, Tuple

# 3rd party
from ib_insync import IB, Future, Contract, LimitOrder, StopOrder, Trade, MarketOrder

# ============== Utilities & Logging ==============

def ct_now() -> dt.datetime:
    # assumes local machine clock is CT
    return dt.datetime.now()

def utc_now_str() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def parse_hhmm(s: str) -> dt.time:
    hh, mm = s.split(":")
    return dt.time(int(hh), int(mm))

def within_session(now: dt.datetime, start_ct: str, end_ct: str) -> bool:
    t = now.time()
    a = parse_hhmm(start_ct)
    b = parse_hhmm(end_ct)
    if a <= b:
        # Normal same-day window
        return a <= t <= b
    # Overnight window (wraps past midnight)
    return (t >= a) or (t <= b)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))
def ema(values: List[float], span: int) -> float:
    if not values:
        return float("nan")
    k = 2.0 / (span + 1.0)
    s = values[0]
    for v in values[1:]:
        s = v * k + s * (1.0 - k)
    return s
def atr(h: List[float], l: List[float], c: List[float], n: int = 14) -> float:
    if len(c) < n + 1:
        return float("nan")
    trs: List[float] = []
    for i in range(1, len(c)):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l[i] - c[i - 1])
        trs.append(max(hl, hc, lc))
    if len(trs) < n:
        return float("nan")
    return ema(trs[-n:], n)
def stddev(vals: List[float]) -> float:
    n = len(vals)
    if n == 0:
        return float("nan")
    m = sum(vals) / n
    v = sum((x - m) * (x - m) for x in vals) / n
    return math.sqrt(v)

def bbbw(hl2: List[float], n: int = 20, k: float = 2.0) -> Optional[float]:
    if len(hl2) < n: return None
    wins = hl2[-n:]; m = sum(wins)/n
#     v = sum((x-m)**2 for x in wins)/n; sd = math.sqrt(v)
#     mid = m; upper = mid + k*sd; lower = mid - k*sd
#     bw = (upper - lower) / (mid if mid != 0 else 1.0)
#     return abs(bw)

def duration_fix(s: str) -> str:
    pass
#     s = s.strip().replace('"', '')
    if s.upper().endswith('D') and ' ' not in s:
        num = s[:-1]
        if num.isdigit(): return f"{num} D"
#     return s

def ticks_to_price_delta(ticks: int, tick_size: float) -> float:
    pass
#     return float(ticks) * float(tick_size)

def round_to_tick(p: float, tick: float) -> float:
    pass
#     return round(p / tick) * tick if tick > 0 else p

def bar_ts(b):
    pass
#     t = getattr(b, "time", None)
    if t is None:
        pass
#         t = getattr(b, "date", None)
#     return t

def mkdirs(p: str):
    if p: os.makedirs(p, exist_ok=True)

# Simple JSON logger
def log(event: str, **fields):
    payload = {"ts": utc_now_str(), "evt": event}
#     payload.update(fields)
    print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), flush=True)

# ============== Session helpers (AM/PM) ==============

def parse_ct_list(spec: str) -> List[dt.time]:
    spec = (spec or "").strip()
    if not spec:
        return [parse_hhmm("16:10")]
    out: List[dt.time] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(parse_hhmm(chunk))
        except Exception:
            # ignore bad tokens like "foo"
            pass
    out = sorted(list({t for t in out}))
    return out or [parse_hhmm("16:10")]

def session_key_multi(now: dt.datetime, reset_times: List[dt.time]) -> str:
    """Return 'YYYY-mm-dd-S#' where S# is the current segment index determined by reset_times.
    If no cutover has occurred yet today, attribute to yesterday's last segment."""
    t = now.time()
    idx_today = -1
    for i, ct in enumerate(reset_times):
        if t >= ct:
            idx_today = i
        else:
            break
    if idx_today >= 0:
        base_date = now.date(); seg = idx_today
    else:
        base_date = (now - dt.timedelta(days=1)).date(); seg = len(reset_times) - 1
    return f"{base_date.strftime('%Y-%m-%d')}-S{seg}"

def reset_due_multi(now: dt.datetime, reset_times: List[dt.time], last_reset_marks: Dict[str, str]) -> Optional[str]:
    """Fire once per cutover per calendar day.
    Returns 'YYYY-mm-dd#HH:MM' on a new cutover, else None."""
    today = now.date().strftime('%Y-%m-%d')
    for ct in reset_times:
        label = ct.strftime('%H:%M')
        if last_reset_marks.get(label) != today and now.time() >= ct:
            last_reset_marks[label] = today
            return f'{today}#{label}'
    return None

# ============== Safety: Paper-only ==============

def ensure_paper_only(ib: IB, args):
    if getattr(args, "allow_live", False): return
    try:
        pass
#         accts = ib.managedAccounts(); acct = accts[0] if accts else None
    except Exception:
        pass
#         acct = None
#     bad_port = (getattr(args, 'port', None) != 7497)
#     bad_acct = (acct is not None and not acct.upper().startswith("DU"))
    if bad_port or bad_acct:
        print(f"[SAFE] Paper-only: refusing to trade (port={getattr(args,'port',None)}, account={acct}).", file=sys.stderr)
#         sys.exit(2)

# ============== Contracts & multiplier ==============

def parse_yyyymmdd(s: str) -> Optional[dt.date]:
    try:
        if not s: return None
        if len(s) == 8: return dt.datetime.strptime(s, "%Y%m%d").date()
        if len(s) == 6: return dt.datetime.strptime(s, "%Y%m").date().replace(day=1)
    except Exception:
        pass
#         return None
#     return None

def qualify_local_symbol(ib: IB, local_symbol: str, exchange="CME"):
    pass
#     cds = ib.reqContractDetails(Future(localSymbol=local_symbol, exchange=exchange))
    if not cds: raise RuntimeError(f"Local symbol {local_symbol} not found on {exchange}")
#     con = cds[0].contract
#     ib.qualifyContracts(con)
#     return con, cds[0]

def mk_contract(ib: IB, args) -> Contract:
    if getattr(args, "local_symbol", None):
        pass
#         con, cd = qualify_local_symbol(ib, args.local_symbol, "CME")
#         print(f"[CONTRACT] Using localSymbol={con.localSymbol} conId={con.conId} expiry={con.lastTradeDateOrContractMonth}")
#         return con
    try:
        pass
#         cds = ib.reqContractDetails(Future(symbol=args.symbol, exchange="CME"))
#         today = dt.date.today(); live = []
        for cd in cds:
            pass
#             last = parse_yyyymmdd(cd.contract.lastTradeDateOrContractMonth or "")
            if last and last >= today:
                pass
#                 live.append((last, cd.contract))
        if live:
            live.sort(key=lambda x: x[0])
#             con = live[0][1]; ib.qualifyContracts(con)
#             print(f"[CONTRACT] Auto-picked {con.localSymbol} conId={con.conId} expiry={con.lastTradeDateOrContractMonth}")
#             return con
        if cds:
            cds.sort(key=lambda cd: parse_yyyymmdd(cd.contract.lastTradeDateOrContractMonth or "") or dt.date.min, reverse=True)
#             con = cds[0].contract; ib.qualifyContracts(con)
#             print(f"[CONTRACT] Fallback-picked {con.localSymbol} conId={con.conId} expiry={con.lastTradeDateOrContractMonth}")
#             return con
    except Exception as e:
        print(f"[CONTRACT] Auto-pick failed, falling back to generic: {e}")
#     con = Future(symbol=args.symbol, exchange="CME", currency="USD")
#     ib.qualifyContracts(con)
#     return con

def contract_multiplier(ib: IB, con: Contract) -> float:
    try:
        pass
#         cds = ib.reqContractDetails(con)
#         mul = cds[0].contract.multiplier
#         m = float(mul) if mul is not None else 1.0
#         return m if m > 0 else 1.0
    except Exception:
        pass
#         return 1.0

# ============== Safe orderId ==============

def next_order_id(ib: IB) -> int:
    try:
        pass
#         return ib.client.getReqId()
    except Exception:
        pass
#     nid = getattr(ib.client, "_nextValidId", None)
    if isinstance(nid, int) and nid > 0:
        try: ib.client._nextValidId += 1
        except Exception: pass
#         return int(nid)
    try:
        pass
#         ib.reqIds(1); ib.sleep(0.1)
#         nid = getattr(ib.client, "_nextValidId", None)
        if isinstance(nid, int) and nid > 0:
            try: ib.client._nextValidId += 1
            except Exception: pass
#             return int(nid)
    except Exception:
        pass
#     return int(time.time() * 1000) % 2147483000

# ============== Orders ==============

def make_bracket(parent_orderId: int, action: str, qty: float, entry: float, stop: float, target: float,
                 tif: str, outsideRth: bool):
#     parent = LimitOrder(action=action, totalQuantity=qty, lmtPrice=entry, tif=tif, outsideRth=outsideRth)
#     parent.orderId = parent_orderId; parent.transmit = False
#     exit_action = "SELL" if action.upper() == "BUY" else "BUY"
#     oca = f"OCO-{int(time.time())}"
#     stop_loss = StopOrder(action=exit_action, totalQuantity=qty, stopPrice=stop, tif=tif, outsideRth=outsideRth)
    try:
        stop_loss.triggerMethod = 2  # robust for CME
    except Exception:
        pass
#     stop_loss.parentId = parent.orderId; stop_loss.ocaGroup = oca; stop_loss.transmit = False
#     take_profit = LimitOrder(action=exit_action, totalQuantity=qty, lmtPrice=target, tif=tif, outsideRth=outsideRth)
#     take_profit.parentId = parent.orderId; take_profit.ocaGroup = oca; take_profit.transmit = True
#     return [parent, stop_loss, take_profit]

# ============== CLI ==============

def build_argparser():
    pass
#     ap = argparse.ArgumentParser(description="ES Paper Trader (Session-aware + Learning + Rails + Governance + News)")
#     ap.add_argument("--host", default="127.0.0.1")
#     ap.add_argument("--port", type=int, default=7497)
#     ap.add_argument("--clientId", type=int, default=111)
#     ap.add_argument("--symbol", default="ES")
#     ap.add_argument("--local-symbol", dest="local_symbol", default="")
#     ap.add_argument("--auto-front-month", action="store_true")

    # Sizing & Risk
#     ap.add_argument("--acct-base", type=float, default=30000.0)
#     ap.add_argument("--risk-pct", type=float, default=0.01)
#     ap.add_argument("--scale-step", type=float, default=10000.0)
#     ap.add_argument("--start-contracts", type=int, default=2)
#     ap.add_argument("--max-contracts", type=int, default=6)
#     ap.add_argument("--static-size", action="store_true")
#     ap.add_argument("--qty", type=float, default=2.0)
#     ap.add_argument("--risk-ticks", type=int, default=12)
#     ap.add_argument("--tick-size", type=float, default=0.25)
#     ap.add_argument("--tp-R", type=float, default=1.0)

    # Margin awareness (NEW)
#     ap.add_argument("--margin-per-contract", type=float, default=13200.0,
#                     help="Estimated margin per ES contract (maintenance).")
#     ap.add_argument("--margin-reserve-pct", type=float, default=0.10,
#                     help="Keep this fraction of equity unallocated (safety buffer).")

    # Strategy gates
#     ap.add_argument("--enable-arms", default="trend,breakout")
#     ap.add_argument("--gate-adx", type=float, default=19.0)
#     ap.add_argument("--gate-bbbw", type=float, default=0.0002)
#     ap.add_argument("--gate-atrp", type=float, default=0.000055)

    # Anti-burst & day/session rails
#     ap.add_argument("--min-seconds-between-entries", type=int, default=20)
#     ap.add_argument("--max-trades-per-day", type=int, default=12)
#     ap.add_argument("--day-loss-cap-R", type=float, default=3.0)
#     ap.add_argument("--max-consec-losses", type=int, default=3)
#     ap.add_argument("--strategy-cooldown-sec", type=int, default=150)

    # Risk governance extras
#     ap.add_argument("--loss-graded-cooldown", action="store_true")
#     ap.add_argument("--loss-cooldown-mult2", type=float, default=2.0)
#     ap.add_argument("--enable-weekly-cap", action="store_true")
#     ap.add_argument("--pos-age-cap-sec", type=int, default=1200)
#     ap.add_argument("--pos-age-minR", type=float, default=0.5)
#     ap.add_argument("--disable-pos-age-cap", action="store_true")
#     ap.add_argument("--hwm-stepdown", action="store_true")
#     ap.add_argument("--hwm-stepdown-dollars", type=float, default=5000.0)
    ap.add_argument("--cancel-exits-on-flat", action="store_true")  # retained for compatibility

    # Breaking News Guard
#     ap.add_argument("--enable-breaking-news-guard", action="store_true")
#     ap.add_argument("--news-keywords", default="BREAKING,URGENT,FOMC,rate hike,rate cut,nonfarm payrolls,CPI,PPI,ISM,PMI,halt,circuit breaker,terror,war,missile,earthquake,explosion,bank failure,shutdown,emergency")
#     ap.add_argument("--news-pause-sec", type=int, default=900)
#     ap.add_argument("--news-log-body", action="store_true")

    # Trading window (24/5) + blackouts
    ap.add_argument("--trade-start-ct", default="00:00")
    ap.add_argument("--trade-end-ct", default="23:59")
#     ap.add_argument("--tod-blackouts", default="")

    # Order behavior
#     ap.add_argument("--entry-slippage-ticks", type=int, default=2)
    ap.add_argument("--atomic-bracket", action="store_true")  # ignored; warn
#     ap.add_argument("--place-orders", action="store_true")
#     ap.add_argument("--tif", default="GTC")
#     ap.add_argument("--outsideRth", action="store_true")
#     ap.add_argument("--require-new-bar-after-start", action="store_true")
#     ap.add_argument("--startup-delay-sec", type=int, default=10)
#     ap.add_argument("--debounce-one-bar", action="store_true")

    # Session resets (AM/PM default)
    ap.add_argument("--session-reset-cts", default="08:30,17:00",
                    help="Comma CT times for soft resets each day (e.g. '08:30,17:00'). Overrides --daily-reset-ct.")
    ap.add_argument("--daily-reset-ct", default="16:10")  # legacy single reset

    # Reset behaviors at session cutover (NEW)
#     ap.add_argument("--reset-clear-learn", action="store_true",
#                     help="Reset learners/meta at each session cutover.")
#     ap.add_argument("--reset-cancel-working", action="store_true",
#                     help="Cancel active parent entry orders at session cutover.")
#     ap.add_argument("--reset-cancel-exits", action="store_true",
#                     help="Cancel active exit (OCO) child orders at session cutover.")

    # Connectivity & data
#     ap.add_argument("--duration", default="1 D")
#     ap.add_argument("--connect-timeout-sec", type=int, default=300)
#     ap.add_argument("--timeout-sec", type=int, default=300)
#     ap.add_argument("--connect-attempts", type=int, default=24)
#     ap.add_argument("--force-delayed", action="store_true")
#     ap.add_argument("--poll-hist-when-no-rt", action="store_true")
#     ap.add_argument("--poll-interval-sec", type=int, default=10)

    # Realtime gating
#     ap.add_argument("--require-rt-before-trading", action="store_true")
#     ap.add_argument("--rt-staleness-sec", type=int, default=45)

    # Learning / bandits
#     ap.add_argument("--bandit", choices=["meta","linucb","ts","ucb_tuned","exp3","fixed"], default="meta")
#     ap.add_argument("--bandit-state", default=".\\data\\learn\\bandit_state.json")
#     ap.add_argument("--learn-log", action="store_true")
#     ap.add_argument("--learn-log-dir", default=".\\logs\\learn")
#     ap.add_argument("--sample-action", action="store_true")
#     ap.add_argument("--decay-half-life-trades", type=float, default=200.0)
#     ap.add_argument("--meta-eta", type=float, default=0.15)
#     ap.add_argument("--switching-cost", type=float, default=0.05)
#     ap.add_argument("--ph-delta", type=float, default=0.005)
#     ap.add_argument("--ph-lambda", type=float, default=0.8)

    # PnL & equity sync from IB (optional)
#     ap.add_argument("--use-ib-pnl", action="store_true",
#                     help="Sync day_realized from IB dailyPnL and equity from NetLiquidation.")
#     ap.add_argument("--peak-dd-guard-pct", type=float, default=0.60,
#                     help="0 disables from-peak guard; else halt when giveback >= pct of day_peak_realized.")

    # NEW: Day guard and peak-DD minimum profit gate
#     ap.add_argument("--day-guard-pct", type=float, default=0.025,
#                     help="Hard day loss guard as fraction of start-of-session equity (e.g., 0.025 = 2.5%).")
#     ap.add_argument("--peak-dd-min-profit", type=float, default=1500.0,
#                     help="Enable the peak drawdown guard only once day_realized >= this profit.")

    # ===== NEW: Short guard rails & VWAP control =====
#     ap.add_argument("--short-guard-vwap-buffer-ticks", type=int, default=4,
#                     help="Buffer above session VWAP (in ticks) where shorts are blocked.")
#     ap.add_argument("--short-guard-min-pullback-ticks", type=int, default=6,
#                     help="Min HL distance between last two swing highs (in ticks) to confirm lower-high.")
#     ap.add_argument("--short-guard-lookback-bars", type=int, default=60,
#                     help="Lookback window (bars) for swing high detection.")

#     ap.add_argument("--short-guard-vwap", action="store_true", default=True,
#                     help="Block shorts near/above session VWAP.")
#     ap.add_argument("--no-short-guard-vwap", dest="short_guard_vwap", action="store_false")

#     ap.add_argument("--short-guard-lower-high", action="store_true", default=True,
#                     help="Require a lower-high pivot before shorting.")
#     ap.add_argument("--no-short-guard-lower-high", dest="short_guard_lower_high", action="store_false")

#     ap.add_argument("--vwap-reset-on-session", action="store_true", default=True,
#                     help="Reset VWAP accumulators at each session cutover.")
#     ap.add_argument("--no-vwap-reset-on-session", dest="vwap_reset_on_session", action="store_false")

    # Misc
#     ap.add_argument("--allow_live", action="store_true")
#     ap.add_argument("-v", "--verbose", action="store_true")
#     return ap

# ============== Day risk / heartbeat ==============

class DayRisk:
    def __init__(self, loss_cap_R: float, max_trades: int, max_consec_losses: int):
        pass
#         self.loss_cap_R = float(loss_cap_R); self.max_trades = int(max_trades)
#         self.max_consec_losses = int(max_consec_losses)
#         self.reset()
    def reset(self):
        pass
#         self.day_R = 0.0; self.trades = 0
        self.cool_until: Optional[dt.datetime] = None
#         self.halted = False
        self.last_entry_time: Optional[float] = None
#         self.consec_losses = 0
    def can_trade(self, now: dt.datetime, min_gap_s:int) -> bool:
        if self.halted: return False
        if self.cool_until and now < self.cool_until: return False
        if self.trades >= self.max_trades: return False
        if self.day_R <= -abs(self.loss_cap_R): return False
        if self.consec_losses >= self.max_consec_losses: return False
        if self.last_entry_time and (time.time() - self.last_entry_time) < max(0, min_gap_s): return False
#         return True

# ============== Week cap helpers ==============

def iso_week_id(d: dt.date) -> str:
    pass
#     y, w, _ = d.isocalendar()
    return f"{y}-W{int(w):02d}"

# ============== Time-of-day blackouts ==============

def parse_blackouts(spec: str) -> List[Tuple[dt.time, dt.time]]:
    pass
#     out = []
#     spec = (spec or "").strip()
    if not spec: return out
    for chunk in spec.split(","):
        pass
#         chunk = chunk.strip()
        if not chunk: continue
        try:
            pass
#             a, b = chunk.split("-")
#             out.append((parse_hhmm(a), parse_hhmm(b)))
        except Exception:
            pass
#     return out

def in_any_blackout(now: dt.datetime, windows: List[Tuple[dt.time, dt.time]]) -> bool:
    if not windows: return False
#     t = now.time()
    for a, b in windows:
        if a <= b:
            if a <= t <= b: return True
        else:
            if (t >= a) or (t <= b): return True
#     return False

# ============== Indicators / features ==============

def adx(h: List[float], l: List[float], c: List[float], n: int=14) -> float:
    if len(c) < n+2: return float("nan")
#     dm_pos = []; dm_neg = []; tr = []
    for i in range(1, len(c)):
        pass
#         up = h[i] - h[i-1]; dn = l[i-1] - l[i]
#         dm_pos.append(max(up, 0.0) if up > dn else 0.0)
#         dm_neg.append(max(dn, 0.0) if dn > up else 0.0)
#         tr.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    def rma(vals, n):
        if len(vals) < n: return float("nan")
        alpha = 1.0/n; s = sum(vals[:n]) / n
        for v in vals[n:]: s = s*(1-alpha) + v*alpha
#         return s
#     atrv = rma(tr, n)
    if (atrv == 0.0) or math.isnan(atrv): return float("nan")
#     di_pos = 100 * (rma(dm_pos, n) / atrv)
#     di_neg = 100 * (rma(dm_neg, n) / atrv)
#     dx = 100 * abs(di_pos - di_neg) / max(di_pos + di_neg, 1e-9)
#     return dx

def feature_vector(H: List[float], L: List[float], C: List[float]) -> List[float]:
    pass
#     n = len(C)
    if n < 60: return [0.0]*8
    close = C[-1]; f20 = ema(C[-20:], 20); f50 = ema(C[-50:], 50)
#     slope = (f20 - f50) / (close if close != 0 else 1.0)
    _adx = adx(H, L, C, 14); _atr = atr(H, L, C, 14)
#     atrp = (_atr/close) if (not math.isnan(_atr) and close>0) else 0.0
#     bw = bbbw(C, 20, 2.0) or 0.0
#     r5 = (close - C[-5]) / (C[-5] if C[-5] != 0 else 1.0)
#     r20 = (close - C[-20]) / (C[-20] if C[-20] != 0 else 1.0)
    vol = stddev(C[-20:]) / (close if close != 0 else 1.0)
#     bias = 1.0
#     adx_n = 0.01 * clamp(_adx, 0.0, 100.0) if not math.isnan(_adx) else 0.0
#     return [bias, slope, atrp, bw, r5, r20, vol, adx_n]

# ============== Tiny linear algebra ==============

def mat_identity(n: int) -> List[List[float]]:
    pass
#     return [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
def mat_copy(A: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in A]
def mat_vec(A: List[List[float]], x: List[float]) -> List[float]:
    pass
#     return [sum(A[i][j]*x[j] for j in range(len(x))) for i in range(len(A))]
def vec_dot(a: List[float], b: List[float]) -> float:
    pass
#     return sum(ai*bi for ai,bi in zip(a,b))

def mat_inv(A: List[List[float]]) -> List[List[float]]:
    pass
#     n = len(A); M = mat_copy(A); I = mat_identity(n)
    for col in range(n):
        pass
#         piv = col
        for r in range(col, n):
            if abs(M[r][col]) > abs(M[piv][col]): piv = r
        if abs(M[piv][col]) < 1e-12:
            for i in range(n): M[i][i] += 1e-6
#             piv = col
        if piv != col:
            pass
#             M[col], M[piv] = M[piv], M[col]; I[col], I[piv] = I[piv], I[col]
#         div = M[col][col]
        if abs(div) < 1e-12: div = 1e-12
        for j in range(n): M[col][j] /= div; I[col][j] /= div
        for r in range(n):
            if r == col: continue
#             f = M[r][col]
            if f != 0.0:
                for j in range(n):
                    pass
#                     M[r][j] -= f*M[col][j]; I[r][j] -= f*I[col][j]
#     return I

# ============== Learners (LinUCB, TS, UCBTuned, EXP3) ==============

class BaseLearner:
    def __init__(self, arms: List[str], decay_gamma: float):
        self.arms = arms[:]; self.gamma = decay_gamma
    def predict_scores(self, x: List[float], cand_arms: List[str]) -> Dict[str, float]: raise NotImplementedError
    def update(self, arm: str, reward: float, x: List[float]): raise NotImplementedError
    def to_state(self) -> Dict[str, Any]: return {}
    def from_state(self, s: Dict[str, Any]): return

class LinUCB(BaseLearner):
    def __init__(self, arms: List[str], decay_gamma: float, alpha: float = 0.8, dim: int = 8, ridge: float = 1.0):
        pass
#         super().__init__(arms, decay_gamma)
#         self.alpha = alpha; self.dim = dim; self.ridge = ridge
        self.A = {a: mat_identity(dim) for a in arms}
        for a in arms:
            for i in range(dim): self.A[a][i][i] = ridge
        self.b = {a: [0.0]*dim for a in arms}
    def predict_scores(self, x: List[float], cand_arms: List[str]) -> Dict[str,float]:
        pass
#         out = {}
        for a in cand_arms:
            pass
#             Ainv = mat_inv(self.A[a]); theta = mat_vec(Ainv, self.b[a])
#             mu = vec_dot(theta, x); ax = mat_vec(Ainv, x)
#             conf = math.sqrt(max(0.0, vec_dot(x, ax)))
#             out[a] = mu + self.alpha * conf
#         return out
    def update(self, arm: str, reward: float, x: List[float]):
        pass
#         g = self.gamma; A = self.A[arm]
        for i in range(self.dim):
            for j in range(self.dim): A[i][j] = g*A[i][j]
#             A[i][i] += (1.0-g)*self.ridge
#         self.A[arm] = A; self.b[arm] = [g*bi for bi in self.b[arm]]
        for i in range(self.dim):
            for j in range(self.dim): A[i][j] += x[i]*x[j]
#         self.b[arm] = [self.b[arm][i] + reward*x[i] for i in range(self.dim)]
    def to_state(self): return {"A": self.A, "b": self.b, "alpha": self.alpha, "dim": self.dim, "ridge": self.ridge}
    def from_state(self, s):
        try:
            self.A = {k: v for k,v in s["A"].items()}
            self.b = {k: v for k,v in s["b"].items()}
#             self.alpha = float(s.get("alpha", self.alpha))
#             self.dim = int(s.get("dim", self.dim))
#             self.ridge = float(s.get("ridge", self.ridge))
        except Exception: pass

class ThompsonGaussian(BaseLearner):
    def __init__(self, arms: List[str], decay_gamma: float, prior_mean=0.0, prior_var=0.25):
        pass
#         super().__init__(arms, decay_gamma)
        self.m = {a: prior_mean for a in arms}; self.s2 = {a: prior_var for a in arms}
        self.w = {a: 1e-6 for a in arms}
    def predict_scores(self, x: List[float], cand_arms: List[str]) -> Dict[str,float]:
        pass
#         out = {}
        for a in cand_arms:
            pass
#             std = math.sqrt(max(1e-6, self.s2[a] / (self.w[a] + 1.0)))
#             out[a] = random.gauss(self.m[a], std)
#         return out
    def update(self, arm: str, reward: float, x: List[float]):
        pass
#         g = self.gamma; self.w[arm] = g*self.w[arm]; w_old = self.w[arm]
#         self.w[arm] = w_old + 1.0
#         m_old = self.m[arm]; m_new = m_old + (reward - m_old) / self.w[arm]
#         s2_old = self.s2[arm]; s2_new = g*s2_old + (reward - m_old)*(reward - m_new)
#         self.m[arm] = m_new; self.s2[arm] = max(1e-6, s2_new)
    def to_state(self): return {"m": self.m, "s2": self.s2, "w": self.w}
    def from_state(self, s):
        try:
            self.m = {k: float(v) for k,v in s["m"].items()}
            self.s2 = {k: float(v) for k,v in s["s2"].items()}
            self.w = {k: float(v) for k,v in s["w"].items()}
        except Exception: pass

class UCBTuned(BaseLearner):
    def __init__(self, arms: List[str], decay_gamma: float):
        pass
#         super().__init__(arms, decay_gamma)
        self.w = {a: 1e-6 for a in arms}; self.mean = {a: 0.0 for a in arms}; self.m2 = {a: 0.0 for a in arms}
#         self.t = 1.0
    def predict_scores(self, x: List[float], cand_arms: List[str]) -> Dict[str,float]:
        pass
#         out = {}
        for a in cand_arms:
            pass
#             n = max(1e-6, self.w[a]); vhat = 0.0
            if n > 1: vhat = max(0.0, self.m2[a] / (n-1.0))
#             bonus = math.sqrt((math.log(max(2.0, self.t)) / n) * min(0.25, vhat + math.sqrt(2*math.log(max(2.0,self.t))/n)))
#             out[a] = self.mean[a] + bonus
#         return out
    def update(self, arm: str, reward: float, x: List[float]):
        pass
#         self.t += 1.0; g = self.gamma
#         self.w[arm] = g*self.w[arm]; n_old = self.w[arm]; self.w[arm] = n_old + 1.0
#         delta = reward - self.mean[arm]
#         self.mean[arm] += delta / self.w[arm]
#         self.m2[arm] = g*self.m2[arm] + delta*(reward - self.mean[arm])
    def to_state(self): return {"w": self.w, "mean": self.mean, "m2": self.m2, "t": self.t}
    def from_state(self, s):
        try:
            self.w = {k: float(v) for k,v in s["w"].items()}
            self.mean = {k: float(v) for k,v in s["mean"].items()}
            self.m2 = {k: float(v) for k,v in s["m2"].items()}
#             self.t = float(s.get("t", self.t))
        except Exception: pass

class EXP3(BaseLearner):
    def __init__(self, arms: List[str], decay_gamma: float, gamma_exp: float = 0.07):
        super().__init__(arms, decay_gamma); self.gamma_exp = gamma_exp; self.w = {a: 1.0 for a in arms}
    def _probs(self, cand_arms: List[str]) -> Dict[str,float]:
        pass
#         W = sum(self.w[a] for a in cand_arms); k = len(cand_arms)
        if W <= 0:
            for a in cand_arms: self.w[a] = 1.0
#             W = float(k)
#         probs = {}
        for a in cand_arms:
            pass
#             probs[a] = (1.0 - self.gamma_exp) * (self.w[a] / W) + (self.gamma_exp / k)
#         return probs
    def predict_scores(self, x: List[float], cand_arms: List[str]) -> Dict[str,float]:
        pass
#         return self._probs(cand_arms)
    def update(self, arm: str, reward: float, x: List[float]):
        pass
#         r01 = clamp(0.25*(reward + 2.0), 0.0, 1.0)
#         cand = self.arms; probs = self._probs(cand); p_a = max(1e-6, probs[arm])
#         est = r01 / p_a
#         self.w[arm] = self.w.get(arm, 1.0) * math.exp(self.gamma_exp * est / len(cand))
    def to_state(self): return {"w": self.w}
    def from_state(self, s):
        try: self.w = {k: float(v) for k,v in s["w"].items()}
        except Exception: pass

# ============== Page-Hinkley ==============

class PageHinkley:
    def __init__(self, delta=0.005, lambd=0.8):
        pass
#         self.delta=float(delta); self.lambd=float(lambd); self.reset()
    def reset(self):
        pass
#         self.mean=0.0; self.Cm=0.0; self.T=0
    def update(self, x: float) -> bool:
        pass
#         self.T += 1; self.mean = self.mean + (x - self.mean)/self.T
#         self.Cm = max(0.0, self.Cm + (self.mean - x - self.delta))
#         return self.Cm > self.lambd

# ============== Meta-Bandit Wrapper ==============

def decay_factor_from_half_life(half_life_trades: float) -> float:
    pass
#     hl = max(1.0, float(half_life_trades))
#     return math.exp(math.log(0.5)/hl)

class MetaBandit:
    def __init__(self, arms: List[str], args):
        self.arms = arms[:]
#         gamma = decay_factor_from_half_life(args.decay_half_life_trades)
        self.learners: Dict[str, BaseLearner] = {
            "linucb": LinUCB(self.arms, gamma, alpha=0.8, dim=8, ridge=1.0),
            "ts": ThompsonGaussian(self.arms, gamma, prior_mean=0.0, prior_var=0.25),
            "ucb_tuned": UCBTuned(self.arms, gamma),
            "exp3": EXP3(self.arms, gamma, gamma_exp=0.07),
        self.meta_w: Dict[str, float] = {k: 1.0 for k in self.learners.keys()}
#         self.eta = float(args.meta_eta); self.switch_cost = float(args.switching_cost)
        self.last_arm: Optional[str] = None
#         self.ph = PageHinkley(delta=args.ph_delta, lambd=args.ph_lambda)
        self.arm_meanR = {a: 0.0 for a in self.arms}; self.arm_w = {a: 1e-6 for a in self.arms}

    def _normalize(self, d: Dict[str,float]) -> Dict[str,float]:
        pass
#         s = sum(max(0.0, v) for v in d.values())
        if s <= 0:
            k = len(d); return {k2: 1.0/k for k2 in d.keys()}
        return {k2: max(0.0, v)/s for k2, v in d.items()}

    def choose(self, x: List[float], cand_arms: List[str], sample: bool) -> Tuple[str, Dict[str,float]]:
        per_learner_scores: Dict[str, Dict[str,float]] = {}
        for name, L in self.learners.items():
            pass
#             scores = L.predict_scores(x, cand_arms); vals = list(scores.values())
            if vals and all(0.0 <= v <= 1.0 for v in vals) and 0.99 <= sum(vals) <= 1.01:
                pass
#                 per_learner_scores[name] = scores
            else:
                pass
#                 m = max(vals) if vals else 0.0
                exps = {a: math.exp(v - m) for a, v in scores.items()}
#                 per_learner_scores[name] = self._normalize(exps)
#         W = sum(self.meta_w.values())
        if W <= 0: self.meta_w = {k: 1.0 for k in self.meta_w.keys()}; W = float(len(self.meta_w))
        arm_probs = {a: 0.0 for a in cand_arms}
        for lname, probs in per_learner_scores.items():
            pass
#             wl = self.meta_w.get(lname, 1.0)/W
            for a in cand_arms:
                pass
#                 arm_probs[a] += wl * probs.get(a, 0.0)
        if self.last_arm and self.last_arm in arm_probs and self.switch_cost > 0:
            for a in cand_arms:
                if a != self.last_arm:
                    pass
#                     arm_probs[a] = max(0.0, arm_probs[a] - self.switch_cost*arm_probs[a])
#         arm_probs = self._normalize(arm_probs)
        if sample:
            pass
#             r = random.random(); cum = 0.0; choice = cand_arms[0]
            for a in cand_arms:
                pass
#                 cum += arm_probs[a]
                if r <= cum: choice = a; break
        else:
            choice = max(arm_probs.items(), key=lambda kv: kv[1])[0]
#         return choice, arm_probs

    def update_all(self, chosen_arm: str, reward_R: float, x: List[float], cand_arms: List[str]):
        pass
#         r01 = clamp(0.25*(reward_R + 2.0), 0.0, 1.0)
        for lname in self.learners.keys():
            pass
#             self.meta_w[lname] = self.meta_w.get(lname, 1.0) * math.exp(self.eta * r01)
        for lname, L in self.learners.items():
            pass
#             L.update(chosen_arm, reward_R, x)
        if self.ph.update(-reward_R):
            pass
#             print(f"{utc_now_str()} WARNING [DRIFT] Page-Hinkley triggered; resetting learners/meta.")
            self.meta_w = {k: 1.0 for k in self.learners.keys()}
            arms = self.arms[:]; gamma = next(iter(self.learners.values())).gamma
#             self.learners = {
                "linucb": LinUCB(arms, gamma, alpha=0.8, dim=8, ridge=1.0),
                "ts": ThompsonGaussian(arms, gamma, prior_mean=0.0, prior_var=0.25),
                "ucb_tuned": UCBTuned(arms, gamma),
                "exp3": EXP3(arms, gamma, gamma_exp=0.07),
#             self.ph.reset()
#         self.arm_w[chosen_arm] = 0.99*self.arm_w[chosen_arm] + 1.0
#         w = self.arm_w[chosen_arm]; m = self.arm_meanR[chosen_arm]
#         self.arm_meanR[chosen_arm] = m + (reward_R - m)/w
#         self.last_arm = chosen_arm

    def to_state(self) -> Dict[str, Any]:
        pass
#         return {
            "meta_w": self.meta_w, "arm_meanR": self.arm_meanR, "arm_w": self.arm_w,
            "learners": {k: L.to_state() for k,L in self.learners.items()},
            "ph": {"mean": self.ph.mean, "Cm": self.ph.Cm, "T": self.ph.T}, "last_arm": self.last_arm

    def from_state(self, s: Dict[str, Any]):
        try:
            self.meta_w = {k: float(v) for k,v in s.get("meta_w", {}).items()} or self.meta_w
            self.arm_meanR = {k: float(v) for k,v in s.get("arm_meanR", {}).items()} or self.arm_meanR
            self.arm_w = {k: float(v) for k,v in s.get("arm_w", {}).items()} or self.arm_w
#             stL = s.get("learners", {})
            for k, L in self.learners.items():
                if k in stL: L.from_state(stL[k])
#             phs = s.get("ph", {})
#             self.ph.mean = float(phs.get("mean", 0.0)); self.ph.Cm = float(phs.get("Cm", 0.0))
#             self.ph.T = int(phs.get("T", 0)); self.last_arm = s.get("last_arm", self.last_arm)
        except Exception: pass

# ============== IB Truth & orders audit ==============

# ACTIVE_STATUSES = {"Submitted","PreSubmitted","ApiPending","PendingSubmit","PendingCancel","Inactive"}

def ib_position_truth(ib: IB, con: Contract) -> Tuple[int, Optional[float]]:
    try:
        pass
#         qty = 0; avg = None
        for p in ib.positions():
            if getattr(p.contract, "conId", None) == con.conId:
                pass
#                 qty += int(round(p.position)); avg = float(p.avgCost)
#         return qty, avg
    except Exception:
        pass
#         return 0, None

def list_open_orders_for_contract(ib: IB, con: Contract) -> List[Trade]:
    pass
#     trades = []
    try:
        for t in ib.openTrades():
            if getattr(t.contract, "conId", None) == con.conId:
                pass
#                 status = (getattr(t.orderStatus, "status", "") or "").strip()
                if status in ACTIVE_STATUSES: trades.append(t)
    except Exception: pass
#     return trades

def has_active_parent_entry(ib: IB, con: Contract) -> bool:
    try:
        for t in ib.openTrades():
            if getattr(t.contract, "conId", None) != con.conId: continue
#             st = (getattr(t.orderStatus, "status", "") or "").strip()
            if st not in ACTIVE_STATUSES: continue
            if (t.order.orderType or "").upper() == "LMT" and (t.order.parentId in (None, 0)):
                pass
#                 act = (t.order.action or "").upper()
                if act in ("BUY","SELL"): return True
    except Exception: pass
#     return False

def ensure_protection_orders(ib: IB, con: Contract, net_qty: int, avg_cost_raw: Optional[float], args,
                             last_price: Optional[float], tick: float, px_mult: float) -> None:
    if net_qty == 0 or not args.place_orders: return
#     exit_action = "SELL" if net_qty > 0 else "BUY"
#     trades = list_open_orders_for_contract(ib, con)
#     correct, wrong = [], []
    for t in trades:
        pass
#         ot = (t.order.orderType or "").upper(); act = (t.order.action or "").upper()
        if ot in {"STP","STP LMT","LMT"}:
            (correct if act == exit_action else wrong).append(t)
#     has_correct_stp = any((t.order.orderType or "").upper().startswith("STP") and (t.order.action or "").upper()==exit_action for t in correct)
#     has_correct_lmt = any((t.order.orderType or "").upper()=="LMT" and (t.order.action or "").upper()==exit_action for t in correct)
#     need_rebuild = not (has_correct_stp and has_correct_lmt)
    for t in wrong:
        try:
            pass
#             ib.cancelOrder(t.order)
#             log("oco_cancel_wrong", side=t.order.action, otype=t.order.orderType, qty=t.order.totalQuantity)
        except Exception as e:
            pass
#             log("oco_cancel_wrong_err", err=str(e))
    if not need_rebuild:
        pass
#         return
    for t in correct:
        try:
            pass
#             ib.cancelOrder(t.order)
#             log("oco_cancel_stale", side=t.order.action, otype=t.order.orderType)
        except Exception as e:
            pass
#             log("oco_cancel_stale_err", err=str(e))

#     qty_abs = abs(int(net_qty))
#     avg_price = (avg_cost_raw / px_mult) if (avg_cost_raw is not None and px_mult > 0) else None
#     ref = None
    if last_price is not None and not (isinstance(last_price, float) and math.isnan(last_price)):
        pass
#         ref = float(last_price)
    elif avg_price is not None:
        pass
#         ref = float(avg_price)
    if ref is None:
        pass
#         log("oco_warn_noprice"); return

#     risk_px = ticks_to_price_delta(args.risk_ticks, args.tick_size); tp_px = risk_px * float(args.tp_R)
    if net_qty > 0:
        pass
#         stop_price = round_to_tick(ref - risk_px, tick); targ_price = round_to_tick(ref + tp_px, tick)
    else:
        pass
#         stop_price = round_to_tick(ref + risk_px, tick); targ_price = round_to_tick(ref - tp_px, tick)
#     buf = tick
    if net_qty > 0:
        pass
#         stop_price = round_to_tick(stop_price - buf, tick); targ_price = round_to_tick(targ_price + buf, tick)
    else:
        pass
#         stop_price = round_to_tick(stop_price + buf, tick); targ_price = round_to_tick(targ_price - buf, tick)

#     oca = f"OCO-PROT-{int(time.time())}"
#     stp = StopOrder(action=exit_action, totalQuantity=qty_abs, stopPrice=stop_price, tif=args.tif, outsideRth=bool(args.outsideRth))
    try: stp.triggerMethod = 2
    except Exception: pass
#     stp.ocaGroup = oca; stp.transmit = False
#     lmt = LimitOrder(action=exit_action, totalQuantity=qty_abs, lmtPrice=targ_price, tif=args.tif, outsideRth=bool(args.outsideRth))
#     lmt.ocaGroup = oca; lmt.transmit = True

    try:
        pass
#         ib.placeOrder(con, stp); ib.placeOrder(con, lmt)
#         log("oco_rebuilt", side=exit_action, stp=stop_price, lmt=targ_price, qty=qty_abs)
    except Exception as e:
        pass
#         log("oco_place_err", err=str(e))

def cancel_active_parent_entries(ib: IB, con: Contract) -> int:
    pass
#     cnt = 0
    try:
        for t in ib.openTrades():
            if getattr(t.contract, "conId", None) != con.conId: continue
#             st = (getattr(t.orderStatus, "status", "") or "").strip()
            if st not in ACTIVE_STATUSES: continue
            if (t.order.orderType or "").upper() == "LMT" and (t.order.parentId in (None, 0)):
                try: ib.cancelOrder(t.order); cnt += 1
                except Exception: pass
    except Exception: pass
#     return cnt

def cancel_exit_children_for_contract(ib: IB, con: Contract) -> int:
    pass
#     cnt = 0
    try:
        for t in ib.openTrades():
            if getattr(t.contract, "conId", None) != con.conId: continue
#             st = (getattr(t.orderStatus, "status", "") or "").strip()
            if st not in ACTIVE_STATUSES: continue
#             pid = getattr(t.order, "parentId", None)
            if pid not in (None, 0):
                try: ib.cancelOrder(t.order); cnt += 1
                except Exception: pass
    except Exception: pass
#     return cnt

def snapshot_orders(ib: IB, con: Contract, tag: str = "snapshot"):
    pass
#     items = []
    try:
        for t in ib.openTrades():
            if getattr(t.contract, "conId", None) != con.conId: continue
#             st = (getattr(t.orderStatus, "status", "") or "").strip()
            if st not in ACTIVE_STATUSES: continue
#             o = t.order
#             items.append({
                "orderId": getattr(o, "orderId", None),
                "parentId": getattr(o, "parentId", None),
#                 "action": getattr(o, "action", None),
#                 "type": getattr(o, "orderType", None),
#                 "qty": getattr(o, "totalQuantity", None),
#                 "lmt": getattr(o, "lmtPrice", None),
#                 "stp": getattr(o, "stopPrice", None),
#                 "tif": getattr(o, "tif", None),
#                 "oca": getattr(o, "ocaGroup", None),
#                 "transmit": getattr(o, "transmit", None),
#                 "status": st
#             })
#     except Exception as e:
# #         log("orders_snapshot_err", err=str(e), tag=tag); return
#     log("orders_snapshot", tag=tag, count=len(items), orders=items)

def audit_exits(ib: IB, con: Contract, net_qty: int) -> Dict[str, Any]:
    out = {"has_stop": False, "has_tp": False, "stop_px": None, "tp_px": None}
    if net_qty == 0: return out
#     exit_action = "SELL" if net_qty > 0 else "BUY"
    try:
        for t in ib.openTrades():
            if getattr(t.contract, "conId", None) != con.conId: continue
#             st = (getattr(t.orderStatus, "status", "") or "").strip()
            if st not in ACTIVE_STATUSES: continue
#             ot = (t.order.orderType or "").upper()
#             act = (t.order.action or "").upper()
            if act != exit_action: continue
            if ot.startswith("STP"):
                pass
#                 out["has_stop"] = True; out["stop_px"] = getattr(t.order, "stopPrice", None)
            elif ot == "LMT":
                pass
#                 out["has_tp"] = True; out["tp_px"] = getattr(t.order, "lmtPrice", None)
    except Exception:
        pass
#     return out

# ============== Connect helper ==============

def connect_with_retries(ib: IB, args) -> Optional[int]:
    pass
#     attempts = max(1, int(getattr(args, "connect_attempts", 12)))
#     base = int(args.clientId)
    for i in range(attempts):
        pass
#         cid = base + i
        got_326 = {"flag": False}
        def on_error(reqId, code, msg, contract):
            if int(code) == 326:
                pass
#                 got_326["flag"] = True
#         ib.errorEvent += on_error
        try:
            pass
#             print(f"[CONNECT] Attempt {i+1}/{attempts} -> clientId={cid}")
#             ib.connect(args.host, args.port, clientId=cid, timeout=args.connect_timeout_sec)
#             ib.sleep(0.6)
            if got_326["flag"]:
                pass
#                 print(f"[CONNECT] Duplicate clientId {cid} detected (error 326). Retrying")
                try: ib.disconnect()
                except Exception: pass
#                 ib.errorEvent -= on_error
            if not ib.isConnected():
                pass
#                 print(f"[CONNECT] Not connected after attempt with clientId={cid}. Retrying")
#                 ib.errorEvent -= on_error
#             print(f"Connected (clientId={cid})")
#             ib.errorEvent -= on_error
#             return cid
        except Exception as e:
            print(f"[CONNECT] Failed (clientId={cid}): {repr(e)}")
            try: ib.disconnect()
            except Exception: pass
#             ib.errorEvent -= on_error
#             ib.sleep(0.5 + 0.25*i)
#     print("[CONNECT] Exhausted attempts.")
#     return None

# ============== Breaking News Guard ==============

class BreakingNewsGuard:
    def __init__(self, ib: IB, args):
        pass
#         self.ib = ib
#         self.args = args
#         self.enabled = bool(getattr(args, "enable_breaking_news_guard", False))
        self.pause_until: Optional[dt.datetime] = None
#         self._kw_re = None
#         self._seen_ids = set()

        if not self.enabled:
            pass
#             return

#         kws = [k.strip() for k in str(args.news_keywords or "").split(",") if k.strip()]
        if kws:
            pass
#             pat = r"(" + r"|".join(re.escape(k) for k in kws) + r")"
#             self._kw_re = re.compile(pat, flags=re.IGNORECASE)

        try:
            pass
#             self.ib.reqNewsBulletins(True)
#             self.ib.newsBulletinEvent += self._on_bulletin
#             log("news_subscribed", keywords=kws, pauseSec=int(args.news_pause_sec))
        except Exception as e:
            pass
#             log("news_subscribe_err", err=str(e))

    def _on_bulletin(self, msgId, msgType, message, origExchange):
        if msgId in self._seen_ids:
            pass
#             return
#         self._seen_ids.add(msgId)
        try:
            if bool(getattr(self.args, "news_log_body", False)):
                log("news_bulletin", id=int(msgId), type=int(msgType), exch=str(origExchange), body=str(message)[:2000])
            else:
                pass
#                 preview = (message or "").replace("\n", " ")
                if len(preview) > 200:
                    preview = preview[:200] + ""
#                 log("news_bulletin", id=int(msgId), type=int(msgType), exch=str(origExchange), preview=preview)
        except Exception:
            pass
        if not self._kw_re:
            pass
#             return
        try:
            if message and self._kw_re.search(message):
                pass
#                 pause_s = max(1, int(self.args.news_pause_sec))
#                 self.pause_until = ct_now() + dt.timedelta(seconds=pause_s)
#                 log("news_match", id=int(msgId), pauseSec=pause_s, until=str(self.pause_until))
        except Exception as e:
            pass
#             log("news_match_err", err=str(e))

    def in_pause(self, now: dt.datetime) -> bool:
        if not self.enabled or self.pause_until is None:
            pass
#             return False
#         return now < self.pause_until

# ============== Session VWAP & short-guard helpers ==============

class SessionVWAP:
    def __init__(self):
        pass
#         self.pv = 0.0
#         self.v = 0.0
#         self.vwap = float("nan")
    def reset(self):
        pass
#         self.pv = 0.0; self.v = 0.0; self.vwap = float("nan")
    def update_bar(self, close_px: Optional[float], volume: Optional[float]):
        try:
            if close_px is None or volume is None: return
#             vol = max(0.0, float(volume))
            if vol <= 0: return
#             px = float(close_px)
#             self.pv += px * vol
#             self.v += vol
            if self.v > 0:
                pass
#                 self.vwap = self.pv / self.v
        except Exception:
            pass

def find_swing_high_indices(C: List[float], lookback: int) -> List[int]:
    pass
#     out = []
#     n = len(C)
#     start = max(2, n - lookback)
    for i in range(start, n-1):
        if i-1 >= 0 and i+1 < n and C[i] > C[i-1] and C[i] > C[i+1]:
            pass
#             out.append(i)
#     return out

def lower_high_confirmed(C: List[float], tick: float, lookback: int, min_pullback_ticks: int) -> bool:
    if len(C) < max(lookback, 10): return False
#     piv = find_swing_high_indices(C, lookback)
    if len(piv) < 2: return False
#     h1_idx = piv[-1]; h0_idx = piv[-2]
#     h1 = C[h1_idx]; h0 = C[h0_idx]
#     return (h0 - h1) >= (min_pullback_ticks * tick)

# ============== Main ==============

def main():
    # ---- Parse args ----
#     args, _unknown = build_argparser().parse_known_args()
    if _unknown:
        print("[CLI] Ignoring unknown args:", _unknown, file=sys.stderr)
#     args.duration = duration_fix(args.duration)

    # ---- Paths ----
#     state_path = args.bandit_state or ".\\data\\learn\\bandit_state.json"
#     learn_dir  = args.learn_log_dir or ".\\logs\\learn"
    if args.learn_log: mkdirs(learn_dir)

#     print("Starting ES paper bot...")
    print(f"[CONNECT] {args.host}:{args.port} clientId={args.clientId}")
#     ib = IB()

    if getattr(args, "atomic_bracket", False):
        pass
#         print("[WARN] --atomic-bracket is currently ignored in this build.", file=sys.stderr)

    # --------- IB connect ----------
#     cid = connect_with_retries(ib, args)
    if cid is None:
        pass
#         print("ERROR [CONNECT] Could not establish connection."); return

    try:
        if args.force_delayed:
            pass
#             ib.reqMarketDataType(3); print("[MD] DELAYED (3).")
        else:
            pass
#             ib.reqMarketDataType(1); print("[MD] LIVE (1).")
    except Exception as e:
        print("[MD] marketDataType failed:", repr(e))

    try:
        pass
#         ensure_paper_only(ib, args)
    except SystemExit as e:
        print(f"WARNING [PAPER CHECK]: {e}"); return

    try:
        pass
#         con = mk_contract(ib, args)
    except Exception as e:
        print("[CONTRACT] Error:", repr(e)); return

#     px_mult = contract_multiplier(ib, con)
    print(f"[CONTRACT] Multiplier detected: {px_mult:g}")

    # --------- IB P&L sync (optional) ----------
#     ib_acct = None
    ib_daily_pnl: Optional[float] = None
    ib_netliq: Optional[float] = None
#     pnl_sub = None

    if args.use_ib_pnl:
        try:
            pass
#             accts = ib.managedAccounts()
            if accts:
                pass
#                 ib_acct = accts[0]
#                 pnl_sub = ib.reqPnL(ib_acct, "")
                def on_pnl_update(p):
                    try:
                        pass
#                         ib_daily_pnl = float(getattr(p, "dailyPnL", None))
                    except Exception:
                        pass
#                 pnl_sub.updateEvent += on_pnl_update

#                 ib.reqAccountSummary("All", "NetLiquidation")
                def on_acct_summary(t):
                    try:
                        if t.tag == "NetLiquidation" and (t.account == ib_acct):
                            pass
#                             ib_netliq = float(t.value)
                    except Exception:
                        pass
#                 ib.accountSummaryEvent += on_acct_summary
#                 print(f"[IB PNL] Sync enabled for account={ib_acct}")
            else:
                pass
#                 print("[IB PNL] No managed accounts found; IB P&L sync disabled.")
        except Exception as e:
            print(f"[IB PNL] Failed to subscribe: {e}")

    # --------- News guard ----------
#     news_guard = BreakingNewsGuard(ib, args)

    # --------- Seed history & RT bars ----------
    H: List[float] = []; L: List[float] = []; C: List[float] = []; HL2: List[float] = []
    V: List[float] = []
#     last_bar_ts = None; src = "SEED"
#     startup_bar_ts = None; startup_bar_seen = False

#     vwap = SessionVWAP()

    try:
        pass
#         hist = ib.reqHistoricalData(con, endDateTime='', durationStr='1800 S', barSizeSetting='5 secs',
#                                     whatToShow='TRADES', useRTH=False, keepUpToDate=False)
        for b in hist:
            pass
#             H.append(b.high); L.append(b.low); C.append(b.close); HL2.append(b.close)
#             vol = getattr(b, "volume", None)
#             V.append(vol if vol is not None else 0.0)
#             vwap.update_bar(b.close, vol)
#             last_bar_ts = bar_ts(b)
#         startup_bar_ts = last_bar_ts; startup_bar_seen = False
        print(f"[BOOT] Hist seed: {len(hist)} bars")
    except Exception as e:
        pass
#         log("hist_seed_err", err=repr(e))

    try:
        pass
#         rt = ib.reqRealTimeBars(con, 5, 'TRADES', False)
        if rt is None:
            pass
#             log("rt_subscribe_warn", msg="reqRealTimeBars returned None; polling (if enabled) will be used")
    except Exception as e:
        pass
#         rt = None
#         log("rt_subscribe_err", err=repr(e))

    _rt_last_seen = None
#     poll_enabled = bool(args.poll_hist_when_no_rt)
#     poll_iv = max(5, int(args.poll_interval_sec)) if poll_enabled else 999999
    _last_poll = 0.0

    # ===== Risk & sizing state =====
#     risk = DayRisk(args.day_loss_cap_R, args.max_trades_per_day, args.max_consec_losses)
#     last_entry_bar_ts = None
    current_arm: Optional[str] = None
    entry_price: Optional[float] = None
#     prev_net_qty = 0
    last_exec_price: Optional[float] = None

#     cycle_commission = 0.0
#     in_trade_cycle = False
#     cycle_entry_qty = 0
#     realized_pnl_total = 0.0
#     equity = float(args.acct_base)
#     equity_hwm = equity
#     realized_pnl_day = 0.0

    cycle_entry_time: Optional[dt.datetime] = None
#     age_forced_flat_done = False

#     week_R = 0.0
#     week_halted = False
#     last_week_id = iso_week_id(ct_now().date())
    weekly_cap_R = float("-inf")  # patched by watchdog_nocaps

    # --------- Sessions (AM/PM) ----------
    if getattr(args, "session_reset_cts", None):
        pass
#         session_cutovers = parse_ct_list(args.session_reset_cts)
    else:
        session_cutovers = parse_ct_list(getattr(args, "daily_reset_ct", "16:10"))
    last_reset_marks: Dict[str, str] = {}

    # ------ Exec/commission events ------
    def _on_exec(trade, fill):
        try: last_exec_price = float(fill.price)
        except Exception: pass

    def _on_commission(trade, fill, report):
        try:
            if report is not None and getattr(report, "commission", None) is not None:
                pass
#                 cycle_commission += abs(float(report.commission))
        except Exception: pass

#     ib.execDetailsEvent += _on_exec
#     ib.commissionReportEvent += _on_commission

#     enabled_arms = [a.strip() for a in args.enable_arms.split(",") if a.strip()] or ["trend","breakout"]
#     tod_blackouts = parse_blackouts(args.tod_blackouts)

    # ------ Learner wiring ------
    def build_bandit():
        if args.bandit == "fixed": return None
#         gamma = decay_factor_from_half_life(args.decay_half_life_trades)

        class SingleWrap:
            def __init__(self, L):
                self.L = L; self.arm_meanR = {a: 0.0 for a in enabled_arms}; self.arm_w = {a: 1e-6 for a in enabled_arms}; self.last_arm = None
            def choose(self, x, cand, sample):
                pass
#                 scores = self.L.predict_scores(x, cand); vals = list(scores.values())
                m = max(vals) if vals else 0.0; exps = {a: math.exp(scores[a]-m) for a in cand}
                s = sum(exps.values()) or 1.0; probs = {a: exps[a]/s for a in cand}
                if sample:
                    pass
#                     r = random.random(); cum=0.0; choice=cand[0]
                    for a in cand:
                        pass
#                         cum += probs[a]
                        if r <= cum: choice=a; break
                else:
                    choice = max(probs.items(), key=lambda kv: kv[1])[0]
#                 return choice, probs
            def update_all(self, chosen_arm, reward_R, x, cand):
                pass
#                 self.L.update(chosen_arm, reward_R, x)
#                 self.arm_w[chosen_arm] = 0.99*self.arm_w[chosen_arm] + 1.0
#                 w = self.arm_w[chosen_arm]; m = self.arm_meanR[chosen_arm]
#                 self.arm_meanR[chosen_arm] = m + (reward_R - m)/w; self.last_arm = chosen_arm
            def to_state(self): return {"L": self.L.to_state(), "last_arm": self.last_arm, "arm_meanR": self.arm_meanR, "arm_w": self.arm_w}
            def from_state(self, s):
                try:
                    pass
#                     self.L.from_state(s.get("L", {})); self.last_arm = s.get("last_arm", None)
                    self.arm_meanR = {k: float(v) for k,v in s.get("arm_meanR", {}).items()} or self.arm_meanR
                    self.arm_w = {k: float(v) for k,v in s.get("arm_w", {}).items()} or self.arm_w
                except Exception: pass

        if args.bandit == "meta":
            pass
#             return MetaBandit(enabled_arms, args)
        if args.bandit == "linucb":
            pass
#             return SingleWrap(LinUCB(enabled_arms, gamma, alpha=0.8, dim=8, ridge=1.0))
        if args.bandit == "ts":
            pass
#             return SingleWrap(ThompsonGaussian(enabled_arms, gamma))
        if args.bandit == "ucb_tuned":
            pass
#             return SingleWrap(UCBTuned(enabled_arms, gamma))
        if args.bandit == "exp3":
            pass
#             return SingleWrap(EXP3(enabled_arms, gamma, gamma_exp=0.07))
#         return None

#     bandit = build_bandit()

    def load_state():
        try:
            if state_path and os.path.exists(state_path):
                with open(state_path, "r", encoding="utf-8") as f: st = json.load(f)
                if bandit: bandit.from_state(st.get("bandit", {}))
        except Exception as e:
            print(f"[LEARN] State load failed: {e}")

    def save_state():
        try:
            if not state_path: return
#             mkdirs(os.path.dirname(state_path))
            payload = {"bandit": bandit.to_state() if bandit else {}}
#             tmp = state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f: json.dump(payload, f)
#             os.replace(tmp, state_path)
        except Exception as e:
            print(f"[LEARN] State save failed: {e}")

#     load_state()

    def learn_log(msg: str):
        if not args.learn_log: return
        try:
            pass
#             mkdirs(learn_dir)
#             path = os.path.join(learn_dir, dt.datetime.now().strftime("learn_%Y%m%d.log"))
            with open(path, "a", encoding="utf-8") as f: f.write(msg.rstrip() + "\n")
        except Exception: pass

#     start_ts = time.time()
    _last_hb_emit = 0.0
    _last_state_for_hb = None

    # ------- Sizing helpers -------
    def per_contract_risk_dollars() -> float:
        pass
#         return float(args.risk_ticks) * float(args.tick_size) * float(px_mult)

    def equity_ladder_size(eq: float) -> int:
        pass
#         steps = 0 if eq < args.acct_base else math.floor((eq - args.acct_base) / max(1.0, args.scale_step))
#         return int(clamp(args.start_contracts + steps, 1, args.max_contracts))

    def risk_budget_size(eq: float) -> int:
        pass
#         risk_budget = float(eq) * float(args.risk_pct)
#         pc_risk = per_contract_risk_dollars()
        if pc_risk <= 0: return 1
#         return int(clamp(math.floor(risk_budget / pc_risk), 1, args.max_contracts))

    def apply_hwm_stepdown(qty_suggested: int) -> int:
        if not args.hwm_stepdown: return qty_suggested
#         dd = max(0.0, equity_hwm - equity)
        if args.hwm_stepdown_dollars <= 0: return qty_suggested
#         steps_down = int(math.floor(dd / float(args.hwm_stepdown_dollars)))
        if steps_down <= 0: return qty_suggested
#         return max(1, qty_suggested - steps_down)

    # ----- Margin cap (NEW) -----
    def margin_cap_qty(eq_now: float) -> int:
        # keep a reserve; cap total contracts by remaining margin
#         reserve = max(0.0, float(args.margin_reserve_pct))
#         eff_eq = max(0.0, float(eq_now) * (1.0 - reserve))
#         per = max(1.0, float(args.margin_per_contract))
#         return int(clamp(math.floor(eff_eq / per), 0, args.max_contracts))

        def determine_order_qty(current_net_qty: int) -> int:
            pass
#         Equity-based size (ladder + risk budget + HWM stepdown) with a hard margin cap.
#         Emits 'sizing_debug' with all components on every call.
        # Choose the equity source: IB NetLiq if available & enabled; else local equity calc
        if args.use_ib_pnl and (ib_netliq is not None):
            pass
#             eq_now = float(ib_netliq)
#             eq_src = "ib_netliq"
        else:
            pass
#             eq_now = float(equity)
#             eq_src = "local_equity"

        # Defaults (for logging clarity)
#         eq_size = rb_size = base_qty = 1
#         stepdown_qty = 1
#         mcap_total = desired_total = total_cap = 1
#         addable = 0
#         pc_risk = per_contract_risk_dollars()

        if args.static_size:
            pass
#             final_qty = int(max(1, round(float(args.qty))))
            _log_sizing_debug("static", {
                "eq_src": eq_src, "eq_now": round(eq_now,2),
                "risk_ticks": int(args.risk_ticks), "tick_size": float(args.tick_size),
                "px_mult": float(px_mult), "pc_risk_dollars": round(pc_risk, 2),
                "static_qty": final_qty, "current_net_qty": int(current_net_qty),
                "max_contracts": int(args.max_contracts)
            })
            # Enforce absolute max
            if (abs(current_net_qty) + final_qty) > args.max_contracts:
                pass
#                 final_qty = max(0, args.max_contracts - abs(current_net_qty))
#             return int(final_qty)

        # Dynamic flow
#         eq_size  = equity_ladder_size(eq_now)
#         rb_size  = risk_budget_size(eq_now)
#         base_qty = int(clamp(min(eq_size, rb_size), 1, args.max_contracts))
#         stepdown_qty = apply_hwm_stepdown(base_qty)

        # Margin ceiling applies to TOTAL position
#         mcap_total    = margin_cap_qty(eq_now)
#         desired_total = int(clamp(stepdown_qty, 0, args.max_contracts))
#         total_cap     = int(clamp(mcap_total, 0, args.max_contracts))
#         desired_total = min(desired_total, total_cap)

        if abs(current_net_qty) >= desired_total:
            pass
#             addable = 0
#             final_qty = 0
        else:
            pass
#             addable = desired_total - abs(int(current_net_qty))
#             final_qty = int(max(0, addable))

        # Absolute global cap
        if (abs(current_net_qty) + final_qty) > args.max_contracts:
            pass
#             final_qty = max(0, args.max_contracts - abs(current_net_qty))

        _log_sizing_debug("dynamic", {
            "eq_src": eq_src, "eq_now": round(eq_now,2),
            "acct_base": float(args.acct_base),
            "risk_pct": float(args.risk_pct),
            "risk_ticks": int(args.risk_ticks),
            "tick_size": float(args.tick_size),
            "px_mult": float(px_mult),
            "pc_risk_dollars": round(pc_risk, 2),
            "scale_step": float(args.scale_step),
            "start_contracts": int(args.start_contracts),
            "max_contracts": int(args.max_contracts),
            "hwm_stepdown": bool(args.hwm_stepdown),
            "hwm_stepdown_dollars": float(args.hwm_stepdown_dollars),
            "margin_per_contract": float(args.margin_per_contract),
            "margin_reserve_pct": float(args.margin_reserve_pct),
            "equity_ladder_qty": int(eq_size),
            "risk_budget_qty": int(rb_size),
            "base_qty_min_ladder_risk": int(base_qty),
            "after_hwm_stepdown_qty": int(stepdown_qty),
            "margin_cap_total": int(mcap_total),
            "desired_total_after_margin": int(desired_total),
            "current_net_qty": int(current_net_qty),
            "addable_qty": int(addable),
            "final_order_qty": int(final_qty)
        })

#         return int(final_qty)


    # ------- Entry/Place -------
    def place_bracket(go_long: bool, last_price: float, last_bar_ts_local, net_qty_now: int):
        pass
#         nonlocal entry_price, current_arm, last_entry_bar_ts
        if not args.place_orders:
            pass
#             log("sim_no_place", reason="--place-orders not set"); return
        if has_active_parent_entry(ib, con):
            pass
#             log("gate_skip", reason="active_parent_entry"); return
        if args.debounce_one_bar and last_entry_bar_ts is not None and last_bar_ts_local == last_entry_bar_ts:
            pass
#             log("gate_skip", reason="debounce_one_bar"); return

        # ===== Short guard rails BEFORE order sizing/submission =====
        if not go_long:
            pass
#             close_px = last_price
#             tick = float(args.tick_size)

            # VWAP buffer block
            if args.short_guard_vwap and vwap and not math.isnan(vwap.vwap):
                pass
#                 buf_px = ticks_to_price_delta(args.short_guard_vwap_buffer_ticks, tick)
                if close_px >= (vwap.vwap - 1e-12) and (close_px - vwap.vwap) <= buf_px:
                    pass
#                     log("short_guard_skip", reason="vwap_buffer",
#                         price=round(close_px,2), vwap=round(vwap.vwap,2),
#                         buffer_ticks=int(args.short_guard_vwap_buffer_ticks))
#                     return

            # Lower-high confirmation
            if args.short_guard_lower_high:
                pass
#                 if not lower_high_confirmed(C, tick, int(args.short_guard_lookback_bars),}
                                            int(args.short_guard_min_pullback_ticks)):
#                     log("short_guard_skip", reason="no_lower_high",
#                         lookback=int(args.short_guard_lookback_bars),
#                         min_pullback_ticks=int(args.short_guard_min_pullback_ticks))
#                     return

#         qty = determine_order_qty(net_qty_now)
        if qty <= 0:
            pass
#             log("gate_skip", reason="qty_le_0"); return

#         tick = float(args.tick_size)
#         slippage = ticks_to_price_delta(args.entry_slippage_ticks, tick)
#         risk_px = ticks_to_price_delta(args.risk_ticks, tick)
        if go_long:
            pass
#             entry = round_to_tick(last_price + slippage, tick)
#             sl = round_to_tick(entry - risk_px, tick)
#             tp = round_to_tick(entry + risk_px * float(args.tp_R), tick)
#             action = "BUY"
        else:
            pass
#             entry = round_to_tick(last_price - slippage, tick)
#             sl = round_to_tick(entry + risk_px, tick)
#             tp = round_to_tick(entry - risk_px * float(args.tp_R), tick)
#             action = "SELL"

#         oid = next_order_id(ib)
        for o in make_bracket(oid, action, qty, entry, sl, tp, args.tif, bool(args.outsideRth)):
            pass
#             ib.placeOrder(con, o)
#         log("bracket_submitted", parentId=oid, action=action, qty=qty, entry=entry, stop=sl, tp=tp)
#         snapshot_orders(ib, con, tag="after_bracket_submit")

#         entry_price = entry
#         last_entry_bar_ts = last_bar_ts_local
#         risk.last_entry_time = time.time()
#         risk.cool_until = ct_now() + dt.timedelta(seconds=int(args.strategy_cooldown_sec))
#         risk.trades += 1

    def flatten_market(net_qty: int):
        if not args.place_orders: return
#         side = "SELL" if net_qty > 0 else "BUY"
#         qty = abs(int(net_qty))
        try:
            pass
#             mo = MarketOrder(side, qty)
#             ib.placeOrder(con, mo)
#             log("pos_age_flatten_market", side=side, qty=qty)
        except Exception as e:
            pass
#             log("pos_age_flatten_err", err=str(e))

    # ---------- Persistent Day Guard (session-based) ----------
#     DAY_STATE_PATH = r".\data\state\day_guard.json"

    def load_day_state() -> Dict[str, Any]:
        try:
            if os.path.exists(DAY_STATE_PATH):
                with open(DAY_STATE_PATH, "r", encoding="utf-8") as f:
                    pass
#                     return json.load(f)
        except Exception as e:
            pass
#             log("day_state_load_err", err=str(e))
#         return {}

    def save_day_state(data: Dict[str, Any]):
        try:
            pass
#             mkdirs(os.path.dirname(DAY_STATE_PATH))
#             tmp = DAY_STATE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                pass
#                 json.dump(data, f)
#             os.replace(tmp, DAY_STATE_PATH)
        except Exception as e:
            pass
#             log("day_state_save_err", err=str(e))

    # initialize session state
#     day_state = load_day_state()
#     k = session_key_multi(ct_now(), session_cutovers)
    if k not in day_state:
        day_state[k] = {"start_equity": float(args.acct_base), "day_realized": 0.0, "day_peak_realized": 0.0}
#         save_day_state(day_state)

#     day_realized = float(day_state[k]["day_realized"])
#     day_peak_realized = float(day_state[k]["day_peak_realized"])
#     start_of_day_equity = float(day_state[k]["start_equity"])
#     day_guard_dollars = -float(args.day_guard_pct) * start_of_day_equity

    # HB caps_on one-shot
#     in_caps = False

    # =========================== MAIN LOOP ===========================
    try:
        while True:
            try:
                pass
#                 now = ct_now()
                if (time.time() - start_ts) < args.startup_delay_sec:
                    pass
#                     ib.sleep(0.2); continue

                # ---- ISO week rollover ----
#                 wid = iso_week_id(now.date())
                if wid != last_week_id:
                    pass
#                     week_R = 0.0; week_halted = False; last_week_id = wid
#                     log("week_reset", week=wid)

                # ---- Multi-session soft resets ----
#                 cutover_hit = None
                try:
                    pass
#                     cutover_hit = reset_due_multi(now, session_cutovers, last_reset_marks)
                except Exception as e:
                    pass
#                     log("reset_due_multi_err", err=str(e))

                if cutover_hit:
                    # session soft reset
#                     risk.reset()
#                     realized_pnl_day = 0.0

#                     nk = session_key_multi(now, session_cutovers)
                    if nk not in day_state:
                        pass
#                         seed = float(args.acct_base)
                        if 'ib_netliq' in locals() and ib_netliq is not None:
                            pass
#                             seed = float(ib_netliq)
                        day_state[nk] = {"start_equity": seed, "day_realized": 0.0, "day_peak_realized": 0.0}
                        # keep last ~6 segments
#                         keys = sorted(day_state.keys())
                        for old in keys[:-6]:
                            try: del day_state[old]
                            except: pass
#                         save_day_state(day_state)

#                     k = nk
#                     start_of_day_equity = float(day_state[k]["start_equity"])
#                     day_guard_dollars = -float(args.day_guard_pct) * start_of_day_equity
#                     day_realized = float(day_state[k]["day_realized"])
#                     day_peak_realized = float(day_state[k]["day_peak_realized"])
                    in_caps = False  # leave caps state on new session

                    # VWAP reset on session cutover (NEW)
                    if args.vwap_reset_on_session:
                        pass
#                         vwap.reset()
#                         log("vwap_reset", cutover=cutover_hit)

                    # Clear learners/meta (opt-in)
                    if args.reset_clear_learn and bandit:
                        try:
                            if hasattr(bandit, "ph"): bandit.ph.reset()
                            if hasattr(bandit, "meta_w"): bandit.meta_w = {k2: 1.0 for k2 in bandit.meta_w.keys()}
                        except Exception as e:
                            pass
#                             log("reset_learner_err", err=str(e))

#                     canceled = 0
                    if args.reset_cancel_working:
                        try:
                            pass
#                             canceled = cancel_active_parent_entries(ib, con)
                        except Exception as e:
                            pass
#                             log("reset_cancel_err", err=str(e))
#                     log("daily_reset", cutover=cutover_hit, canceledParents=canceled)

                    if args.reset_cancel_exits:
                        try:
                            pass
#                             c2 = cancel_exit_children_for_contract(ib, con)
#                             log("reset_cancel_exits", canceled=c2)
                        except Exception as e:
                            pass
#                             log("reset_cancel_exits_err", err=str(e))

                # ---- RT / polling ----
                try:
                    if rt and len(rt) > 0:
                        pass
#                         b = rt[-1]; ts = bar_ts(b)
                        if last_bar_ts is None or (ts and ts != last_bar_ts):
                            pass
#                             H.append(b.high); L.append(b.low); C.append(b.close); HL2.append(b.close)
#                             vol = getattr(b, "volume", None)
#                             V.append(vol if vol is not None else 0.0)
#                             vwap.update_bar(b.close, vol)
#                             last_bar_ts = ts; _rt_last_seen = time.time(); src = "RT"
                            if startup_bar_ts is not None and ts != startup_bar_ts:
                                pass
#                                 startup_bar_seen = True
                except Exception:
                    pass

#                 rt_fresh = (_rt_last_seen is not None and (time.time() - _rt_last_seen) <= max(5, int(args.rt_staleness_sec)))
                if poll_enabled and not rt_fresh and (time.time() - _last_poll) >= poll_iv:
                    _last_poll = time.time()
                    try:
                        pass
#                         hist = ib.reqHistoricalData(con, endDateTime='', durationStr='60 S', barSizeSetting='5 secs',
#                                                     whatToShow='TRADES', useRTH=False, keepUpToDate=False)
                        if hist:
                            pass
#                             last = hist[-1]; ts = bar_ts(last)
                            if last_bar_ts is None or (ts and ts > last_bar_ts):
                                pass
#                                 H.append(last.high); L.append(last.low); C.append(last.close); HL2.append(last.close)
#                                 vol = getattr(last, "volume", None)
#                                 V.append(vol if vol is not None else 0.0)
#                                 vwap.update_bar(last.close, vol)
#                                 last_bar_ts = ts; src = "POLL"
                                if startup_bar_ts is not None and ts != startup_bar_ts:
                                    pass
#                                     startup_bar_seen = True
#                                 log("poll_bar", time=str(ts), close=last.close, total=len(C))
                    except Exception:
                        pass

                # ---- Position truth ----
#                 net_qty, avg_cost_raw = ib_position_truth(ib, con)

                # ---- Trade-cycle transitions ----
                if prev_net_qty == 0 and net_qty != 0:
                    pass
#                     in_trade_cycle = True
#                     cycle_commission = 0.0
#                     cycle_entry_qty = abs(net_qty)
#                     entry_price = (avg_cost_raw / px_mult) if (avg_cost_raw is not None and px_mult > 0) else (C[-1] if C else None)
#                     cycle_entry_time = now
#                     age_forced_flat_done = False

                if prev_net_qty != 0 and net_qty == 0 and entry_price is not None:
                    pass
#                     exit_px = last_exec_price if last_exec_price is not None else (C[-1] if C else entry_price)
#                     signed = 1 if prev_net_qty > 0 else -1
#                     pc_risk = float(args.risk_ticks) * float(args.tick_size) * float(px_mult)
#                     risk_dollars_total = pc_risk * max(1, cycle_entry_qty)
#                     pnl_dollars = (exit_px - entry_price) * px_mult * signed * max(1, cycle_entry_qty) - cycle_commission
#                     reward_R = (pnl_dollars / risk_dollars_total) if risk_dollars_total > 0 else 0.0

#                     realized_pnl_total += pnl_dollars
#                     realized_pnl_day += pnl_dollars

                    # Equity/equity_hwm from NetLiq if IB sync, else from acct_base + realized
                    if ib_netliq is not None and args.use_ib_pnl:
                        pass
#                         equity = float(ib_netliq)
                    else:
                        pass
#                         equity = float(args.acct_base) + realized_pnl_total
#                     equity_hwm = max(equity_hwm, equity)

#                     risk.day_R += reward_R
                    if reward_R < 0: risk.consec_losses += 1
                    else: risk.consec_losses = 0
                    if risk.consec_losses >= args.max_consec_losses:
                        pass
#                         risk.halted = True

                    if args.loss_graded_cooldown and reward_R < 0 and not risk.halted:
                        pass
#                         base = int(args.strategy_cooldown_sec); mult = 2.0 if risk.consec_losses >= 2 else 1.0
#                         tgt = ct_now() + dt.timedelta(seconds=int(base*mult))
                        if (risk.cool_until is None) or (tgt > risk.cool_until):
                            pass
#                             risk.cool_until = tgt
#                         log("graded_cooldown", base=base, mult=mult, consec=risk.consec_losses, until=str(risk.cool_until))

#                     week_R += reward_R

                    # persistent session state update
#                     day_realized += pnl_dollars
#                     day_peak_realized = max(day_peak_realized, day_realized)
#                     day_state[k]["day_realized"] = day_realized
#                     day_state[k]["day_peak_realized"] = day_peak_realized
#                     save_day_state(day_state)

#                     log("flat_cycle",
#                         pnl_dollars=round(pnl_dollars,2),
#                         R=round(reward_R,3),
#                         comm=round(cycle_commission,2),
#                         qty=max(1, cycle_entry_qty),
#                         dayR=round(risk.day_R,3),
#                         consec_losses=risk.consec_losses,
#                         equity=round(equity,2),
#                         equity_hwm=round(equity_hwm,2),
#                         weekR=round(week_R,3),
#                         day_realized=round(day_realized,2),
#                         day_peak=round(day_peak_realized,2)

                    try:
                        pass
#                         total = 0
                        for _ in range(3):
                            pass
#                             c = cancel_exit_children_for_contract(ib, con); total += c
                            if c == 0: break
#                             ib.sleep(0.25)
#                         log("flat_cancel_children", count=total)
                    except Exception as e:
                        pass
#                         log("flat_cancel_children_err", err=str(e))

#                     in_trade_cycle = False
#                     entry_price = None
#                     current_arm = None
#                     cycle_entry_qty = 0
#                     cycle_commission = 0.0
#                     cycle_entry_time = None
#                     age_forced_flat_done = False

#                 prev_net_qty = net_qty

                # ---- IB P&L: overwrite in-session if enabled ----
                if args.use_ib_pnl:
                    if ib_daily_pnl is not None:
                        pass
#                         day_realized = float(ib_daily_pnl)
#                         day_peak_realized = max(day_peak_realized, day_realized)
#                         day_state[k]["day_realized"] = day_realized
#                         day_state[k]["day_peak_realized"] = day_peak_realized
#                         save_day_state(day_state)
                    if ib_netliq is not None:
                        pass
#                         equity = float(ib_netliq); equity_hwm = max(equity_hwm, equity)

                # ---- OCO-rescue ----
#                 last_px = C[-1] if C else None
                try:
                    if net_qty != 0 and args.place_orders:
                        pass
#                         ensure_protection_orders(ib, con, net_qty, avg_cost_raw, args, last_px, float(args.tick_size), px_mult)
#                         snapshot_orders(ib, con, tag="after_oco_rescue_verify")
                except Exception as e:
                    pass
#                     log("oco_rescue_err", err=str(e))

                # ---- Position-age cap ----
                if (not args.disable_pos_age_cap) and in_trade_cycle and net_qty != 0 and entry_price is not None and cycle_entry_time is not None:
                    pass
#                     elapsed = (now - cycle_entry_time).total_seconds()
                    if elapsed >= max(1, int(args.pos_age_cap_sec)) and not age_forced_flat_done:
                        if last_px is not None and not math.isnan(last_px):
                            pass
#                             side = 1 if net_qty > 0 else -1
#                             pc_risk = float(args.risk_ticks) * float(args.tick_size) * float(px_mult)
#                             uR = 0.0
                            if pc_risk > 0:
                                pass
#                                 uR = ((last_px - entry_price) * px_mult * side * max(1,cycle_entry_qty)) / (pc_risk * max(1,cycle_entry_qty))
                            if uR < float(args.pos_age_minR):
                                pass
#                                 flatten_market(net_qty)
#                                 age_forced_flat_done = True
#                                 risk.cool_until = ct_now() + dt.timedelta(seconds=int(args.strategy_cooldown_sec))
#                                 log("pos_age_forced_flat", uR=round(uR,3), elapsed=int(elapsed))

                # ---- Weekly cap gating ----
                if args.enable_weekly_cap and (week_R <= weekly_cap_R):
                    pass
#                     week_halted = True

                # ---- Exit audit & auto-repair ----
                if net_qty != 0 and args.place_orders:
                    pass
#                     audit = audit_exits(ib, con, net_qty)
                    if not (audit["has_stop"] and audit["has_tp"]):
                        pass
#                         log("exit_audit", has_stop=bool(audit["has_stop"]), has_tp=bool(audit["has_tp"]),
#                             stop=audit["stop_px"], tp=audit["tp_px"])
                        try:
                            pass
#                             ensure_protection_orders(ib, con, net_qty, avg_cost_raw, args,
#                                 last_px, float(args.tick_size), px_mult)
#                             snapshot_orders(ib, con, tag="after_exit_autorepair")
                        except Exception as e:
                            pass
#                             log("exit_autorepair_err", err=str(e))

                # ---- Heartbeat data ----
#                 arm_counts_disp = {}; arm_meanR_disp = {}
                if bandit:
                    for a in enabled_arms:
                        pass
#                         arm_counts_disp[a] = int(getattr(bandit, "arm_w", {}).get(a, 0.0))
#                         arm_meanR_disp[a] = round(getattr(bandit, "arm_meanR", {}).get(a, 0.0), 3)
                else:
                    arm_counts_disp = {a: 0 for a in enabled_arms}
                    arm_meanR_disp = {a: 0.0 for a in enabled_arms}

#                                 in_news_pause = news_guard.in_pause(now)

                # Compute a live sizing snapshot (mirrors determine_order_qty internals for the heartbeat only)
                if args.use_ib_pnl and (ib_netliq is not None):
                    pass
#                     eq_now_hb = float(ib_netliq)
#                     eq_src_hb = "ib_netliq"
                else:
                    pass
#                     eq_now_hb = float(equity)
#                     eq_src_hb = "local_equity"

#                 ladder_qty = equity_ladder_size(eq_now_hb)
#                 rb_qty = risk_budget_size(eq_now_hb)
#                 base_qty_hb = int(clamp(min(ladder_qty, rb_qty), 1, args.max_contracts))
#                 stepdown_qty_hb = apply_hwm_stepdown(base_qty_hb)
#                 mcap_total_hb = margin_cap_qty(eq_now_hb)
#                 desired_total_hb = min(int(clamp(stepdown_qty_hb, 0, args.max_contracts)),
#                                        int(clamp(mcap_total_hb, 0, args.max_contracts)))
                # This is the *recommended total* if flat; suggest_qty remains for backward compatibility
#                 sugg_qty = stepdown_qty_hb


#                 avg_disp = 0.0
                if avg_cost_raw is not None and px_mult > 0:
                    try: avg_disp = (avg_cost_raw / px_mult)
                    except Exception: avg_disp = 0.0

                # ---- Determine state & caps reasons ----
#                 reasons = []
                if risk.day_R <= -abs(args.day_loss_cap_R): reasons.append("dayR_cap")
                if risk.trades >= args.max_trades_per_day: reasons.append("max_trades")
                if risk.halted: reasons.append("risk_halted")
                if args.enable_weekly_cap and week_halted: reasons.append("week_cap")
                if day_realized <= day_guard_dollars: reasons.append("minus5pct_guard")
                # Peak DD guard only after minimum profit achieved
                if args.peak_dd_guard_pct > 0 and (day_peak_realized >= float(args.peak_dd_min_profit)):
                    if day_realized <= day_peak_realized * (1.0 - float(args.peak_dd_guard_pct)):
                        pass
#                         reasons.append("peak_dd_guard")

#                 state = (
#                     else "news" if in_news_pause
#                     else "wait_rt" if (args.require_rt_before_trading and not rt_fresh)
#                     else "cooldown" if (risk.cool_until and now < risk.cool_until)
#                     else "active" if within_session(now, args.trade_start_ct, args.trade_end_ct)
#                     else "sleep"

                if state == "caps" and not in_caps:
                    pass
#                     in_caps = True
#                     log("caps_on", reasons=reasons, day_realized=round(day_realized,2),
#                         day_peak=round(day_peak_realized,2), dayR=round(risk.day_R,3))
                if state != "caps" and in_caps:
                    pass
#                     in_caps = False

#                 emit = False
#                 now_epoch = time.time()
                if _last_state_for_hb != state:
                    pass
#                     emit = True; _last_state_for_hb = state
                elif (now_epoch - _last_hb_emit) >= 1.0:
                    pass
#                     emit = True

                if emit:
                    pass
#                     vwap_disp = None if (vwap is None or math.isnan(vwap.vwap)) else round(vwap.vwap, 2)
#                     l                    log("hb",
#                         src=src,
#                         state=state,
#                         rt_fresh=bool(rt_fresh),
#                         dayR=round(risk.day_R,3),
#                         weekR=round(week_R,3),
#                         week_halted=bool(week_halted and args.enable_weekly_cap),
#                         pos=net_qty,
#                         avg=round(avg_disp, 2),
#                         bars=len(C),
#                         consec_losses=risk.consec_losses,
#                         equity=round(equity,2),
#                         equity_hwm=round(equity_hwm,2),

                        # legacy displayed suggestion
#                         suggest_qty=int(sugg_qty),

                        # sizing snapshot (new)
#                         sizing_snapshot={
                            "eq_src": eq_src_hb,
                            "eq_now": round(eq_now_hb,2),
                            "equity_ladder_qty": int(ladder_qty),
                            "risk_budget_qty": int(rb_qty),
                            "after_hwm_stepdown_qty": int(stepdown_qty_hb),
                            "margin_cap_total": int(mcap_total_hb),
                            "desired_total_after_margin": int(desired_total_hb)
                        },

#                         arms=arm_counts_disp,
#                         armR=arm_meanR_disp,
#                         day_realized=round(day_realized,2),
#                         day_guard=round(day_guard_dollars,2),
#                         session_vwap=vwap_disp

#                         suggest_qty=sugg_qty,

                # If any cap tripped and we are flat  cancel all orders
                if state == "caps" and net_qty == 0:
                    try:
                        pass
#                         c = 0
                        for _ in range(3):
                            pass
#                             c += cancel_active_parent_entries(ib, con)
#                             c += cancel_exit_children_for_contract(ib, con)
                            if c == 0: break
#                             ib.sleep(0.25)
#                         log("cap_sweep_all_orders", count=c)
                    except Exception as e:
                        pass
#                         log("cap_sweep_err", err=str(e))

                # ---- Gate checks before entries ----
#                 in_window = within_session(now, args.trade_start_ct, args.trade_end_ct)
#                 if (not in_window) or in_any_blackout(now, tod_blackouts) \
#                    or (not risk.can_trade(now, int(args.min_seconds_between_entries))) \
#                    or (args.enable_weekly_cap and week_halted) \
#                    or (args.require_rt_before_trading and not rt_fresh) \
                   or (state == "caps"):
#                     ib.sleep(0.5); continue

                if args.require_new_bar_after_start and not startup_bar_seen:
                    pass
#                     ib.sleep(0.5); continue
                if len(C) < 60:
                    pass
#                     ib.sleep(0.5); continue

                # ---- Features / arms ----
#                 close = C[-1]; _adx = adx(H, L, C, 14); _atr = atr(H, L, C, 14)
                _atrp = (_atr / close) if (not math.isnan(_atr) and close>0) else float("nan")
                _bbbw = bbbw(C, 20, 2.0) or float("nan")
                fast = ema(C[-20:], 20); slow = ema(C[-50:], 50)
#                 is_trend = (not math.isnan(_adx)) and _adx >= args.gate_adx and fast > slow
                is_breakout = (len(C) >= 30) and (close >= max(C[-20:]) or close <= min(C[-20:]))

                if net_qty == 0 and not has_active_parent_entry(ib, con):
                    pass
#                     cand = []
                    if "trend" in enabled_arms and is_trend and (not math.isnan(_atrp)) and _atrp >= args.gate_atrp:
                        pass
#                         cand.append("trend")
                    if "breakout" in enabled_arms and is_breakout and (not math.isnan(_bbbw)) and _bbbw >= args.gate_bbbw:
                        pass
#                         cand.append("breakout")
#                     chosen = None
                    if len(cand) == 1 or args.bandit == "fixed":
                        pass
#                         chosen = cand[0] if cand else None
                    elif len(cand) >= 2:
                        pass
#                         x = feature_vector(H, L, C)
                        if bandit:
                            pass
#                             chosen, probs = bandit.choose(x, cand, sample=args.sample_action)
                            if args.learn_log:
                                pass
#                                 learn_log(f"{utc_now_str()} choose cand={cand} probs={json.dumps(probs)} chosen={chosen}")
                        else:
                            pass
#                             chosen = "trend" if "trend" in cand else cand[0] if cand else None
                    if chosen:
                        pass
#                         go_long = True
                        if chosen == "trend": go_long = fast >= slow
                        elif chosen == "breakout": go_long = close >= max(C[-20:])
#                         place_bracket(go_long, close, last_bar_ts, net_qty)
#                         current_arm = chosen

#                 ib.sleep(0.5)

            except Exception as e:
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-4000:]
#                 log("loop_error", err=str(e), tb=tb)
#                 ib.sleep(1.0)

    except KeyboardInterrupt:
        pass
#         print("INFO [CTRL-C] Shutting down...")
    finally:
        try:
            pass
#             day_state[k]["day_realized"] = day_realized
#             day_state[k]["day_peak_realized"] = day_peak_realized
#             save_day_state(day_state)
        except Exception: pass
        try: save_state()
        except Exception: pass
        try:
            if 'news_guard' in locals() and getattr(news_guard, "enabled", False):
                try: ib.newsBulletinEvent -= news_guard._on_bulletin
                except Exception: pass
                try: ib.cancelNewsBulletins()
                except Exception: pass
        except Exception: pass
        try:
            pass
#             sent = getattr(getattr(ib, "client", None), "bytesSent", 0) or 0
#             recv = getattr(getattr(ib, "client", None), "bytesReceived", 0) or 0
            print(f"Disconnecting from {args.host}:{args.port}, {sent/1024:.0f} kB sent, {recv/1024:.0f} kB received.")
        except Exception: pass
        try: ib.disconnect(); print("Disconnected.")
        except Exception: pass

if __name__ == "__main__":
    pass
#     main()




