# pt_healthcheck.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from typing import Optional, Tuple

from ib_insync import IB, util, Future, Stock


@dataclass
class HealthResult:
    ok_api: bool
    ok_hist: bool
    ok_fresh: bool
    ok_warmup: bool
    ok_mktdata: Optional[bool]  # None if not tested
    bars: int
    last_bar_iso: str
    reason: str  # empty if OK


def _now_local_iso() -> str:
    # local time with offset
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_healthcheck(
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 9901,
    # What to pull for history (simple default = SPY 1-min, because it is easy to qualify)
    # You can switch to ES by changing use_es=True and setting the expiry properly.
    use_es: bool = False,
    es_expiry: str = "202603",
    duration: str = "2 D",
    bar_size: str = "1 min",
    use_rth: bool = False,
    min_bars_required: int = 300,
    max_staleness_seconds: int = 600,  # 10 minutes
    test_market_data: bool = False,
) -> HealthResult:
    util.startLoop()
    ib = IB()

    ok_api = ok_hist = ok_fresh = ok_warmup = False
    ok_mktdata: Optional[bool] = None
    bars_n = 0
    last_bar_iso = ""
    reason = ""

    try:
        ib.connect(host, port, clientId=client_id, timeout=5)
        ok_api = ib.isConnected()
        if not ok_api:
            return HealthResult(False, False, False, False, None, 0, "", "API not connected")

        # Contract
        if use_es:
            contract = Future("ES", es_expiry, "CME")
            ib.qualifyContracts(contract)
            what = "TRADES"
        else:
            contract = Stock("SPY", "SMART", "USD")
            ib.qualifyContracts(contract)
            what = "TRADES"

        # Historical bars
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what,
            useRTH=use_rth,
            formatDate=1,
            keepUpToDate=False,
        )

        bars_n = len(bars)
        if bars_n <= 0:
            return HealthResult(True, False, False, False, None, 0, "", "No historical bars returned")

        # last bar time
        last_dt = bars[-1].date
        # ib_insync sometimes gives datetime or string; normalize
        if isinstance(last_dt, str):
            last_bar_iso = last_dt
            # can't compute freshness reliably; treat as unknown -> fail fresh with reason
            return HealthResult(True, True, False, False, None, bars_n, last_bar_iso,
                                "Last bar timestamp is string; cannot compute freshness")
        else:
            last_bar_iso = last_dt.astimezone().isoformat(timespec="seconds")

        ok_hist = True

        # Freshness check
        now = datetime.now().astimezone()
        delta = (now - last_dt.astimezone()).total_seconds()
        ok_fresh = (delta >= 0) and (delta <= max_staleness_seconds)

        # Warm-up check
        ok_warmup = bars_n >= min_bars_required

        # Optional market data ping
        if test_market_data:
            try:
                t = ib.reqMktData(contract)
                ib.sleep(2)
                ok_mktdata = (t.bid is not None) or (t.ask is not None) or (t.last is not None)
                ib.cancelMktData(contract)
            except Exception as e:
                ok_mktdata = False
                reason = f"Market data test failed: {e}"

        # Final reason
        if not ok_fresh:
            reason = f"History stale: last bar {int(delta)}s old"
        elif not ok_warmup:
            reason = f"Warmup insufficient: bars={bars_n} need>={min_bars_required}"
        elif ok_mktdata is False and test_market_data:
            reason = reason or "Market data not flowing"
        else:
            reason = ""

        return HealthResult(
            ok_api=ok_api,
            ok_hist=ok_hist,
            ok_fresh=ok_fresh,
            ok_warmup=ok_warmup,
            ok_mktdata=ok_mktdata,
            bars=bars_n,
            last_bar_iso=last_bar_iso,
            reason=reason,
        )

    except Exception as e:
        return HealthResult(ok_api, False, False, False, None, bars_n, last_bar_iso, f"Exception: {e}")
    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:
            pass


def write_health_files(res: HealthResult, out_dir: str = r".\run") -> str:
    """
    Writes run\\health.json.

    Two readiness levels are recorded:
      - ready_next_go: API + history + warmup (last-session bars are acceptable when the market is closed)
      - live_ok_go   : ready_next_go + fresh bars (relevant when the market is open)

    For backward compatibility, ready_go is set to live_ok_go.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "health.json")

    payload = asdict(res)
    payload["ts_local"] = _now_local_iso()

    ready_next = bool(res.ok_api and res.ok_hist and res.ok_warmup)
    live_ok = bool(ready_next and res.ok_fresh)

    payload["ready_next_go"] = ready_next
    payload["live_ok_go"] = live_ok

    # Backward-compatible field (older dashboards may read this)
    payload["ready_go"] = live_ok

    # Prefer a readable reason message
    if ready_next and not live_ok:
        payload["ready_reason"] = res.reason or "Live bars not fresh (market likely closed)"
    else:
        payload["ready_reason"] = "" if live_ok else (res.reason or "NO-GO")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path

if __name__ == "__main__":
    # Defaults: SPY history, 2 days of 1-min bars, warmup >= 300 bars, stale threshold 10 min
    res = run_healthcheck(
        host="127.0.0.1",
        port=4002,
        client_id=9901,
        use_es=False,
        min_bars_required=300,
        max_staleness_seconds=600,
        test_market_data=False,
    )
    path = write_health_files(res)

go_next = bool(res.ok_api and res.ok_hist and res.ok_warmup)
go_live = bool(go_next and res.ok_fresh)

status_next = "READY_NEXT=GO" if go_next else "READY_NEXT=NO-GO"
status_live = "LIVE_OK=GO" if go_live else "LIVE_OK=NO-GO"

print(f"{status_next} | {status_live} - {res.reason or 'OK'}")
print("bars=", res.bars, "last=", res.last_bar_iso)
print("wrote:", path)
