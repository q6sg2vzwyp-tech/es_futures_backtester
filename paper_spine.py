#!/usr/bin/env python3

paper_spine.py — Minimal “Stable Paper Mode” spine runner

Goal: deterministic boot + IB connect + contract resolve + market data + heartbeat + clean shutdown.
- NO order placement
- NO learners/shadow/OCO
- Uses existing ib_core where possible

import os
import time
import json
import argparse
import datetime as dt
from typing import Optional

# Prefer existing core modules if present
from ib_core import connect_ib, resolve_contract

from ib_insync import IB

RUN_DIR = os.path.abspath(r".\run")
HB_TXT = os.path.join(RUN_DIR, "heartbeat.txt")
HB_KV = os.path.join(RUN_DIR, "heartbeat_kv.txt")
SHUTDOWN_FLAG = os.path.join(RUN_DIR, "SHUTDOWN.flag")

def utc_now_str() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def jlog(evt: str, **fields):
    payload = {"ts": utc_now_str(), "evt": evt}
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False))

def write_hb_kv(kv: dict):
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(HB_KV, "w", encoding="utf-8") as f:
        for k, v in kv.items():
            f.write(f"{k}={v}\n")
    with open(HB_TXT, "w", encoding="utf-8") as f:
        f.write(json.dumps(kv, ensure_ascii=False) + "\n")

def parse_args():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002)
    p.add_argument("--clientId", type=int, default=1111)
    p.add_argument("--force_delayed", action="store_true", help="Use delayed market data type (3).")
    p.add_argument("--paper_abort", action="store_true", default=True, help="Abort if managed account isn't DU* (paper).")
    p.add_argument("--hb_sec", type=float, default=1.0, help="Heartbeat write interval.")
    # pass-through args used by ib_core
    p.add_argument("--symbol", default="ES")
    p.add_argument("--exchange", default="CME")
    p.add_argument("--currency", default="USD")
    p.add_argument("--contract_mode", default="expiry", choices=["expiry","front_month","conId"])
    p.add_argument("--expiry", default=None)
    p.add_argument("--conId", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)

    hb = {
        "state": "-",
        "idle_reason": "starting",
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
        "orders_disabled_paper_safety": True,  # spine never trades
    }
    write_hb_kv(hb)
    jlog("spine_start", host=args.host, port=args.port, clientId=args.clientId)

    class _Logger:
        def __call__(self, evt: str, **kw):
            jlog(evt, **kw)
    logger = _Logger()

    hb["idle_reason"] = "connecting"
    write_hb_kv(hb)
    ib: IB = connect_ib(args, logger)
    jlog("connected", clientId=args.clientId)

    acct = None
    try:
        accts = ib.managedAccounts()
        acct = (accts[0] if accts else None)
        jlog("managedAccounts", accounts=accts)
    except Exception as e:
        jlog("managedAccounts_err", err=str(e))

    if args.paper_abort and acct and (not str(acct).upper().startswith("DU")):
        hb["idle_reason"] = "paper_abort"
        write_hb_kv(hb)
        raise RuntimeError(f"SAFETY ABORT: Non-paper account detected: {acct}")

    try:
        ib.reqMarketDataType(3 if args.force_delayed else 1)
        hb["rt_enabled"] = (not args.force_delayed)
        hb["rt_status"] = "ok" if not args.force_delayed else "disabled"
        write_hb_kv(hb)
        jlog("md_type", delayed=bool(args.force_delayed))
    except Exception as e:
        jlog("md_type_err", err=str(e))

    hb["idle_reason"] = "resolving_contract"
    write_hb_kv(hb)
    con = resolve_contract(ib, args, logger)
    jlog("contract_ok", symbol=getattr(con, "symbol", None), conId=getattr(con, "conId", None))

    ticker = None
    last_tick_ts: Optional[float] = None

    try:
        ticker = ib.reqMktData(con, "", False, False)
        jlog("mktdata_subscribed")
    except Exception as e:
        jlog("mktdata_subscribe_err", err=str(e))

    hb["idle_reason"] = "running"
    hb["state"] = "active"
    write_hb_kv(hb)

    last_hb = 0.0
    try:
        while True:
            if os.path.exists(SHUTDOWN_FLAG):
                jlog("shutdown_flag_seen", path=SHUTDOWN_FLAG)
                break

            try:
                if ticker is not None:
                    px = None
                    if getattr(ticker, "last", None) is not None:
                        px = float(ticker.last)
                    elif getattr(ticker, "close", None) is not None:
                        px = float(ticker.close)
                    if px is not None and px == px:
                        last_tick_ts = time.time()
            except Exception:
                pass

            now = time.time()
            if now - last_hb >= max(0.2, float(args.hb_sec)):
                last_hb = now
                hb["rt_age_sec"] = None if last_tick_ts is None else round(now - last_tick_ts, 3)
                write_hb_kv(hb)
                jlog("hb", **hb)

            ib.sleep(0.2)
    finally:
        try:
            if ticker is not None:
                ib.cancelMktData(con)
        except Exception:
            pass
        try:
            ib.disconnect()
        except Exception:
            pass
        hb["state"] = "stopped"
        hb["idle_reason"] = "stopped"
        write_hb_kv(hb)
        jlog("spine_exit")

if __name__ == "__main__":
    main()
