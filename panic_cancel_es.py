#!/usr/bin/env python3
# Cancel all non-filled ES orders for the current IB client session—even if Inactive/PendingCancel.
from ib_insync import *
import time, sys, datetime as dt

HOST, PORT, CLIENT_ID = "127.0.0.1", 7497, 111
LOCAL_SYMBOL = "ESZ5"   # adjust if needed (matches what you trade)

ACTIVE_OR_SEMI = {"ApiPending","PendingSubmit","PendingCancel","PreSubmitted","Submitted","Inactive"}

def status_of(t): 
    try: return (t.orderStatus.status or "").strip()
    except: return ""

def note(t): 
    try: return f"#{t.order.orderId} {t.order.action} {t.order.orderType} {getattr(t.order,'lmtPrice',getattr(t.order,'auxPrice',''))}"
    except: return "?"

def main():
    ib = IB()
    print("[CONNECT] ->", HOST, PORT, CLIENT_ID)
    ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=30)

    cds = ib.reqContractDetails(Future(localSymbol=LOCAL_SYMBOL, exchange='CME'))
    if not cds:
        print(f"[ERR] Could not resolve {LOCAL_SYMBOL} on CME."); return
    con = cds[0].contract
    ib.qualifyContracts(con)
    print(f"[CONTRACT] {con.localSymbol} conId={con.conId}")

    # Fresh snapshot
    ib.reqOpenOrders(); ib.sleep(0.2)

    cancels = 0
    def try_cancel(x, why):
        nonlocal cancels
        try:
            st = status_of(x)
            if st in ("Filled","Cancelled"): 
                return
            # ALWAYS attempt cancel; swallow 161 “Unable to cancel” etc.
            tgt = x.order if hasattr(x,'order') else x
            oid = getattr(tgt, 'orderId', None)
            ib.cancelOrder(tgt)
            print(f"[CANCEL] sent {why} {oid} st={st}")
            cancels += 1
        except Exception as e:
            print(f"[CANCEL-WARN] {why}: {e}")

    # openTrades() + trades() coverage
    for t in list(ib.openTrades()):
        if getattr(t.contract,"conId",None) == con.conId and status_of(t) not in ("Filled","Cancelled"):
            try_cancel(t, "openTrades")

    for t in list(ib.trades()):
        if getattr(t.contract,"conId",None) == con.conId and status_of(t) not in ("Filled","Cancelled"):
            try_cancel(t, "trades")

    # Give IB a moment to process
    ib.sleep(0.5)
    ib.reqOpenOrders(); ib.sleep(0.5)

    # Check if anything still active-ish
    left = []
    for t in ib.openTrades():
        if getattr(t.contract,"conId",None) == con.conId and status_of(t) in ACTIVE_OR_SEMI:
            left.append(note(t))
    if left:
        print("[LEFTOVER]", left)
        print("[GLOBAL] Issuing reqGlobalCancel() as last resort (this session’s orders).")
        try:
            ib.client.reqGlobalCancel()
        except Exception as e:
            print("[GLOBAL-WARN]", e)
        ib.sleep(1.0)

    print(f"[DONE] Cancel attempts: {cancels}")
    ib.disconnect()

if __name__ == "__main__":
    main()
