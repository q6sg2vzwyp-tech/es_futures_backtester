# run_trader_supervisor.py
# Minimal supervisor for paper_trader.py with ES autoroll (quarterly),
# daily 08:15 CT restart, crash relaunch, and simple logs.

from __future__ import annotations

import subprocess
import threading
import time
from datetime import date, datetime
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# ========= CONFIG =========
PYTHON = r"C:\Users\owner\AppData\Local\Programs\Python\Python313\python.exe"
WORKDIR = Path(__file__).resolve().parent
CHILD = "paper_trader.py"

HOST = "127.0.0.1"
PORT = 7497
CLIENT_ID = 29  # pick an id not used elsewhere (29 matches your last run)

# ES contract universe
SYMBOL = "ES"
EXCHANGE = "CME"
CURRENCY = "USD"
ROLL_WARN_DAYS = 7  # roll to next quarterly if <= this many days to expiry

# Fallback conId if autoroll fails
CONID_FALLBACK = 637533641  # (ESU5 in your logs) update when needed

# Daily restart time (assumes host clock is CT; adjust if needed)
RESTART_HH_MM = (8, 15)

# paper_trader args you like (mirrors your last good run)
CHILD_ARGS = [
    "--tp-ladder",
    "1.0R:50,2.0R:50",
    "--place-orders",
    "--debug-regime",
    "--enable-arms",
    "trend,breakout",
    "--persist-chooser",
    "--chooser-model-path",
    str(WORKDIR / "models" / "linucb_state.json"),
    "--rth-only",
    "--one-trade-at-a-time",
    "--strategy-cooldown-sec",
    "60",
    "--debounce-one-bar",
    "--daily-summary",
    "--shadow-enable",
    "--stall-secs",
    "30",
    # log file per-day (paper_trader expands %Y%m%d):
    "--log-file",
    str(WORKDIR / "logs" / "runtime_%Y%m%d.txt"),
]

LOGS_DIR = WORKDIR / "logs"
MODELS_DIR = WORKDIR / "models"
ACTIVE_CONID_FILE = MODELS_DIR / "active_conid.txt"

HEARTBEAT_SEC = 15
CRASH_COOLDOWN_SEC = 5


# ========= UTILS =========
def now_ct():
    # assuming machine runs CT; if not, replace with zoneinfo("America/Chicago")
    return datetime.now()


def sup(msg: str):
    print(f"[SUP] {now_ct():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def needs_daily_restart(last_restart_date: str | None) -> bool:
    h, m = RESTART_HH_MM
    n = now_ct()
    if (n.hour > h) or (n.hour == h and n.minute >= m):
        today = n.strftime("%Y-%m-%d")
        return today != last_restart_date
    return False


def ensure_dirs():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def read_active_conid_fallback() -> int | None:
    try:
        if ACTIVE_CONID_FILE.exists():
            return int(ACTIVE_CONID_FILE.read_text().strip())
    except Exception as e:

        logger.debug('Swallowed exception in archive\run_ppt_trader_supervisor.py: %s', e)
    return None


def write_active_conid(conid: int):
    try:
        ACTIVE_CONID_FILE.write_text(str(conid), encoding="utf-8")
    except Exception as e:
        sup(f"WARNING: could not store active conId: {e!r}")


# ========= AUTOROLL via IBAPI =========
def resolve_es_conid_via_ib() -> tuple[int | None, dict]:
    """
    Connects to IB Gateway/TWS, requests ES futures chain on CME,
    filters quarterly contracts (H=03, M=06, U=09, Z=12),
    picks front ≥ today; if days_to_exp <= ROLL_WARN_DAYS pick next.
    Returns (conId, info_dict) or (None, {"error": "..."}).
    """
    try:
        from ibapi.client import EClient
        from ibapi.contract import Contract
        from ibapi.wrapper import EWrapper

        class App(EWrapper, EClient):
            def __init__(self):
                EClient.__init__(self, self)
                self.details = []
                self._evt = threading.Event()
                self._next_req = 11000

            def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
                # suppress console spam; could log if desired
                pass

            def contractDetails(self, reqId, contractDetails):
                self.details.append(contractDetails)

            def contractDetailsEnd(self, reqId):
                self._evt.set()

        app = App()
        app.connect(HOST, PORT, CLIENT_ID)
        t = threading.Thread(target=app.run, daemon=True)
        t.start()

        base = Contract()
        base.symbol = SYMBOL
        base.secType = "FUT"
        base.exchange = EXCHANGE
        base.currency = CURRENCY

        req_id = app._next_req
        app._next_req += 1
        app._evt.clear()
        app.details.clear()
        app.reqContractDetails(req_id, base)

        end_by = time.time() + 8.0
        while time.time() < end_by:
            if app._evt.wait(timeout=0.05):
                break

        try:
            app.disconnect()
        except Exception as e:

            logger.debug('Swallowed exception in archive\run_ppt_trader_supervisor.py: %s', e)
        ds = list(app.details)
        if not ds:
            return None, {"error": "no_details_returned"}

        def is_quarterly(month: int) -> bool:
            return month in (3, 6, 9, 12)

        today = now_ct().date()
        candidates = []
        for cd in ds:
            c = cd.contract
            raw = c.lastTradeDateOrContractMonth or ""  # e.g. "20251219"
            if len(raw) < 6:
                continue
            y = int(raw[:4])
            m = int(raw[4:6])
            d = int(raw[6:8]) if len(raw) >= 8 else 15
            if not is_quarterly(m):
                continue
            try:
                dte = date(y, m, d)
            except Exception as e:

                logger.debug('Swallowed exception in archive\run_ppt_trader_supervisor.py: %s', e)
 continue
            if dte >= today:
                candidates.append((c.conId, dte, c.localSymbol or "", f"{y}-{m:02d}-{d:02d}"))

        candidates.sort(key=lambda x: x[1])
        if not candidates:
            return None, {"error": "no_quarterlies_after_today"}

        front_conid, front_dt, front_local, front_dt_str = candidates[0]
        days_to_exp = (front_dt - today).days
        if days_to_exp <= ROLL_WARN_DAYS and len(candidates) >= 2:
            next_conid, next_dt, next_local, next_dt_str = candidates[1]
            return next_conid, {
                "mode": "ROLLED_NEXT",
                "from": {
                    "conId": front_conid,
                    "local": front_local,
                    "exp": front_dt_str,
                },
                "to": {"conId": next_conid, "local": next_local, "exp": next_dt_str},
                "days_to_exp": days_to_exp,
            }
        return front_conid, {
            "mode": "FRONT_MONTH",
            "front": {"conId": front_conid, "local": front_local, "exp": front_dt_str},
            "days_to_exp": days_to_exp,
        }

    except Exception as e:
        return None, {"error": f"ibapi_exception: {e!r}"}


# ========= CHILD PROC =========
def build_cmd(conid: int) -> list[str]:
    base = [
        PYTHON,
        "-X",
        "faulthandler",
        "-u",
        CHILD,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--clientId",
        str(CLIENT_ID),
        "--conId",
        str(conid),
    ]
    return base + CHILD_ARGS


def main():
    ensure_dirs()
    print("\n============================")
    print("Starting ES Supervisor...")
    print("============================")
    sup(f"Workdir={WORKDIR}")
    sup(f"Python={PYTHON}")

    last_restart_date = None
    child = None
    child_log_handle = None

    while True:
        # daily restart
        if needs_daily_restart(last_restart_date):
            if child and child.poll() is None:
                sup("Daily restart: stopping child.")
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
            if child_log_handle:
                try:
                    child_log_handle.close()
                except Exception as e:

                    logger.debug('Swallowed exception in archive\run_ppt_trader_supervisor.py: %s', e)
            child = None
            child_log_handle = None
            last_restart_date = now_ct().strftime("%Y-%m-%d")

        # ensure child running
        if child is None or child.poll() is not None:
            if child is not None and child.poll() is not None:
                sup(f"Child exited rc={child.returncode}; relaunch in {CRASH_COOLDOWN_SEC}s.")
                if child_log_handle:
                    try:
                        child_log_handle.flush()
                        child_log_handle.close()
                    except Exception as e:

                        logger.debug('Swallowed exception in archive\run_ppt_trader_supervisor.py: %s', e)
                time.sleep(CRASH_COOLDOWN_SEC)

            # autoroll
            conid, info = resolve_es_conid_via_ib()
            if conid is None:
                last_active = read_active_conid_fallback()
                chosen = last_active if last_active is not None else CONID_FALLBACK
                sup(f"AUTOROLL: FAILED ({info.get('error','?')}); using fallback conId={chosen}.")
            else:
                mode = info.get("mode", "FRONT_MONTH")
                if mode == "ROLLED_NEXT":
                    frm = info.get("from", {})
                    to = info.get("to", {})
                    sup(
                        f"AUTOROLL: ROLL (≤{ROLL_WARN_DAYS}d). {frm.get('local','?')}→{to.get('local','?')} "
                        f"({frm.get('exp','?')}→{to.get('exp','?')}) conId {frm.get('conId')}→{to.get('conId')}."
                    )
                else:
                    fm = info.get("front", {})
                    sup(
                        f"AUTOROLL: FRONT {fm.get('local','?')} (exp {fm.get('exp','?')}, "
                        f"dte={info.get('days_to_exp','?')}) conId={fm.get('conId')}."
                    )
                chosen = conid

            write_active_conid(chosen)

            # launch
            cmd = build_cmd(chosen)
            sup("Launching child:")
            sup("  " + " ".join(cmd))
            log_path = LOGS_DIR / f"child_{now_ct():%Y%m%d_%H%M%S}.txt"
            child_log_handle = open(log_path, "w", encoding="utf-8")
            child = subprocess.Popen(
                cmd, cwd=WORKDIR, stdout=child_log_handle, stderr=subprocess.STDOUT
            )
            sup(f"Child log: {log_path}")

        # heartbeat
        state = "RUNNING" if child.poll() is None else "STOPPED"
        sup(f"child={state}")
        time.sleep(HEARTBEAT_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sup("Supervisor interrupted; exiting.")
