import os
import sys
import time
import subprocess
from datetime import datetime


# ==========================================================
#  WATCHDOG (RESTART ON CRASHES, NO DUPES, WITH BACKOFF)
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR  = os.path.join(BASE_DIR, "run")
os.makedirs(RUN_DIR, exist_ok=True)

SHUTDOWN_FLAG = os.path.join(RUN_DIR, "SHUTDOWN.flag")
LOG_PATH      = os.path.join(RUN_DIR, "watchdog_simple.log")
WD_LOCK       = os.path.join(RUN_DIR, "watchdog_single.lock")

PYTHON_EXE = sys.executable
TRADER_PY  = os.path.join(BASE_DIR, "paper_trader.py")


def shutdown_requested() -> bool:
    return os.path.exists(SHUTDOWN_FLAG)


def _is_pid_running_windows(pid: int) -> bool:
    try:
        out = subprocess.check_output(
            ["cmd", "/c", f'tasklist /FI "PID eq {pid}"'],
            text=True,
            stderr=subprocess.DEVNULL
        )
        return str(pid) in (out or "")
    except Exception:
        return False


def _acquire_watchdog_lock() -> None:
    # Ensure only ONE watchdog_single runs
    try:
        fd = os.open(WD_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        other_pid = 0
        try:
            with open(WD_LOCK, "r", encoding="utf-8") as f:
                other_pid = int((f.read() or "0").strip() or "0")
        except Exception:
            other_pid = 0

        if other_pid and _is_pid_running_windows(other_pid):
            raise SystemExit(f"[WD] Another watchdog_single is already running (pid={other_pid}). Exiting.")

        # stale lock: remove and retry once
        try:
            os.remove(WD_LOCK)
        except Exception:
            raise SystemExit("[WD] watchdog lock exists and could not be removed. Exiting.")

        fd = os.open(WD_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

    os.write(fd, str(os.getpid()).encode("utf-8"))
    os.close(fd)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def main():
    _acquire_watchdog_lock()

    log("[WD] Watchdog started.")
    log(f"[WD] Trader target: {TRADER_PY}")
    log("--------------------------------------------------")

    crash_streak = 0

    while True:
        if shutdown_requested():
            log("[WD] Shutdown flag detected. Exiting watchdog.")
            break

        log("[WD] Launching paper_trader...")
        proc = subprocess.Popen([PYTHON_EXE, TRADER_PY])
        rc = proc.wait()

        log(f"[WD] paper_trader exited (rc={rc})")

        if shutdown_requested():
            log("[WD] Shutdown requested. Exiting.")
            break

        # rc=0 = clean exit => DO NOT restart (prevents thrash)
        if rc == 0:
            log("[WD] Trader exited cleanly (rc=0). Not restarting.")
            break

        # rc=2 = singleton guard says another instance exists
        if rc == 2:
            log("[WD] Trader reports another instance exists (rc=2). Exiting watchdog to prevent thrash.")
            break

        # Restart on crashes / any other non-zero rc
        crash_streak += 1
        delay = min(2 * (2 ** (crash_streak - 1)), 60)
        log(f"[WD] Restarting trader in {delay:.0f} seconds (crash_streak={crash_streak})...")
        time.sleep(delay)

    log("[WD] Watchdog stopped.")


if __name__ == "__main__":
    main()
