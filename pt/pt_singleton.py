# pt_singleton.py
# ------------------------------------------------------------
# Safe single-instance guard with stale-lock recovery (Windows)
# ------------------------------------------------------------

import os
import sys
import atexit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR  = os.path.join(BASE_DIR, "run")
os.makedirs(RUN_DIR, exist_ok=True)

LOCK_PATH    = os.path.join(RUN_DIR, "paper_trader.lock")
PIDLOCK_PATH = os.path.join(RUN_DIR, "paper_trader.pidlock")


def _pid_alive(pid: int) -> bool:
    """Return True if PID exists (Windows-safe)."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        return False


def acquire_or_exit():
    """
    Acquire singleton lock for paper_trader.

    - If another live PID holds it -> exit(2)
    - If stale lock -> clear and continue
    """

    # --- If pidlock exists, check if real process is alive ---
    if os.path.exists(PIDLOCK_PATH):
        try:
            old_pid = int(open(PIDLOCK_PATH).read().strip())
        except Exception:
            old_pid = None

        if old_pid and _pid_alive(old_pid):
            print(f"paper_trader: another instance is running (pid={old_pid}); exiting.")
            sys.exit(2)

        # stale pidlock -> remove
        try:
            os.remove(PIDLOCK_PATH)
        except Exception:
            pass

    # stale lock file -> remove
    if os.path.exists(LOCK_PATH):
        try:
            os.remove(LOCK_PATH)
        except Exception:
            pass

    # --- Create fresh lock + pidlock ---
    with open(PIDLOCK_PATH, "w") as f:
        f.write(str(os.getpid()))

    with open(LOCK_PATH, "w") as f:
        f.write("running")

    # --- Cleanup on exit ---
    def _cleanup():
        for p in (LOCK_PATH, PIDLOCK_PATH):
            try:
                os.remove(p)
            except Exception:
                pass

    atexit.register(_cleanup)

    return True
