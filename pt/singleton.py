from __future__ import annotations

# pt/singleton.py
#
# Hard singleton enforcement on Windows:
#   1) Global named mutex (robust across processes)
#   2) File lock fallback (msvcrt 1-byte nonblocking)
#
# Behavior:
#   - If PT_FORCE_SINGLETON is truthy => skip enforcement (returns None)
#   - If PT_DEBUG_SINGLETON is truthy => prints diagnostics to stderr
#
# Call acquire_or_exit() as early as possible and keep the returned handle alive
# for the lifetime of the process.

import os
import sys
from typing import Optional, Tuple, Union

ERROR_ALREADY_EXISTS = 183


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _dbg(msg: str) -> None:
    if _env_truthy("PT_DEBUG_SINGLETON"):
        try:
            sys.stderr.write(msg.rstrip() + "\n")
            sys.stderr.flush()
        except Exception:
            pass


def lock_path(app_name: str = "paper_trader") -> str:
    # module lives at <root>\pt\singleton.py
    base_dir = os.path.dirname(os.path.abspath(__file__))  # ...\pt
    root = os.path.dirname(base_dir)                       # ...\project_root
    os.makedirs(os.path.join(root, "run"), exist_ok=True)
    if app_name == "paper_trader":
        fname = "paper_trader.lock"
    else:
        fname = f"{app_name}.lock"
    return os.path.join(root, "run", fname)


class _WinMutexHandle:
    def __init__(self, name: str, handle: int):
        self.name = name
        self.handle = handle

    def close(self) -> None:
        try:
            import ctypes
            if self.handle:
                ctypes.windll.kernel32.CloseHandle(self.handle)
        except Exception:
            pass
        self.handle = 0


def _acquire_windows_mutex(app_name: str) -> Tuple[_WinMutexHandle, bool]:
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.CreateMutexW.restype = ctypes.c_void_p
    k32.GetLastError.restype = ctypes.c_uint32

    mutex_name = f"Global\\{app_name.upper()}_SINGLETON"
    h = k32.CreateMutexW(None, False, mutex_name)
    if not h:
        raise RuntimeError(f"CreateMutexW failed for {mutex_name!r}")
    already = (k32.GetLastError() == ERROR_ALREADY_EXISTS)
    return _WinMutexHandle(mutex_name, int(h)), already


def _acquire_file_lock(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a+b")
    try:
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)  # raises OSError if locked
        try:
            f.seek(0)
            f.truncate(0)
            f.write(str(os.getpid()).encode("ascii", errors="ignore"))
            f.flush()
        except Exception:
            pass
        return f
    except OSError:
        try:
            f.close()
        except Exception:
            pass
        raise
    except Exception:
        try:
            f.close()
        except Exception:
            pass
        raise


def _exit_rc2(reason: str) -> None:
    try:
        sys.stderr.write(reason.rstrip() + "\n")
        sys.stderr.flush()
    except Exception:
        pass
    raise SystemExit(2)


def acquire_or_exit(app_name: str = "paper_trader", *, force_env: str = "PT_FORCE_SINGLETON"):
    if _env_truthy(force_env):
        _dbg(f"[SINGLETON] FORCE enabled via {force_env}=1; skipping enforcement.")
        return None

    pid = os.getpid()
    ppid = os.getppid() if hasattr(os, "getppid") else None

    # 1) Windows mutex
    try:
        mtx, already = _acquire_windows_mutex(app_name)
        _dbg(f"[SINGLETON] mutex acquired name={mtx.name} pid={pid} ppid={ppid} already_exists={already}")
        if already:
            mtx.close()
            _exit_rc2(f"paper_trader: another instance detected (mutex already exists): {mtx.name}")
    except SystemExit:
        raise
    except Exception as e:
        _dbg(f"[SINGLETON] mutex unavailable ({type(e).__name__}: {e}); falling back to file lock")

    # 2) File lock fallback (or supplemental)
    lp = lock_path(app_name)
    try:
        fh = _acquire_file_lock(lp)
        _dbg(f"[SINGLETON] file lock acquired path={lp} pid={pid} ppid={ppid}")
        # Keep BOTH handles alive: return a tuple if mutex succeeded too
        try:
            # if mutex succeeded above, it would have returned; so only file lock here
            return fh
        except Exception:
            return fh
    except OSError:
        _exit_rc2(f"paper_trader: another instance detected (lock busy): {lp}")
    except Exception as e:
        _exit_rc2(f"paper_trader: singleton enforcement failed: {type(e).__name__}: {e}")


def acquire_paper_trader_lock():
    return acquire_or_exit("paper_trader")
