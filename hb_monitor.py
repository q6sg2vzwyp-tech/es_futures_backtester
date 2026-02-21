#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hb_monitor.py (v3.21.0)

Fixes / Hardening:
- NEVER exit on an unexpected exception during refresh; instead:
    * show a visible error banner on screen
    * append full traceback to logs/hb_monitor_error.log
    * sleep briefly and keep running
  This prevents the "opens then immediately exits" behavior (especially when spawned
  in a separate window via watchdog where the traceback flashes and the window closes).

- If --alt-screen is enabled, force ANSI clear (do NOT call cls). Prevents the
  “blank black screen with cursor” behavior on some Windows Terminal hosts.

- Windows lock-safe heartbeat reads:
    * Use CreateFileW with FILE_SHARE_DELETE so this monitor will NOT block the bot's
      atomic heartbeat updates (os.replace) on Windows.

- Singleton guard:
    * By default, refuses to start if another hb_monitor instance is already running.
      Override with --no-singleton.

Usage:
  python hb_monitor.py
  python hb_monitor.py --alt-screen --interval 1.0
  python hb_monitor.py --ansi --interval 1.0
"""

import csv
import json
import os
import sys
import time
import math
import datetime as dt
import traceback
import atexit
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path




# ---------------------------
# Singleton guard (Windows + file lock)
# ---------------------------
HB_MUTEX_NAME = r"Local\es_futures_backtester_hb_monitor"
HB_LOCK_PATH_DEFAULT = os.path.join("run", "hb_monitor.lock")


def _acquire_windows_mutex(name: str):
    if os.name != "nt":
        return 1
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        CreateMutexW = k32.CreateMutexW
        CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
        CreateMutexW.restype = wintypes.HANDLE
        GetLastError = k32.GetLastError
        GetLastError.restype = wintypes.DWORD
        h = CreateMutexW(None, False, name)
        if not h:
            return None
        if GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            k32.CloseHandle(h)
            return None
        return int(h)
    except Exception:
        return None


def _release_windows_mutex(handle):
    if os.name != "nt" or not handle:
        return
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.ReleaseMutex(ctypes.c_void_p(handle))
        k32.CloseHandle(ctypes.c_void_p(handle))
    except Exception:
        pass


def _acquire_file_lock(lock_path_abs: str):
    try:
        os.makedirs(os.path.dirname(lock_path_abs), exist_ok=True)
        fd = os.open(lock_path_abs, os.O_RDWR | os.O_CREAT)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
        except Exception:
            pass
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 4096)
            except OSError:
                os.close(fd)
                return None
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                return None
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode("utf-8", errors="ignore"))
            os.fsync(fd)
        except Exception:
            pass
        return fd
    except Exception:
        return None
DEFAULT_HB_PATH = os.path.join("run", "heartbeat.txt")
DEFAULT_TRADES_PATH = os.path.join("results", "trades.csv")
DEFAULT_SHADOW_RTS_PATH = os.path.join("results", "shadow_roundtrips.csv")
DEFAULT_HEALTH_PATH = os.path.join("run", "health.json")
DEFAULT_ERR_LOG = os.path.join("logs", "hb_monitor_error.log")
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_LOCK_PATH = os.path.join(BASE_DIR, "run", "hb_monitor.lock")

ARM_ORDER = [
    "trend_ema",
    "trend_sma",
    "breakout_atr",
    "pullback_vwap",
    "momentum_rsi",
    "range_fade",
    "trend_pullback",
    # shadow arms
    "trend_ema2",
    "breakout_adx",
    "range_fade_strict",
    "mean_revert_ema",
    "momentum_pullback",
]

NON_ARM_LABELS = {"trend", "chop", "unknown"}
BUYSELL = {"BUY", "SELL"}


def _is_bad_float(v: float) -> bool:
    try:
        return (math.isnan(v) or math.isinf(v))
    except Exception:
        return True


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        v = float(x)
        return default if _is_bad_float(v) else v

    s = str(x).strip()
    if not s or s in ("-", "NA", "na", "None", "null"):
        return default

    if " " in s:
        s = s.split()[0]

    try:
        v = float(s.replace(",", ""))
        return default if _is_bad_float(v) else v
    except Exception:
        return default


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    if x is None:
        return default
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        try:
            if _is_bad_float(float(x)):
                return default
            return int(x)
        except Exception:
            return default

    s = str(x).strip()
    if not s or s == "-":
        return default
    try:
        v = float(s)
        if _is_bad_float(v):
            return default
        return int(v)
    except Exception:
        return default


def _compute_drawdown_pct(equity: Optional[float], equity_hwm: Optional[float]) -> float:
    try:
        if equity is None or equity_hwm is None:
            return 0.0
        hwm = float(equity_hwm)
        eq = float(equity)
        if hwm <= 0:
            return 0.0
        dd = (hwm - eq) / hwm * 100.0
        return float(dd) if dd > 0 else 0.0
    except Exception:
        return 0.0


def _parse_caps(caps: Any) -> List[str]:
    if caps is None:
        return []
    if isinstance(caps, list):
        return [str(x) for x in caps if str(x).strip()]
    if isinstance(caps, (int, float)):
        return [str(caps)]
    s = str(caps).strip()
    if not s or s == "-":
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x) for x in obj if str(x).strip()]
        except Exception:
            pass
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def _looks_like_price_not_R(v: float) -> bool:
    try:
        return abs(float(v)) > 50.0
    except Exception:
        return True


# ---------------- Console refresh ----------------

ALT_ON = "\x1b[?1049h"
ALT_OFF = "\x1b[?1049l"
ANSI_CLEAR = "\x1b[2J\x1b[H"


def _enable_vt_mode_windows() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes  # type: ignore
        kernel32 = ctypes.windll.kernel32  # type: ignore
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _enter_alt_screen() -> None:
    try:
        sys.stdout.write(ALT_ON)
        sys.stdout.flush()
    except Exception:
        pass


def _exit_alt_screen() -> None:
    try:
        sys.stdout.write(ALT_OFF)
        sys.stdout.flush()
    except Exception:
        pass


def _clear_screen(mode: str) -> None:
    # When in alt-screen on Windows, NEVER use cls; force ANSI clear.
    if mode == "ansi":
        try:
            sys.stdout.write(ANSI_CLEAR)
            sys.stdout.flush()
            return
        except Exception:
            pass

    if mode == "cls":
        os.system("cls" if os.name == "nt" else "clear")
        return

    # auto
    if os.name == "nt":
        os.system("cls")
    else:
        try:
            sys.stdout.write(ANSI_CLEAR)
            sys.stdout.flush()
        except Exception:
            os.system("clear")


def _append_error_log(msg: str) -> None:
    try:
        os.makedirs(os.path.dirname(DEFAULT_ERR_LOG) or ".", exist_ok=True)
        with open(DEFAULT_ERR_LOG, "a", encoding="utf-8", newline="\n") as f:
            f.write(msg)
            if not msg.endswith("\n"):
                f.write("\n")
    except Exception:
        pass


# ---------------- Windows-safe file reading (avoid blocking os.replace) ----------------

def _read_text_share_delete(path: str) -> str:
    """
    Read a text file in a way that will NOT prevent another process from atomically
    replacing the file on Windows (os.replace). On non-Windows, falls back to normal open().
    """
    if os.name != "nt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # On Windows, explicitly request FILE_SHARE_DELETE so a writer can replace the file
    # while we're reading it.
    try:
        import ctypes  # type: ignore

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        GENERIC_READ = 0x80000000
        FILE_SHARE_READ = 0x00000001
        FILE_SHARE_WRITE = 0x00000002
        FILE_SHARE_DELETE = 0x00000004
        OPEN_EXISTING = 3
        FILE_ATTRIBUTE_NORMAL = 0x00000080

        CreateFileW = kernel32.CreateFileW
        CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        CreateFileW.restype = ctypes.c_void_p

        ReadFile = kernel32.ReadFile
        ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
        ReadFile.restype = ctypes.c_int

        CloseHandle = kernel32.CloseHandle
        CloseHandle.argtypes = [ctypes.c_void_p]
        CloseHandle.restype = ctypes.c_int

        handle = CreateFileW(
            path,
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if not handle or handle == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_last_error(), "CreateFileW failed")

        try:
            chunks = []
            bufsize = 64 * 1024
            buf = ctypes.create_string_buffer(bufsize)
            read = ctypes.c_uint32(0)
            while True:
                ok = ReadFile(handle, buf, bufsize, ctypes.byref(read), None)
                if not ok:
                    raise OSError(ctypes.get_last_error(), "ReadFile failed")
                n = int(read.value)
                if n <= 0:
                    break
                chunks.append(buf.raw[:n])
            data = b"".join(chunks)
            # Heartbeat content is ASCII/UTF-8 JSON; be tolerant.
            return data.decode("utf-8", errors="replace")
        finally:
            try:
                CloseHandle(handle)
            except Exception:
                pass
    except Exception:
        # Fallback: normal open (still usually fine, but may block replace in some environments)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


# ---------------- Singleton guard (avoid two hb_monitor instances) ----------------


def _print_help() -> None:
    print("""hb_monitor.py (v3.21.0)

Usage:
  python hb_monitor.py
  python hb_monitor.py --alt-screen --interval 1.0
  python hb_monitor.py --ansi --interval 1.0
  python hb_monitor.py --hb-path .\\run\\heartbeat.txt

Options:
  --hb-path <path>        Heartbeat file path (default: run/heartbeat.txt)
  --trades-path <path>    Trades CSV path (default: results/trades.csv)
  --shadow-path <path>    Shadow roundtrips CSV path (default: results/shadow_roundtrips.csv)
  --interval <sec>        Refresh interval (default: 2.0; min: 0.25)
  --alt-screen            Use terminal alternate screen buffer
  --ansi                  Force ANSI clear (avoid cls)
  --no-singleton          Allow multiple hb_monitor instances (ignore lock)
  -h, --help              Show this help and exit
  --version               Print version and exit
""")


def _pid_alive(pid: int) -> bool:
    """Best-effort check whether a PID is alive."""
    try:
        pid = int(pid)
        if pid <= 0:
            return False
    except Exception:
        return False

    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except Exception:
        return False


def _read_lock_pid(lock_path: str) -> Optional[int]:
    try:
        with open(lock_path, "r", encoding="utf-8", errors="replace") as f:
            s = (f.read() or "").strip()
        if not s:
            return None
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _acquire_singleton_lock(lock_path: str) -> Optional[int]:
    """
    Best-effort process-level singleton: create lock file exclusively.

    v3.21.0 hardening:
    - Lock file contains PID (already).
    - If lock exists but PID is not alive, remove stale lock and acquire.
    Returns an OS file descriptor if acquired, else None.
    """
    try:
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    except Exception:
        pass

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    def _try_create() -> Optional[int]:
        try:
            fd = os.open(lock_path, flags)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8"))
            except Exception:
                pass
            return fd
        except FileExistsError:
            return None
        except Exception:
            return None

    fd = _try_create()
    if fd is not None:
        return fd

    # Lock exists: check for staleness.
    try:
        pid = _read_lock_pid(lock_path)
        if pid is None:
            return None
        if not _pid_alive(pid):
            try:
                os.remove(lock_path)
            except Exception:
                return None
            return _try_create()
        return None
    except Exception:
        return None


def _release_singleton_lock(fd: Optional[int], lock_path: str) -> None:
    try:
        if fd is not None:
            os.close(fd)
    except Exception:
        pass
    try:
        if fd is not None and os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


# ---------------- Heartbeat loader ----------------

def load_heartbeat(path: str) -> Dict[str, Any]:
    hb: Dict[str, Any] = {}
    if not os.path.exists(path):
        return hb

    text = ""
    for _ in range(3):
        try:
            text = _read_text_share_delete(path).strip()
            if text:
                break
        except Exception:
            time.sleep(0.02)

    if not text:
        return hb

    try:
        hb_json = json.loads(text)
        if isinstance(hb_json, dict):
            for k in ("extra_fields", "extra", "extras"):
                extra = hb_json.get(k)
                if isinstance(extra, dict):
                    for ek, ev in extra.items():
                        hb_json.setdefault(ek, ev)
            return hb_json
    except Exception:
        pass

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            hb[key.strip()] = val.strip()

    return hb


def load_health(path: str) -> Dict[str, Any]:
    """Load pt_healthcheck output (run/health.json). Safe to call even if missing."""
    h: Dict[str, Any] = {}
    if not os.path.exists(path):
        return h

    text = ""
    for _ in range(3):
        try:
            text = _read_text_share_delete(path).strip()
            if text:
                break
        except Exception:
            time.sleep(0.02)

    if not text:
        return h

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# ---------------- Trades stats ----------------

def load_trades(path: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "lifetime_trades": 0,
        "wins": 0,
        "losses": 0,
        "flat": 0,
        "win_rate": 0.0,
        "avg_R": 0.0,
        "realized_pnl": 0.0,
        "last_trades": [],
        "rows": [],
    }

    if not os.path.exists(path):
        return stats

    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        return stats

    if not rows:
        return stats

    stats["rows"] = rows
    stats["lifetime_trades"] = len(rows)

    wins = losses = flats = 0
    sum_R = 0.0
    n_R = 0
    total_pnl = 0.0

    for row in rows:
        pnl_val = _safe_float(row.get("pnl") or row.get("pnl_usd") or row.get("pnlUSD") or 0.0, 0.0) or 0.0
        total_pnl += pnl_val

        raw_R = (row.get("R") or row.get("r") or "")
        raw_R = raw_R.strip() if isinstance(raw_R, str) else str(raw_R).strip()
        R_val = _safe_float(raw_R, None)

        if R_val is None:
            flats += 1
        else:
            if R_val > 0:
                wins += 1
            elif R_val < 0:
                losses += 1
            else:
                flats += 1
            sum_R += R_val
            n_R += 1

    stats["wins"] = wins
    stats["losses"] = losses
    stats["flat"] = flats
    stats["realized_pnl"] = total_pnl
    stats["win_rate"] = (wins / float(wins + losses)) * 100.0 if (wins + losses) > 0 else 0.0
    stats["avg_R"] = (sum_R / n_R) if n_R > 0 else 0.0

    return stats


# ---------------- Shadow roundtrip stats ----------------

def _parse_shadow_row(fields: List[str]) -> Optional[Tuple[str, float]]:
    if not fields:
        return None
    f = [x.strip() for x in fields]

    if len(f) == 11:
        arm = f[3].strip()
        side = f[4].strip().upper()
        R = _safe_float(f[9], None)

        if side not in BUYSELL:
            alt_arm = f[4].strip()
            alt_side = f[5].strip().upper() if len(f) > 5 else ""
            alt_R = _safe_float(f[9], None)
            if alt_side in BUYSELL and alt_arm and alt_arm.lower() not in NON_ARM_LABELS:
                arm = alt_arm
                side = alt_side
                R = alt_R

        if not arm or arm.lower() in NON_ARM_LABELS:
            return None
        if R is None or _looks_like_price_not_R(R):
            return None
        return (arm, float(R))

    if len(f) >= 8:
        arm = f[2].strip()
        R = _safe_float(f[7], None)
        if not arm or arm.lower() in NON_ARM_LABELS:
            return None
        if R is None or _looks_like_price_not_R(R):
            return None
        return (arm, float(R))

    return None


def load_shadow_roundtrip_stats(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}

    per_arm_R = defaultdict(list)
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            _header = next(reader, None)
            for row in reader:
                parsed = _parse_shadow_row(row)
                if not parsed:
                    continue
                arm, R = parsed
                per_arm_R[arm].append(R)
    except Exception:
        return {}

    stats: Dict[str, Any] = {}
    for arm_name in ARM_ORDER:
        rs = per_arm_R.get(arm_name, [])
        if rs:
            stats[arm_name] = (len(rs), sum(rs) / float(len(rs)))

    for arm_name, rs in per_arm_R.items():
        if arm_name in stats:
            continue
        if rs:
            stats[arm_name] = (len(rs), sum(rs) / float(len(rs)))

    return stats


def fmt_money(x: Any, suffix: str = "USD") -> str:
    val = _safe_float(x, None)
    if val is None:
        return f"- {suffix}"
    return f"{val:.2f} {suffix}"


def fmt_pct(x: Any) -> str:
    val = _safe_float(x, None)
    if val is None:
        return "0.00 %"
    return f"{val:.2f} %"


def _as_pct_maybe_fraction(x: Any) -> float:
    v = _safe_float(x, None)
    if v is None:
        return 0.0
    if 0.0 <= v <= 1.0:
        return float(v * 100.0)
    return float(v)


def render_dashboard(
    hb: Dict[str, Any],
    trade_stats: Dict[str, Any],
    shadow_stats: Dict[str, Any],
    health: Dict[str, Any],
    *,
    clear_mode: str,
    error_banner: Optional[str] = None,
) -> None:
    _clear_screen(clear_mode)

    now_local = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hb_ts = hb.get("ts") or hb.get("timestamp") or "-"
    print(f"[hb_monitor v3.21.0] now={now_local} | hb_ts={hb_ts} | hb_file={DEFAULT_HB_PATH}")
    if error_banner:
        print(f"[ERROR] {error_banner}")
    print()

    timestamp = hb_ts
    state = hb.get("state") or hb.get("bot_state") or "-"
    idle_reason = hb.get("idle_reason") or hb.get("idle") or ""
    regime = hb.get("regime") or "unknown"

    caps_list = _parse_caps(hb.get("caps") or hb.get("CAPPED") or [])
    caps_str = ", ".join(caps_list) if caps_list else "-"

    bayes_source = hb.get("bayes_source") or hb.get("BAYES_SOURCE") or "-"
    restart_ct = hb.get("restart_ct") or hb.get("restart_time") or hb.get("DAILY_RESTART_CT") or "-"

    day_R = _safe_float(hb.get("day_R"), 0.0) or 0.0
    week_R = _safe_float(hb.get("week_R"), 0.0) or 0.0

    print("ES Paper Trader Heartbeat Dashboard (v3.21.0)")
    print("-------------------------------------------")
    print(f"HB file : {DEFAULT_HB_PATH}")
    print(f"Trades  : {DEFAULT_TRADES_PATH}")
    print()

    print("State")
    print("-----")
    print(f"Timestamp    : {timestamp}")
    print(f"State        : {state}  | idle_reason: {idle_reason}")
    print(f"Regime       : {regime}")
    print(f"Caps         : {caps_str}")
    print(f"bayes_source : {bayes_source}  | restart_ct: {restart_ct}")
    print(f"day_R        : {day_R:.3f}")
    print(f"week_R       : {week_R:.3f}")
    print()

    # ---------------- Healthcheck (pt_healthcheck.py) ----------------
    print("Health (Connection + History)")
    print("---------------------------")
    if not health:
        print(f"health.json   : (missing) {DEFAULT_HEALTH_PATH}")
        print("READY_NEXT    : -")
        print("LIVE_OK       : -")
    else:
        ok_api = bool(health.get("ok_api"))
        ok_hist = bool(health.get("ok_hist"))
        ok_warmup = bool(health.get("ok_warmup"))
        ok_fresh = bool(health.get("ok_fresh"))

        ready_next = bool(ok_api and ok_hist and ok_warmup)
        live_ok = bool(ready_next and ok_fresh)

        bars_h = health.get("bars", "-")
        last_bar = health.get("last_bar_iso", "-")
        ts_local = health.get("ts_local", "-")
        reason = health.get("ready_reason") or health.get("reason") or ""

        print(f"health.json   : {DEFAULT_HEALTH_PATH}  | ts: {ts_local}")
        print(f"READY_NEXT    : {'GO' if ready_next else 'NO-GO'}")
        print(f"LIVE_OK       : {'GO' if live_ok else 'NO-GO'}")
        print(f"bars/last     : {bars_h}  | {last_bar}")
        if reason:
            print(f"reason        : {reason}")
    print()

    pos_state = hb.get("pos_state") or "-"
    net_qty_val = _safe_int(hb.get("net_qty") or hb.get("position") or 0, 0) or 0
    last_px = _safe_float(hb.get("px") or hb.get("last_px") or hb.get("last_price") or hb.get("last") or None, None)
    entry_px_val = _safe_float(hb.get("entry_px"), None)
    unreal_pnl_val = _safe_float(hb.get("pnl_unreal_usd") or hb.get("unreal_pnl") or hb.get("unreal") or None, None)
    bars = _safe_int(hb.get("bars") or hb.get("bar_count") or 0, 0) or 0

    current_arm = hb.get("current_arm")
    current_side = hb.get("current_side")
    last_signal_arm = hb.get("last_signal_arm")
    last_signal_side = hb.get("last_signal_side")

    arm_pos = current_arm or "-"
    side_pos = (current_side or "-") or "-"
    sig_arm = last_signal_arm or "-"
    sig_side = (last_signal_side or "-") or "-"

    atr_points = _safe_float(hb.get("atr_points") or hb.get("ATR") or None, None)
    adx_val = _safe_float(hb.get("adx_val") or hb.get("ADX") or None, None)

    print("Position & PnL")
    print("----------------")
    print(f"pos_state    : {pos_state}  | net_qty: {net_qty_val}")
    print(f"Arm (pos)    : {arm_pos:<16} | side: {side_pos}")
    print(f"Last signal  : {sig_arm:<16} | side: {sig_side}")
    print("last px      : -" if last_px is None else f"last px      : {last_px:.2f}")

    if atr_points is not None or adx_val is not None:
        atr_s = "-" if atr_points is None else f"{atr_points:.2f}"
        adx_s = "-" if adx_val is None else f"{adx_val:.1f}"
        print(f"ATR/ADX      : atr={atr_s} pts | adx={adx_s}")

    if net_qty_val == 0 or entry_px_val is None or entry_px_val <= 0:
        print("entry px     : -")
        print("unreal PnL   : - USD")
    else:
        print(f"entry px     : {entry_px_val:.2f}")
        if unreal_pnl_val is None or abs(unreal_pnl_val) > 50000:
            print("unreal PnL   : - USD")
        else:
            print(f"unreal PnL   : {unreal_pnl_val:.2f} USD")

    print(f"bars         : {bars}")
    print()

    open_orders = hb.get("open_orders") or hb.get("open_orders_count") or 0
    stop_px = hb.get("stop_px") or hb.get("stop_price") or "-"
    target_px = hb.get("target_px") or hb.get("target_price") or "-"

    print("Orders")
    print("------")
    print(f"open_orders  : {open_orders}")
    print(f"stop_px      : {stop_px}")
    print(f"target_px    : {target_px}")
    print()

    trades_today = hb.get("trades_today") or hb.get("day_trades") or 0
    pnl_today = hb.get("running_pnl_today") or hb.get("pnl_today") or hb.get("day_pnl") or 0.0
    total_trades = hb.get("total_trades") or trade_stats.get("lifetime_trades", 0)

    print("Performance (Today)")
    print("-------------------")
    print(f"trades_today : {trades_today}")
    print(f"day_R        : {day_R:.3f}")
    print(f"PnL (today)  : {fmt_money(pnl_today)}")
    print(f"total_trades : {total_trades}")
    print()

    equity_val = _safe_float(hb.get("equity") or hb.get("netliq") or hb.get("acct_netliq") or None, None)
    equity_hwm_val = _safe_float(hb.get("equity_hwm") or hb.get("hwm") or None, None)
    hwm_factor = hb.get("hwm_factor") or 1.0

    dd_raw = _safe_float(hb.get("drawdown_pct"), None)
    if dd_raw is None:
        dd_raw = _safe_float(hb.get("drawdown"), None)
    if dd_raw is None:
        dd_raw = _compute_drawdown_pct(equity_val, equity_hwm_val)

    margin_used_pct = _as_pct_maybe_fraction(
        hb.get("margin_used_pct") if hb.get("margin_used_pct") is not None else hb.get("margin_used")
    )

    meta_ema_R = _safe_float(hb.get("meta_ema_R") or hb.get("meta_R") or 0.0, 0.0) or 0.0
    meta_aggr = _safe_float(hb.get("meta_aggr") or 1.0, 1.0) or 1.0
    sharpe_R = _safe_float(hb.get("sharpe_R") or hb.get("sharpe") or 0.0, 0.0) or 0.0
    boost_mode = hb.get("boost_mode") or "off"
    boost_factor = _safe_float(hb.get("boost_factor"), 1.0) or 1.0

    print("Equity / Meta")
    print("-------------")
    print(f"equity       : {equity_val:.1f}" if equity_val is not None else "equity       : -")
    print(f"equity_hwm   : {equity_hwm_val:.1f}" if equity_hwm_val is not None else "equity_hwm   : -")
    print(f"hwm_factor   : {hwm_factor}")
    print(f"drawdown     : {fmt_pct(dd_raw)}")
    print(f"margin used  : {margin_used_pct:.2f} %")
    print(f"meta_ema_R   : {float(meta_ema_R):.3f}")
    print(f"meta_aggr    : {float(meta_aggr):.3f}")
    print(f"Sharpe*(R)   : {float(sharpe_R):.3f}")
    print(f"boost_mode   : {boost_mode}  (factor={boost_factor:.3f})")
    print()

    shadow_pnl_today = hb.get("shadow_pnl_today") or 0.0
    shadow_R_today = hb.get("shadow_R_today") or 0.0
    shadow_trades_today = hb.get("shadow_trades_today") or 0

    print("Shadow Learning (virtual trades while capped)")
    print("---------------------------------------------")
    print(f"shadow_trades: {int(_safe_int(shadow_trades_today, 0) or 0)}")
    print(f"shadow_R     : {float(_safe_float(shadow_R_today, 0.0) or 0.0):.3f}")
    print(f"shadow PnL   : {fmt_money(shadow_pnl_today)}")
    print()

    print("Shadow Arm Performance (lifetime)")
    print("---------------------------------")
    if not shadow_stats:
        print("(no shadow roundtrips yet)")
    else:
        print("arm                 n    mean_R")
        print("--------------------------------")
        printed = set()
        for arm_name in ARM_ORDER:
            if arm_name in shadow_stats:
                n_val, mean_R_val = shadow_stats[arm_name]
                mean_str = "   -  " if mean_R_val is None else f"{mean_R_val:6.3f}"
                print(f"{arm_name:18} {n_val:4d} {mean_str}")
                printed.add(arm_name)
        for arm_name in [a for a in shadow_stats.keys() if a not in printed]:
            n_val, mean_R_val = shadow_stats[arm_name]
            mean_str = "   -  " if mean_R_val is None else f"{mean_R_val:6.3f}"
            print(f"{arm_name:18} {n_val:4d} {mean_str}")

    print()
    print("(Refreshing... Press Ctrl+C to exit.)")
    sys.stdout.flush()



def _parse_args(argv: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "hb_path": DEFAULT_HB_PATH,
        "trades_path": DEFAULT_TRADES_PATH,
        "shadow_path": DEFAULT_SHADOW_RTS_PATH,
        "interval": 2.0,
        "alt_screen": False,
        "ansi": False,
        "help": False,
        "version": False,
    }

    i = 0
    while i < len(argv):
        a = argv[i]

        if a in ("-h", "--help"):
            out["help"] = True
        elif a == "--version":
            out["version"] = True
        elif a == "--alt-screen":
            out["alt_screen"] = True
        elif a == "--ansi":
            out["ansi"] = True
        elif a == "--no-singleton":
            out["no_singleton"] = True
        elif a == "--interval" and i + 1 < len(argv):
            out["interval"] = float(_safe_float(argv[i + 1], 2.0) or 2.0)
            i += 1
        elif a == "--hb-path" and i + 1 < len(argv):
            out["hb_path"] = str(argv[i + 1])
            i += 1
        elif a == "--trades-path" and i + 1 < len(argv):
            out["trades_path"] = str(argv[i + 1])
            i += 1
        elif a == "--shadow-path" and i + 1 < len(argv):
            out["shadow_path"] = str(argv[i + 1])
            i += 1
        else:
            # Positional args fallback (backwards compatible)
            if a and not a.startswith("--"):
                if out["hb_path"] == DEFAULT_HB_PATH:
                    out["hb_path"] = a
                elif out["trades_path"] == DEFAULT_TRADES_PATH:
                    out["trades_path"] = a
                elif out["shadow_path"] == DEFAULT_SHADOW_RTS_PATH:
                    out["shadow_path"] = a
        i += 1

    try:
        out["interval"] = max(0.25, float(out["interval"]))
    except Exception:
        out["interval"] = 2.0

    return out



def main() -> None:
    cfg = _parse_args(sys.argv[1:])

    if bool(cfg.get("help")):
        _print_help()
        return
    if bool(cfg.get("version")):
        print("hb_monitor v3.21.0")
        return

    # Prevent multiple hb_monitor instances from fighting over the same files.
    lock_fd: Optional[int] = None
    if not bool(cfg.get("no_singleton")):
        lock_fd = _acquire_singleton_lock(DEFAULT_LOCK_PATH)
        if lock_fd is None:
            print(f"hb_monitor: another instance is already running (lock={DEFAULT_LOCK_PATH}).")
            print("If you really want multiple instances, run with --no-singleton.")
            return

        # Best-effort cleanup on exit (reduces stale locks).
        try:
            atexit.register(_release_singleton_lock, lock_fd, DEFAULT_LOCK_PATH)
        except Exception:
            pass

    hb_path = cfg["hb_path"]
    trades_path = cfg["trades_path"]
    shadow_path = cfg["shadow_path"]
    interval = float(cfg["interval"])

    use_alt = bool(cfg["alt_screen"])
    use_ansi = bool(cfg["ansi"])

    # If using alt screen, force ANSI clear (no cls).
    if use_alt:
        clear_mode = "ansi"
    else:
        clear_mode = "cls" if (os.name == "nt" and not use_ansi) else "ansi"

    if use_alt:
        _enable_vt_mode_windows()
        _enter_alt_screen()

    last_err_banner: Optional[str] = None

    try:
        while True:
            try:
                hb = load_heartbeat(hb_path)
                health = load_health(DEFAULT_HEALTH_PATH)
                trade_stats = load_trades(trades_path)
                shadow_stats = load_shadow_roundtrip_stats(shadow_path)
                render_dashboard(
                    hb,
                    trade_stats,
                    shadow_stats,
                    health,
                    clear_mode=clear_mode,
                    error_banner=last_err_banner,
                )
                last_err_banner = None
            except Exception as e:
                # Keep the monitor alive; record the failure.
                ts = dt.datetime.now().isoformat(timespec="seconds")
                tb = traceback.format_exc()
                banner = f"{ts} refresh_error: {type(e).__name__}: {e}"
                last_err_banner = banner

                _append_error_log("\n" + "=" * 80)
                _append_error_log(banner)
                _append_error_log(tb)

                # Try to show something immediately even if rendering failed.
                try:
                    _clear_screen(clear_mode)
                    print(f"[hb_monitor v3.21.0] {banner}")
                    print(f"(traceback appended to {DEFAULT_ERR_LOG})")
                    sys.stdout.flush()
                except Exception:
                    pass

                time.sleep(max(0.5, interval))

            time.sleep(interval)

    except KeyboardInterrupt:
        pass
    finally:
        _release_singleton_lock(lock_fd, DEFAULT_LOCK_PATH)
        if use_alt:
            _exit_alt_screen()
        print("\nExiting hb_monitor.")


if __name__ == "__main__":
    main()