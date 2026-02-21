from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import json
import os
import threading
import time

_HB_JSON_PATH_DEFAULT = r".\run\heartbeat.txt"
_HB_KV_PATH_DEFAULT   = r".\run\heartbeat_kv.txt"

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
    "parent_entry_id": None,
    "parent_to_mkt_limit_sec": None,
    "parent_to_mkt_age_sec": None,
    "parent_to_mkt_remaining_sec": None,
}

_log: Optional[Callable[..., Any]] = None
_ct_now: Optional[Callable[[], Any]] = None

_hb_json_path: str = _HB_JSON_PATH_DEFAULT
_hb_kv_path: str   = _HB_KV_PATH_DEFAULT

_thread_started = False
_thread_lock = threading.Lock()


def init_heartbeat(
    log_fn: Callable[..., Any],
    *,
    ct_now_fn: Optional[Callable[[], Any]] = None,
    hb_json_path: str = _HB_JSON_PATH_DEFAULT,
    hb_kv_path: str = _HB_KV_PATH_DEFAULT,
) -> None:
    global _log, _ct_now, _hb_json_path, _hb_kv_path
    _log = log_fn
    _ct_now = ct_now_fn
    _hb_json_path = hb_json_path or _HB_JSON_PATH_DEFAULT
    _hb_kv_path = hb_kv_path or _HB_KV_PATH_DEFAULT


def hb_update(**kv: Any) -> None:
    with _hb_lock:
        _hb_state.update(kv)


def _write_atomic(path: str, content: str) -> None:
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        return


def _write_json(payload: Dict[str, Any]) -> None:
    _write_atomic(_hb_json_path, json.dumps(payload, ensure_ascii=False) + "\n")


def _write_kv(payload: Dict[str, Any]) -> None:
    try:
        lines = [f"{k}={v}" for k, v in payload.items()]
        _write_atomic(_hb_kv_path, "\n".join(lines) + "\n")
    except Exception:
        return


def _hb_loop() -> None:
    while True:
        with _hb_lock:
            payload = dict(_hb_state)

        try:
            if _ct_now is not None:
                payload["ts"] = _ct_now().isoformat(timespec="seconds")
        except Exception:
            pass

        _write_json(payload)
        _write_kv(payload)

        try:
            if _log is not None:
                _log("hb", **payload)
        except Exception:
            pass

        time.sleep(1.0)


def start_heartbeat_thread() -> None:
    global _thread_started
    with _thread_lock:
        if _thread_started:
            return
        t = threading.Thread(target=_hb_loop, daemon=True, name="pt-heartbeat")
        t.start()
        _thread_started = True
