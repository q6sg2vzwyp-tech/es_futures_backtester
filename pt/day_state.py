from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Callable


def mkdirs(p: str):
    try:
        if p:
            os.makedirs(p, exist_ok=True)
    except Exception:
        pass


def load_json(path: str, log: Optional[Callable[..., None]] = None) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        if log:
            try:
                log("day_state_load_err", path=path, err=str(e))
            except Exception:
                pass
    return {}


def save_json(path: str, data: Dict[str, Any], log: Optional[Callable[..., None]] = None):
    try:
        mkdirs(os.path.dirname(path))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception as e:
        if log:
            try:
                log("day_state_save_err", path=path, err=str(e))
            except Exception:
                pass


def default_payload(start_equity: float, week_id: str) -> Dict[str, Any]:
    return {
        "start_equity": float(start_equity),
        "day_realized": 0.0,
        "day_peak_realized": 0.0,
        "day_R": 0.0,
        "trades": 0,
        "consec_losses": 0,
        "week_R": 0.0,
        "last_week_id": str(week_id),
    }


def ensure_entry(
    day_state: Dict[str, Any],
    key: str,
    *,
    start_equity: float,
    week_id: str,
    save_fn: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    if key not in day_state:
        day_state[key] = default_payload(start_equity, week_id)
        save_fn(day_state)

    payload = day_state[key] or {}
    defaults = default_payload(payload.get("start_equity", start_equity), week_id)

    # Fill missing fields
    for fld, default_val in defaults.items():
        if fld not in payload:
            payload[fld] = default_val

    day_state[key] = payload
    return payload


def persist_snapshot(
    *,
    day_state: Dict[str, Any],
    key: str,
    start_equity: float,
    week_id: str,
    day_realized: float,
    day_peak_realized: float,
    day_R: float,
    trades: int,
    consec_losses: int,
    week_R: float,
    save_fn: Callable[[Dict[str, Any]], None],
    log: Optional[Callable[..., None]] = None,
) -> None:
    """Persist a standardized snapshot into day_state[key] and save via save_fn."""
    try:
        payload = ensure_entry(
            day_state,
            key,
            start_equity=float(start_equity),
            week_id=str(week_id),
            save_fn=save_fn,
        )
        payload["day_realized"] = float(day_realized)
        payload["day_peak_realized"] = float(day_peak_realized)
        payload["day_R"] = float(day_R)
        payload["trades"] = int(trades)
        payload["consec_losses"] = int(consec_losses)
        payload["week_R"] = float(week_R)
        payload["last_week_id"] = str(week_id)

        day_state[key] = payload
        save_fn(day_state)
    except Exception as e:
        if log:
            try:
                log("day_state_snapshot_err", err=str(e))
            except Exception:
                pass
