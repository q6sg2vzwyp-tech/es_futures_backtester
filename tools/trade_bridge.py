"""
pt.trade_bridge (stage2+ stubbed)

Safe minimal implementation to bypass corrupted historical versions.
Provides commonly imported symbols used by core runtime modules.

Exports:
  - new_trade_id()
  - log_event(...)
  - log_trade(...)
  - handle_realized_pnl_event(...)

Logs to: run/events.log and run/trades.log
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid
from pathlib import Path as _Path
from typing import Any, Dict, Optional, Tuple


_RUN_DIR = _Path(__file__).resolve().parents[1] / "run"
_RUN_DIR.mkdir(parents=True, exist_ok=True)

_EVENTS_LOG = _RUN_DIR / "events.log"
_TRADES_LOG = _RUN_DIR / "trades.log"


def new_trade_id() -> str:
    return _uuid.uuid4().hex


def _ts() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_event(event: str, **fields: Any) -> None:
    """Append a single-line event record. Never raises."""
    try:
        parts = [f"ts={_ts()}", f"event={event}"]
        for k, v in fields.items():
            parts.append(f"{k}={v!r}")
        _EVENTS_LOG.open("a", encoding="utf-8").write(" ".join(parts) + "\n")
    except Exception:
        pass


def log_trade(trade: Dict[str, Any], *, trade_id: Optional[str] = None) -> None:
    """Append a single-line trade record. Never raises."""
    try:
        tid = trade_id or trade.get("trade_id") or new_trade_id()
        parts = [f"ts={_ts()}", f"trade_id={tid}"]
        for k, v in trade.items():
            if k == "trade_id":
                continue
            parts.append(f"{k}={v!r}")
        _TRADES_LOG.open("a", encoding="utf-8").write(" ".join(parts) + "\n")
    except Exception:
        pass


def handle_realized_pnl_event(*args: Any, **kwargs: Any) -> None:
    """
    Compatibility shim.

    Older versions of the repo import this from pt.trade_bridge.
    We don't assume a specific payload schema; we just log the event.
    """
    try:
        # keep it compact—kwargs can be large; args tuple is still useful
        log_event("REALIZED_PNL", args=args, **kwargs)
    except Exception:
        pass
