#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trade_bridge.py  (compat + trade logger + tiny learner)  (v2.3)

Bridges paper_trader.py <-> trade_log_core.py (and legacy trades_core)

- Detects changes in IB account RealizedPnL.
- Treats each delta as one "closed trade".
- Writes:
    * results/trades.csv        (canonical 8-col, hb_monitor-safe)
    * results/trades_rich.csv   (enriched summary via trade_log_core)
    * results/trade_events.csv  (append-only event ledger; execution truth spine)

Option 1 mode (selected):
- RealizedPnL delta remains the primary "close" trigger for summary rows.
- Order submit/fill/cancel events are recorded opportunistically (never block trading).

IMPORTANT:
- DO NOT increment day_risk.trades_this_hour here.
  loop_core increments trades_this_hour on *successful entry*.
"""

from __future__ import annotations

import os
import csv
import time
import json
import uuid
import datetime as dt
from datetime import datetime
from typing import Optional, Any, List, Dict, Tuple

from learner_bandit import save_thompson

# Legacy import still used as fallback (last resort)
from trades_core import (
    log_trade as _log_trade_legacy,
    DEFAULT_RISK_TICKS,
    DEFAULT_TICK_VALUE,
)

# Preferred new centralized logger (returns a dict w/ computed R)
try:
    from trade_log_core import log_trade as _log_trade_core
except Exception:
    _log_trade_core = None

ES_MULTIPLIER = 50.0  # ES contract multiplier
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Canonical paths
EVENTS_PATH = os.path.join(BASE_DIR, "results", "trade_events.csv")
SHADOW_TRADE_LOG = os.path.join(BASE_DIR, "results", "shadow_trades.csv")

CANON_TRADES_FIELDS = ["timestamp", "side", "qty", "entry_px", "exit_px", "pnl", "R", "tags"]


TRADES_LEDGER_PATH = os.path.join(BASE_DIR, "results", "trades_ledger.csv")

def _ensure_parent(path: str) -> None:
    """Ensure parent directory exists."""
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass


def new_trade_id(prefix: str = "T") -> str:
    """Stable identifier you can stamp onto IB orders via orderRef."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def log_event(
    event: str,
    trade_id: str,
    **fields: Any,
) -> None:
    """
    Append an atomic event row to results/trade_events.csv.

    Schema is stable; any extra fields are JSON-packed into extra_json.
    This function is intentionally non-throwing for trading safety.
    """
    try:
        _ensure_parent(EVENTS_PATH)
        file_exists = os.path.exists(EVENTS_PATH)

        cols = [
            "ts",
            "event",
            "trade_id",
            "order_id",
            "arm",
            "side",
            "qty",
            "fill_px",
            "expected_px",
            "commission",
            "reason",
            "extra_json",
        ]

        ts = fields.pop("ts", None)
        if ts is None:
            ts = dt.datetime.now(dt.timezone.utc).astimezone()
        elif isinstance(ts, datetime):
            pass
        else:
            # best-effort: stringify
            try:
                ts = dt.datetime.fromisoformat(str(ts))
            except Exception:
                ts = dt.datetime.now(dt.timezone.utc).astimezone()

        row = {
            "ts": ts.isoformat(timespec="seconds"),
            "event": str(event or ""),
            "trade_id": str(trade_id or ""),
            "order_id": str(fields.pop("order_id", "") or ""),
            "arm": str(fields.pop("arm", "") or ""),
            "side": str(fields.pop("side", "?") or "?"),
            "qty": int(fields.pop("qty", 0) or 0),
            "fill_px": "" if fields.get("fill_px", None) is None else f"{float(fields.pop('fill_px')):.4f}",
            "expected_px": "" if fields.get("expected_px", None) is None else f"{float(fields.pop('expected_px')):.4f}",
            "commission": "" if fields.get("commission", None) is None else f"{float(fields.pop('commission')):.4f}",
            "reason": str(fields.pop("reason", "") or ""),
            "extra_json": json.dumps(fields, separators=(",", ":"), ensure_ascii=False) if fields else "",
        }

        with open(EVENTS_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if not file_exists:
                w.writeheader()
            w.writerow(row)
    except Exception:
        # Never let logging crash trading
        return


def _read_first_line(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (f.readline() or "").strip()
    except Exception:
        return ""


def _canon_trades_writer_append(path: str, row: Dict[str, Any]) -> None:
    """
    Append a row to trades.csv using canonical 8-column schema.
    Also performs safe header repair if needed.
    """
    _ensure_parent(path)

    expected_header = ",".join(CANON_TRADES_FIELDS)
    file_exists = os.path.exists(path)

    if not file_exists:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CANON_TRADES_FIELDS)
            w.writeheader()
            w.writerow({k: row.get(k, "") for k in CANON_TRADES_FIELDS})
        return

    first = _read_first_line(path)
    if first and first != expected_header:
        try:
            with open(path, "r", encoding="utf-8") as fr:
                old_lines = fr.read().splitlines()
        except Exception:
            old_lines = []
        body = old_lines[1:] if len(old_lines) >= 1 else []
        try:
            with open(path, "w", newline="", encoding="utf-8") as fw:
                fw.write(expected_header + "\n")
                for line in body:
                    fw.write(line + "\n")
        except Exception:
            pass

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CANON_TRADES_FIELDS)
        w.writerow({k: row.get(k, "") for k in CANON_TRADES_FIELDS})


def _ledger_append(path: str, row: Dict[str, Any]) -> None:
    """
    Append-only writer for trades_ledger.csv.

    - Never rewrites existing files (no header repair).
    - Writes header only if file is missing or empty.
    """
    _ensure_parent(path)
    header = ",".join(CANON_TRADES_FIELDS)
    try:
        need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    except Exception:
        need_header = True

    with open(path, "a", newline="", encoding="utf-8") as f:
        if need_header:
            try:
                f.write(header + "\n")
            except Exception:
                pass
        w = csv.DictWriter(f, fieldnames=CANON_TRADES_FIELDS)
        w.writerow({k: row.get(k, "") for k in CANON_TRADES_FIELDS})


def _log_shadow_trade_row(
    *,
    ts: datetime,
    arm: str,
    side: str,
    prev_px: float,
    last_px: float,
    px_diff: float,
    pnl_delta: float,
    trade_R: float,
    gate_reason: str,
) -> None:
    """Append a single ghost trade row to results/shadow_trades.csv."""
    _ensure_parent(SHADOW_TRADE_LOG)
    file_exists = os.path.exists(SHADOW_TRADE_LOG)

    fieldnames = [
        "timestamp",
        "arm",
        "side",
        "prev_px",
        "last_px",
        "px_diff",
        "pnl_usd_1ct",
        "R",
        "gate_reason",
    ]

    with open(SHADOW_TRADE_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": ts.isoformat(timespec="seconds"),
                "arm": arm,
                "side": side,
                "prev_px": f"{prev_px:.2f}",
                "last_px": f"{last_px:.2f}",
                "px_diff": f"{px_diff:.4f}",
                "pnl_usd_1ct": f"{pnl_delta:.2f}",
                "R": f"{trade_R:.4f}",
                "gate_reason": gate_reason,
            }
        )


def shadow_bandit_update_for_signal(
    *,
    bandit,
    arm: Optional[str],
    side: Optional[str],
    prev_px: Optional[float],
    last_px: Optional[float],
    args,
    logger,
    gate_reason: Optional[str] = None,
):
    if bandit is None or not arm or side is None or prev_px is None or last_px is None:
        return bandit

    try:
        side_u = str(side).strip().upper()
        if side_u not in ("BUY", "SELL", "LONG", "SHORT"):
            return bandit

        px_diff = float(last_px) - float(prev_px)

        if side_u in ("BUY", "LONG"):
            pnl_delta = px_diff * ES_MULTIPLIER
            side_emit = "BUY"
        else:
            pnl_delta = -px_diff * ES_MULTIPLIER
            side_emit = "SELL"

        risk_ticks = float(getattr(args, "risk_ticks", DEFAULT_RISK_TICKS) or DEFAULT_RISK_TICKS)
        tick_size = float(getattr(args, "tick_size", 0.25) or 0.25)
        risk_dollars = risk_ticks * tick_size * ES_MULTIPLIER
        if risk_dollars <= 0:
            return bandit

        trade_R = pnl_delta / risk_dollars
        trade_R = max(-3.0, min(3.0, float(trade_R)))

        try:
            _log_shadow_trade_row(
                ts=datetime.now(),
                arm=arm,
                side=side_emit,
                prev_px=float(prev_px),
                last_px=float(last_px),
                px_diff=px_diff,
                pnl_delta=pnl_delta,
                trade_R=trade_R,
                gate_reason=gate_reason or "",
            )
        except Exception as e:
            if logger is not None:
                try:
                    logger.error(f"[shadow_log] failed to write row: {e}")
                except Exception:
                    pass

        update_fn = getattr(bandit, "update", None)
        if callable(update_fn):
            update_fn(arm, float(trade_R))
            if logger is not None:
                try:
                    logger.info(
                        f"[shadow_bandit] arm={arm} side={side_emit} "
                        f"prev_px={prev_px:.2f} last_px={last_px:.2f} "
                        f"R={trade_R:.3f} gate={gate_reason}"
                    )
                except Exception:
                    pass

    except Exception as e:
        if logger is not None:
            try:
                logger.error(f"[shadow_bandit] failed: {e}")
            except Exception:
                pass

    return bandit


def _safe_get(attr_obj: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(attr_obj, name))
    except Exception:
        return default


def _safe_set(obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def _normalize_side_for_logger(side: str) -> str:
    s = (side or "").strip().upper()
    if s in ("BUY", "LONG"):
        return "BUY"
    if s in ("SELL", "SHORT"):
        return "SELL"
    return "?"


def _emit_side_for_canon(side_buy_sell: str) -> str:
    s = (side_buy_sell or "").strip().upper()
    if s == "BUY":
        return "LONG"
    if s == "SELL":
        return "SHORT"
    return "?"


def handle_realized_pnl_event(
    *,
    ib,
    con,
    now_ct: datetime,
    acct_realized_pnl: Optional[float],
    last_acct_realized: Optional[float],
    args,
    day_risk,
    week_state,
    bandit,
    meta,
    current_arm: Optional[str],
    current_side: Optional[str],
    gate_reason: Optional[str],
    trades_today: int,
    total_trades: int,
    running_pnl_today: float,
    wins_today: int,
    losses_today: int,
    last_trade_close_ts: Optional[float],
    es_avg_px: Optional[float],
    last_px: float,
    trade_log_csv: str,
    learn_model_path: str,
    day_date,
    caps: List[str],
    pos_entry_px: Optional[float] = None,
    pos_entry_ts: Optional[str] = None,
    net: Optional[int] = None,
    logger=None,
):
    """Primary close detector (RealizedPnL delta)."""

    # Normalize acct_realized_pnl
    try:
        if acct_realized_pnl is None:
            acct_realized_pnl = last_acct_realized
        if acct_realized_pnl is not None:
            acct_realized_pnl = float(acct_realized_pnl)
    except Exception:
        acct_realized_pnl = last_acct_realized

    # First run: seed last_acct_realized, no trade
    if last_acct_realized is None and acct_realized_pnl is not None:
        last_acct_realized = acct_realized_pnl
        return (
            last_acct_realized,
            last_trade_close_ts,
            trades_today,
            total_trades,
            running_pnl_today,
            wins_today,
            losses_today,
            day_risk,
            week_state,
            bandit,
            meta,
            current_arm,
            current_side,
        )

    if acct_realized_pnl is None:
        return (
            last_acct_realized,
            last_trade_close_ts,
            trades_today,
            total_trades,
            running_pnl_today,
            wins_today,
            losses_today,
            day_risk,
            week_state,
            bandit,
            meta,
            current_arm,
            current_side,
        )

    if last_acct_realized is not None and abs(acct_realized_pnl - last_acct_realized) < 1e-9:
        return (
            last_acct_realized,
            last_trade_close_ts,
            trades_today,
            total_trades,
            running_pnl_today,
            wins_today,
            losses_today,
            day_risk,
            week_state,
            bandit,
            meta,
            current_arm,
            current_side,
        )

    # ---- New trade detected ----
    pnl_delta = (
        acct_realized_pnl - (last_acct_realized or 0.0)
        if last_acct_realized is not None
        else 0.0
    )

    trade_ts = now_ct
    last_acct_realized = acct_realized_pnl
    last_trade_close_ts = trade_ts.timestamp()

    trades_today += 1
    total_trades += 1
    running_pnl_today += pnl_delta

    if pnl_delta > 0:
        wins_today += 1
    elif pnl_delta < 0:
        losses_today += 1

    side_norm = _normalize_side_for_logger(current_side or "")
    if side_norm == "?":
        if pnl_delta > 0:
            side_norm = "BUY"
        elif pnl_delta < 0:
            side_norm = "SELL"

    qty = 1

    # Entry price snapshot preferred
    entry_px = None
    if pos_entry_px is not None:
        try:
            entry_px = float(pos_entry_px)
        except Exception:
            entry_px = None

    if entry_px is None:
        if es_avg_px is not None:
            entry_px = float(es_avg_px)
        elif last_px is not None:
            entry_px = float(last_px)

    exit_px = float(last_px)

    risk_ticks = float(getattr(args, "risk_ticks", DEFAULT_RISK_TICKS) or DEFAULT_RISK_TICKS)
    tick_size = float(getattr(args, "tick_size", 0.25) or 0.25)

    # tags string
    tags_parts: List[str] = []
    try:
        if gate_reason:
            tags_parts.append(f"gate_at_close={str(gate_reason)}")
        if caps:
            safe_caps = [str(c) for c in caps if c]
            if safe_caps:
                tags_parts.append("caps_at_close=" + ",".join(safe_caps))
        if current_arm:
            tags_parts.append(f"arm={str(current_arm)}")
        if net is not None:
            try:
                tags_parts.append(f"net_at_close={int(net)}")
            except Exception:
                tags_parts.append(f"net_at_close={str(net)}")
        if pos_entry_ts:
            tags_parts.append(f"entry_ts={str(pos_entry_ts)}")
    except Exception:
        tags_parts = []

    tags_str = ";".join(tags_parts).replace("\r", " ").replace("\n", " ").strip()
    if len(tags_str) > 512:
        tags_str = tags_str[:512] + "...(trunc)"

    # event row (close)
    try:
        log_event(
            "close_realized_pnl",
            new_trade_id("CLOSE"),
            ts=trade_ts,
            arm=current_arm or "",
            side=side_norm,
            qty=int(qty or 1),
            fill_px=exit_px,
            expected_px=exit_px,
            reason=gate_reason or "realized_pnl",
            pnl_delta=float(pnl_delta),
            entry_px=None if entry_px is None else float(entry_px),
            exit_px=float(exit_px),
            caps=list(caps or []),
            net_at_close=None if net is None else int(net),
            pos_entry_ts=pos_entry_ts or "",
            tags=tags_str,
        )
    except Exception:
        pass

    # paths
    try:
        trade_log_csv_abs = os.path.abspath(trade_log_csv) if trade_log_csv else os.path.abspath(os.path.join("results", "trades.csv"))
    except Exception:
        trade_log_csv_abs = os.path.abspath(os.path.join("results", "trades.csv"))

    rich_dir = os.path.abspath("results")
    rich_csv_abs = os.path.join(rich_dir, "trades_rich.csv")
    try:
        os.makedirs(rich_dir, exist_ok=True)
    except Exception:
        pass

    # trade_log_core (rich)
    info = None
    try:
        if callable(_log_trade_core):
            info = _log_trade_core(
                side=side_norm,  # BUY/SELL
                qty=qty,
                entry_px=entry_px if entry_px is not None else 0.0,
                exit_px=exit_px,
                pnl=pnl_delta,
                reason=(gate_reason or "realized_pnl"),
                timestamp=trade_ts,
                symbol="ES",
                strategy="",
                arm=(current_arm or ""),
                stop_px=None,
                target_px=None,
                risk_usd=None,
                notes=tags_str,
                trades_path=rich_csv_abs,
                baseline=False,
            )
    except Exception as e:
        info = None
        if logger is not None:
            try:
                logger.error(f"[trade_bridge] trade_log_core.log_trade failed: {type(e).__name__}: {e}")
            except Exception:
                pass

    # compute R
    trade_R = None
    try:
        if isinstance(info, dict) and info.get("R") is not None:
            trade_R = float(info["R"])
    except Exception:
        trade_R = None

    if trade_R is None:
        try:
            risk_dollars = float(risk_ticks) * float(tick_size) * ES_MULTIPLIER * max(int(qty or 1), 1)
            trade_R = float(pnl_delta) / float(risk_dollars) if risk_dollars > 0 else 0.0
        except Exception:
            trade_R = 0.0

    # canonical trades.csv
    try:
        _canon_trades_writer_append(
            trade_log_csv_abs,
            {
                "timestamp": trade_ts.isoformat(timespec="seconds"),
                "side": _emit_side_for_canon(side_norm),
                "qty": int(qty or 1),
                "entry_px": f"{(entry_px if entry_px is not None else 0.0):.2f}",
                "exit_px": f"{float(exit_px):.2f}",
                "pnl": f"{float(pnl_delta):.2f}",
                "R": f"{float(trade_R):.6f}",
                "tags": tags_str,
            },
        )
        # mirror into trades_ledger.csv (same canonical 8-col row; append-only ledger)
        try:
            _ledger_append(
                TRADES_LEDGER_PATH,
                {
                    "timestamp": trade_ts.isoformat(timespec="seconds"),
                    "side": _emit_side_for_canon(side_norm),
                    "qty": str(int(qty or 1)),
                    "entry_px": f"{float(entry_px if entry_px is not None else 0.0):.2f}",
                    "exit_px": f"{float(exit_px):.2f}",
                    "pnl": f"{float(pnl_delta):.2f}",
                    "R": f"{float(trade_R):.6f}",
                    "tags": tags_str or "",
                },
            )
        except Exception:
            pass


        # ALSO append to append-only ledger (audit/backtest history)
        try:
            _canon_trades_writer_append(
                        TRADES_LEDGER_PATH,
                        {
                            "timestamp": trade_ts.isoformat(timespec="seconds"),
                            "side": _emit_side_for_canon(side_norm),
                            "qty": int(qty or 1),
                            "entry_px": f"{(entry_px if entry_px is not None else 0.0):.2f}",
                            "exit_px": f"{float(exit_px):.2f}",
                            "pnl": f"{float(pnl_delta):.2f}",
                            "R": f"{float(trade_R):.6f}",
                            "tags": tags_str,
                        },
                    )
        except Exception:
            pass
    except Exception as e:
        try:
            _log_trade_legacy(
                csv_path=trade_log_csv_abs,
                timestamp=trade_ts,
                side="LONG" if side_norm == "BUY" else ("SHORT" if side_norm == "SELL" else "?"),
                qty=int(qty or 1),
                entry_px=entry_px if entry_px is not None else 0.0,
                exit_px=float(exit_px),
                pnl=float(pnl_delta),
                tags=tags_str,
                risk_ticks=float(risk_ticks or DEFAULT_RISK_TICKS),
                tick_value=DEFAULT_TICK_VALUE,
            )

            # ALSO append to ledger (legacy fallback path)
            try:
                _log_trade_legacy(
                                csv_path=TRADES_LEDGER_PATH,
                                timestamp=trade_ts,
                                side="LONG" if side_norm == "BUY" else ("SHORT" if side_norm == "SELL" else "?"),
                                qty=int(qty or 1),
                                entry_px=entry_px if entry_px is not None else 0.0,
                                exit_px=float(exit_px),
                                pnl=float(pnl_delta),
                                tags=tags_str,
                                risk_ticks=float(risk_ticks or DEFAULT_RISK_TICKS),
                                tick_value=DEFAULT_TICK_VALUE,
                            )
            except Exception:
                pass
        except Exception:
            if logger is not None:
                try:
                    logger.error("[trade_bridge] canonical+legacy trade logging failed")
                except Exception:
                    pass

    # Update DayRisk / WeekState
    try:
        _safe_set(day_risk, "day_R", _safe_get(day_risk, "day_R", 0.0) + float(trade_R))
    except Exception:
        pass

    try:
        _safe_set(week_state, "week_R", _safe_get(week_state, "week_R", 0.0) + float(trade_R))
    except Exception:
        pass

    # Update loss counters
    try:
        if float(trade_R) < 0:
            cur_consec = int(getattr(day_risk, "consec_losses", 0) or 0)
            _safe_set(day_risk, "consec_losses", cur_consec + 1)
            _safe_set(day_risk, "last_loss_ts", float(time.time()))
        elif float(trade_R) > 0:
            _safe_set(day_risk, "consec_losses", 0)
    except Exception:
        pass

    # Tiny meta learner
    try:
        if meta is not None:
            prev_ema = float(getattr(meta, "ema_R", 0.0) or 0.0)
            alpha = 0.2
            new_ema = (1.0 - alpha) * prev_ema + alpha * float(trade_R)
            setattr(meta, "ema_R", new_ema)

            prev_n = int(getattr(meta, "n_trades", 0) or 0)
            setattr(meta, "n_trades", prev_n + 1)
    except Exception:
        pass

    # Bandit update + persist
    try:
        if bandit is not None and current_arm:
            reward = max(-3.0, min(3.0, float(trade_R)))
            update_fn = getattr(bandit, "update", None)
            if callable(update_fn):
                update_fn(current_arm, reward)
                try:
                    save_thompson(learn_model_path, bandit)
                except Exception as e:
                    if logger is not None:
                        try:
                            logger.error(f"[bandit_update] save_thompson failed: {e}")
                        except Exception:
                            pass
    except Exception as e:
        if logger is not None:
            try:
                logger.error(f"[bandit_update] failed: {e}")
            except Exception:
                pass

    return (
        last_acct_realized,
        last_trade_close_ts,
        trades_today,
        total_trades,
        running_pnl_today,
        wins_today,
        losses_today,
        day_risk,
        week_state,
        bandit,
        meta,
        current_arm,
        current_side,
    )

def handle_realized_pnl_event(*args, **kwargs):
    # missing 'button' the program expects
    try:
        log_event("REALIZED_PNL", args=args, **kwargs)
    except Exception:
        pass
