# cli_core.py
from __future__ import annotations

import argparse
import os
from typing import Optional


def _project_root_from_file() -> str:
    """
    Resolve a stable project root for defaults that should not depend on CWD.
    Assumes cli_core.py lives in the repo root. If you later move it into a
    subfolder, adjust accordingly.
    """
    return os.path.dirname(os.path.abspath(__file__))


def build_arg_parser(base_dir: Optional[str] = None) -> argparse.ArgumentParser:
    """
    Build the CLI parser for ES Paper Trader (modular rebuild).

    NOTE: base_dir is optional; it is only used to derive default paths.
    We prefer a stable default based on this file's location (repo root),
    not the process working directory, to avoid "log path drift".
    """
    p = argparse.ArgumentParser(description="ES Paper Trader (modular rebuild)")

    # --------------------
    # IBKR / Contract
    # --------------------
    p.add_argument("--ib-host", default="127.0.0.1")
    p.add_argument("--ib-port", type=int, default=4002)
    p.add_argument("--ib-client-id", type=int, default=111)

    # Prefer "auto" in modern workflows; if your code supports it,
    # this avoids hard-coding expired contracts.
    p.add_argument("--local-symbol", default="auto")
    p.add_argument("--exchange", default="GLOBEX")
    p.add_argument("--currency", default="USD")

    # --------------------
    # Modes
    # --------------------
    p.add_argument("--place-orders", action="store_true")
    p.add_argument("--use-ib-pnl", action="store_true")
    p.add_argument("--risk-profile", default="balanced")
    p.add_argument("--learn-mode", default="advisory", choices=["off", "advisory", "control"])

    # --------------------
    # Session / Time (CT)
    # --------------------
    p.add_argument("--trade-start-ct", default="8:35")
    p.add_argument("--trade-end-ct", default="15:00")
    p.add_argument("--session-reset-cts", default="08:30")
    p.add_argument("--vwap-reset-on-session", action="store_true")

    p.add_argument(
        "--auto-flat-ct",
        type=str,
        default=None,
        help="HH:MM CT; at/after this time the bot will flatten any open ES position and stop trading for the day",
    )
    p.add_argument(
        "--preclose-sweep-ct",
        type=str,
        default="15:55",
        help="HH:MM CT; safety sweep time: retry flatten-until-flat and then lock out new entries for the day",
    )
    p.add_argument(
        "--weekend-flatten",
        action="store_true",
        help="If set, any Sat/Sun will force flatten-until-flat and halt trading (prevents weekend carry).",
    )

    # --------------------
    # Cadence / Execution
    # --------------------
    p.add_argument("--min-seconds-between-entries", type=int, default=15)
    p.add_argument("--strategy-cooldown-sec", type=int, default=15)
    p.add_argument("--parent-to-mkt-sec", type=int, default=5)

    p.add_argument("--poll-hist-when-no-rt", action="store_true")
    p.add_argument("--poll-interval-sec", type=int, default=10)
    p.add_argument("--rt-staleness-sec", type=int, default=45)
    p.add_argument("--startup-delay-sec", type=int, default=0)

    # --------------------
    # Position sizing / Risk
    # --------------------
    p.add_argument("--start-contracts", type=int, default=2)
    p.add_argument("--max-contracts", type=int, default=6)
    p.add_argument("--risk-pct", type=float, default=0.015)
    p.add_argument("--risk-ticks", type=int, default=12)
    p.add_argument("--tp-R", type=float, default=1.0)
    p.add_argument("--tick-size", type=float, default=0.25)

    p.add_argument("--pos-age-cap-sec", type=int, default=900)
    p.add_argument("--pos-age-minR", type=float, default=0.5)

    p.add_argument("--hwm-stepdown", action="store_true")
    p.add_argument("--hwm-stepdown-dollars", type=float, default=5000.0)

    # --------------------
    # Risk rails
    # --------------------
    p.add_argument("--day-guard-pct", type=float, default=0.0)
    p.add_argument("--max-trades-per-day", type=int, default=20)
    p.add_argument("--max-consec-losses", type=int, default=6)
    p.add_argument("--day-loss-cap-R", type=float, default=5.0)
    p.add_argument("--weekly-cap-mult", type=float, default=4.0)
    p.add_argument("--post-flat-cooldown-sec", type=int, default=60)

    # --------------------
    # Learning / Logging
    # --------------------
    p.add_argument("--learn-while-capped", action="store_true")
    p.add_argument("--learn-log", action="store_true")

    if base_dir is None:
        base_dir = _project_root_from_file()

    p.add_argument("--learn-log-dir", default=os.path.join(base_dir, "logs", "learn"))
    p.add_argument("--use-bayes-best", action="store_true")

    p.add_argument(
        "--boost-mode",
        default="off",
        choices=["off", "normal", "war"],
        help="Dynamic risk scaling: off | normal | war",
    )

    return p


def parse_args(base_dir: Optional[str] = None):
    return build_arg_parser(base_dir=base_dir).parse_args()


def postprocess_args(args, logger=None):
    """
    Clamp unsafe / nonsensical parameters so runtime code stays clean.
    This is intentionally conservative to prevent accidental overtrading configs.
    """
    # Cadence clamps
    if getattr(args, "min_seconds_between_entries", 15) < 10:
        if logger:
            logger.warning(
                "[config] min_seconds_between_entries=%s is very low; clamping to 10.",
                args.min_seconds_between_entries,
            )
        args.min_seconds_between_entries = 10

    if getattr(args, "strategy_cooldown_sec", 15) < 10:
        if logger:
            logger.warning(
                "[config] strategy_cooldown_sec=%s is very low; clamping to 10.",
                args.strategy_cooldown_sec,
            )
        args.strategy_cooldown_sec = 10

    # Basic validity clamps
    if getattr(args, "start_contracts", 1) < 1:
        if logger:
            logger.warning("[config] start_contracts=%s invalid; clamping to 1.", args.start_contracts)
        args.start_contracts = 1

    if getattr(args, "max_contracts", args.start_contracts) < args.start_contracts:
        if logger:
            logger.warning(
                "[config] max_contracts=%s < start_contracts=%s; clamping max_contracts to start_contracts.",
                args.max_contracts,
                args.start_contracts,
            )
        args.max_contracts = args.start_contracts

    if getattr(args, "risk_ticks", 1) < 1:
        if logger:
            logger.warning("[config] risk_ticks=%s invalid; clamping to 1.", args.risk_ticks)
        args.risk_ticks = 1

    if getattr(args, "tp_R", 1.0) <= 0:
        if logger:
            logger.warning("[config] tp_R=%s invalid; clamping to 1.0.", args.tp_R)
        args.tp_R = 1.0

    if getattr(args, "day_loss_cap_R", 5.0) <= 0:
        if logger:
            logger.warning("[config] day_loss_cap_R=%s invalid; clamping to 5.0.", args.day_loss_cap_R)
        args.day_loss_cap_R = 5.0

    if getattr(args, "weekly_cap_mult", 4.0) < 1.0:
        if logger:
            logger.warning("[config] weekly_cap_mult=%s very low; clamping to 1.0.", args.weekly_cap_mult)
        args.weekly_cap_mult = 1.0

    return args
