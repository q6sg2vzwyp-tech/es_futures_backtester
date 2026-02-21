from __future__ import annotations

import argparse

def build_argparser():
    ap = argparse.ArgumentParser(description="ES Paper Trader (session-aware + Thompson learner + rails)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--clientId", type=int, default=111)
    ap.add_argument("--symbol", default="ES")
    ap.add_argument("--local-symbol", dest="local_symbol", default="")
    ap.add_argument("--place-orders", action="store_true")
    ap.add_argument("--tif", default="GTC")
    ap.add_argument("--outsideRth", action="store_true")

    # Auto roll / contract selection
    ap.add_argument(
        "--roll-by-volume",
        action="store_true",
        help=(
            "When using --symbol (and not --local-symbol), pick the front contract by "
            "liquidity (recent daily volume + futures open interest) among nearest expiries "
            "instead of purely earliest expiry."
        ),
    )

    # Sizing & Risk
    ap.add_argument("--acct-base", type=float, default=30000.0)
    ap.add_argument("--risk-pct", type=float, default=0.015)
    ap.add_argument("--scale-step", type=float, default=10000.0)
    ap.add_argument("--start-contracts", type=int, default=2)
    ap.add_argument("--max-contracts", type=int, default=6)
    ap.add_argument("--static-size", action="store_true")
    ap.add_argument("--qty", type=float, default=2.0)
    ap.add_argument("--risk-ticks", type=int, default=12)
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--tp-R", type=float, default=1.0)

    # Margin awareness
    ap.add_argument("--margin-per-contract", type=float, default=22000.0)
    ap.add_argument("--margin-reserve-pct", type=float, default=0.10)

    # Strategy gates
    ap.add_argument("--enable-arms", default="trend,breakout")
    ap.add_argument("--gate-adx", type=float, default=19.0)
    ap.add_argument("--gate-atrp", type=float, default=0.000055)
    ap.add_argument("--gate-bbbw", type=float, default=0.0)  # 0 disables

    # Anti-burst & day/session rails (slower base cadence)
    ap.add_argument("--min-seconds-between-entries", type=int, default=35)
    ap.add_argument(
        "--post-flat-cooldown-sec",
        type=int,
        default=15,
        help="Extra cooldown seconds after any flatten/fill before re-entry.",
    )
    ap.add_argument("--max-trades-per-day", type=int, default=10)
    ap.add_argument("--day-loss-cap-R", type=float, default=3.0)
    ap.add_argument("--weekly-cap-mult", type=float, default=3.0,
                    help="Weekly R cap: -weekly_cap_mult * abs(day_loss_cap_R)")
    ap.add_argument("--max-consec-losses", type=int, default=3)
    ap.add_argument("--learn-while-capped", action="store_true",
                help="Update meta-learner with zero-reward on veto/caps")
    ap.add_argument("--strategy-cooldown-sec", type=int, default=180)

    # Risk governance extras
    ap.add_argument("--pos-age-cap-sec", type=int, default=900)
    ap.add_argument("--pos-age-minR", type=float, default=0.5)
    ap.add_argument("--hwm-stepdown", action="store_true")
    ap.add_argument("--hwm-stepdown-dollars", type=float, default=5000.0)

    # Trading window (24/5) + optional TOD blackouts
    ap.add_argument("--trade-start-ct", default="00:00")
    ap.add_argument("--trade-end-ct", default="23:59")
    ap.add_argument("--tod-blackouts", default="")
    ap.add_argument(
        "--holidays-file",
        default=r".\data\calendar\exchange_holidays.txt",
        help="Optional YYYY-MM-DD list of full exchange holidays (one per line).",
    )

    # Order behavior
    ap.add_argument("--entry-slippage-ticks", type=int, default=2)
    ap.add_argument("--require-new-bar-after-start", action="store_true")
    ap.add_argument("--startup-delay-sec", type=int, default=0)
    ap.add_argument("--debounce-one-bar", action="store_true")

    # NEW: auto-promote parent LIMIT entry to MARKET after N seconds
    ap.add_argument(
        "--parent-to-mkt-sec",
        type=int,
        default=5,  # non-zero sane default to avoid stale parents
        help=(
            "If >0, convert a working parent LIMIT entry order to MARKET after "
            "this many seconds if still unfilled."
        ),
    )

    # Session resets (AM/PM segmentation only)
    ap.add_argument("--session-reset-cts", default="08:30,16:00,17:00")
    ap.add_argument("--daily-reset-ct", default="16:10")  # legacy

    # Connectivity & data
    ap.add_argument("--connect-timeout-sec", type=int, default=60)
    ap.add_argument("--timeout-sec", type=int, default=60)
    ap.add_argument("--connect-attempts", type=int, default=10)
    ap.add_argument("--force-delayed", action="store_true")
    ap.add_argument("--poll-hist-when-no-rt", action="store_true")
    ap.add_argument("--poll-interval-sec", type=int, default=10)
    ap.add_argument("--require-rt-before-trading", action="store_true")
    ap.add_argument("--rt-staleness-sec", type=int, default=45)

    # Learning
    ap.add_argument("--bandit", choices=["thompson"], default="thompson")
    ap.add_argument("--learn-mode", choices=["shadow", "advisory", "control"], default="advisory")
    ap.add_argument("--decay-half-life-trades", type=float, default=200.0)
    ap.add_argument("--learn-log", action="store_true")
    ap.add_argument("--learn-log-dir", default=r".\logs\learn")
    ap.add_argument(
        "--param-arms",
        default="",
        help=(
            "Semicolon-separated like "
            "'A:risk_ticks=10,tp_R=1.0,entry_slippage_ticks=1; "
            "B:risk_ticks=12,tp_R=1.2,entry_slippage_ticks=2'"
        ),
    )

    # PnL & equity sync
    ap.add_argument("--use-ib-pnl", action="store_true")
    ap.add_argument("--peak-dd-guard-pct", type=float, default=0.0)
    ap.add_argument("--day-guard-pct", type=float, default=0.0)
    ap.add_argument("--peak-dd-min-profit", type=float, default=2000.0)

    # Short guard rails & VWAP control (flags kept for compatibility)
    ap.add_argument("--short-guard-vwap-buffer-ticks", type=int, default=4)
    ap.add_argument("--short-guard-min-pullback-ticks", type=int, default=6)
    ap.add_argument("--short-guard-lookback-bars", type=int, default=60)
    ap.add_argument(
        "--no-short-guard-lower-high",
        dest="short_guard_lower_high",
        action="store_false",
    )
    ap.add_argument("--vwap-reset-on-session", action="store_true", default=True)
    ap.add_argument("--no-vwap-reset-on-session", dest="vwap_reset_on_session", action="store_false")

    # Safety
    ap.add_argument("--allow_live", action="store_true")

    # Risk profile selector (hook for future presets)
    ap.add_argument(
        "--risk-profile",
        choices=["balanced", "aggressive", "conservative"],
        default="balanced",
    )

    # News kill
    ap.add_argument("--news-file-kill", default=r".\data\kill\news_kill.json")
    ap.add_argument("--news-flatten-on-kill", action="store_true")
    ap.add_argument("--news-cancel-only", action="store_true")
    ap.add_argument("--news-blackouts", default="")
    ap.add_argument("--news-bulletin-listen", action="store_true")
    ap.add_argument(
        "--news-keywords",
        default=(
            "FOMC,rate,nonfarm,employment,inflation,CPI,PPI,ISM,PMI,"
            "Jerome Powell,press conference"
        ),
    )
    ap.add_argument("--news-kill-minutes", type=int, default=15)

    # Optional CSV hook for segmented logs
    ap.add_argument("--segment-trade-csv", default=r".\logs\trades_segmented.csv")
    return ap

def parse_args(argv: list[str] | None = None):
    """Convenience wrapper."""
    ap = build_argparser()
    return ap.parse_args(argv)
