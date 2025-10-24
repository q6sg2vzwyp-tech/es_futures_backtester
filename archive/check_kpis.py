"""
check_kpis.py — Read the latest paper_trader KPIs and print a dashboard.
- Auto-detects latest results/ib_session_* directory (or use --session).
- Reads kpis.json (Sharpe/MAR/DD/size/etc).
- Optionally cross-checks trades_log.csv to show hit rate, avg win/loss, PF.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

DEF_RESULTS_ROOT = Path("results")


def find_latest_session(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("ib_session_")]
    if not candidates:
        return None
    # Sort by directory name (they’re timestamped) and mtime as fallback
    candidates.sort(key=lambda p: (p.name, p.stat().st_mtime), reverse=True)
    return candidates[0]


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def pct(x: float) -> str:
    return f"{x:.1f}%"


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def summarize_trades(trades_csv: Path, window: int | None):
    """
    Returns dict with count, winrate, avg_win, avg_loss, profit_factor, mean, std, sharpe
    based on RealizedPnL from the last N filled rows (window if given).
    """
    if not trades_csv.exists():
        return None

    pnl = []
    with trades_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [row for row in r if (row.get("RealizedPnL") not in (None, "", "None"))]
    if not rows:
        return None

    if window is not None and window > 0:
        rows = rows[-window:]

    for row in rows:
        x = safe_float(row.get("RealizedPnL"))
        if x is not None:
            pnl.append(x)

    if not pnl:
        return None

    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    count = len(pnl)
    winrate = 100.0 * len(wins) / count if count else 0.0
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    gross_win = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    mu = mean(pnl)
    sigma = pstdev(pnl) if len(pnl) > 1 else 0.0
    sharpe = (mu / sigma) if sigma > 0 else 0.0

    return dict(
        count=count,
        winrate=winrate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        mean=mu,
        std=sigma,
        sharpe=sharpe,
    )


def main():
    ap = argparse.ArgumentParser(description="Print KPI dashboard for latest paper_trader session")
    ap.add_argument(
        "--session",
        default="",
        help="Path to a specific results session (results/ib_session_YYYYmmdd_HHMMSS)",
    )
    ap.add_argument(
        "--results-root",
        default=str(DEF_RESULTS_ROOT),
        help="Root results directory (default: results)",
    )
    ap.add_argument("--window", type=int, default=120, help="Trade window for stats (default: 120)")
    ap.add_argument("--no-trade-stats", action="store_true", help="Skip reading trades_log.csv")
    args = ap.parse_args()

    root = Path(args.results_root)
    session_dir = Path(args.session) if args.session else find_latest_session(root)
    if not session_dir or not session_dir.exists():
        print(
            f"[ERR] Could not find a results session under '{root}'. Run paper_trader first.",
            file=sys.stderr,
        )
        sys.exit(1)

    kpi_path = session_dir / "kpis.json"
    trades_csv = session_dir / "trades_log.csv"

    if not kpi_path.exists():
        print(f"[ERR] Missing kpis.json in {session_dir}", file=sys.stderr)
        sys.exit(2)

    kpi = load_json(kpi_path)
    promo = kpi.get("promo_thresholds", {})
    sharpe_kpi = kpi.get("sharpe_per_trade", 0.0)
    mar_kpi = kpi.get("mar", 0.0)
    maxdd = kpi.get("max_drawdown_pct", 0.0)
    eq = kpi.get("equity", 0.0)
    size_level = kpi.get("size_level", 0)
    max_qty = kpi.get("max_qty", 1)
    trades_n = kpi.get("trades", 0)

    print("=" * 64)
    print(" PAPER TRADER — KPI DASHBOARD")
    print("=" * 64)
    print(f" Session: {session_dir.name}")
    print(f" Equity:  {fmt_money(eq)}")
    print(f" Trades:  {trades_n}")
    print(f" Size:    level {size_level} (maxQty={max_qty})")
    print("-" * 64)
    print(f" Sharpe/trade: {sharpe_kpi:.2f}")
    print(f" MAR:          {mar_kpi:.2f}")
    print(f" Max DD:       {maxdd:.2f}%")
    print("-" * 64)

    # Optional cross-check from trades_log.csv
    if not args.no_trade_stats:
        stats = summarize_trades(trades_csv, args.window)
        if stats:
            print(f" Window (last {stats['count']} trades)")
            print(f"   Win rate:     {stats['winrate']:.1f}%")
            print(f"   Avg win:      {fmt_money(stats['avg_win'])}")
            print(f"   Avg loss:     {fmt_money(stats['avg_loss'])}")
            print(
                f"   Profit factor:{'∞' if stats['profit_factor']==float('inf') else f'{stats['profit_factor']:.2f}'}"
            )
            print(f"   Mean/Std:     {fmt_money(stats['mean'])} / {fmt_money(stats['std'])}")
            print(f"   Sharpe (calc):{stats['sharpe']:.2f}")
            print("-" * 64)
        else:
            print("(No filled trades found in trades_log.csv yet.)")
            print("-" * 64)

    # Promotion readiness
    sharpe_min = float(promo.get("sharpe_min", 1.5))
    mar_min = float(promo.get("mar_min", 0.5))
    maxdd_cap = float(promo.get("maxdd_pct", 10.0))
    window_trades = int(promo.get("window_trades", args.window))

    ready = (
        trades_n >= window_trades
        and sharpe_kpi >= sharpe_min
        and mar_kpi >= mar_min
        and maxdd <= maxdd_cap
    )

    print(" PROMOTION READINESS")
    print(
        f"   Window trades: {trades_n}/{window_trades}  {'OK' if trades_n >= window_trades else 'WAIT'}"
    )
    print(
        f"   Sharpe ≥ {sharpe_min}:   {sharpe_kpi:.2f}  {'OK' if sharpe_kpi >= sharpe_min else 'NO'}"
    )
    print(f"   MAR ≥ {mar_min}:       {mar_kpi:.2f}  {'OK' if mar_kpi >= mar_min else 'NO'}")
    print(f"   MaxDD ≤ {maxdd_cap}%:  {maxdd:.2f}%  {'OK' if maxdd <= maxdd_cap else 'NO'}")
    print("-" * 64)
    print(" READY TO PROMOTE?", "YES ✅" if ready else "NO ❌")
    print("=" * 64)


if __name__ == "__main__":
    main()
