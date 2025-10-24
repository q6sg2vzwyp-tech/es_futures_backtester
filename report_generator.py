"""
Minimal, dependency-light report generator for batch_runner.py

Creates:
- PDF report (plots + KPI table)
- HTML summary page (links to CSVs and embeds charts)

Expected input: a trades DataFrame with at least columns:
  ['EntryTime','ExitTime','Side','EntryPrice','ExitPrice','Qty','NetPnL','TimeOfDay']
Optionally: 'Strategy', 'WF_Period', 'ParamsJSON'.

Usage:
    from report_generator import generate_summary_report
    generate_summary_report(df, out_dir="results/Trend_Following", title="Trend — Walk-Forward")

Notes:
- No seaborn; only matplotlib + pandas.
- Assumes timestamps are timezone-naive or consistent.
- Robust to empty/partial data: skips plots that can't be made.
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fmt_money(x: float) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)


def _kpis(trades: pd.DataFrame) -> dict[str, Any]:
    k = {
        "trades": len(trades) if trades is not None else 0,
        "net_pnl": float(trades["NetPnL"].sum()) if len(trades) else 0.0,
        "gross_pnl": (
            float(trades.get("GrossPnL", pd.Series(dtype=float)).sum())
            if len(trades) and "GrossPnL" in trades.columns
            else np.nan
        ),
        "fees": (
            float(trades.get("Fees", pd.Series(dtype=float)).sum())
            if len(trades) and "Fees" in trades.columns
            else np.nan
        ),
        "win_rate": None,
        "avg_trade": None,
        "max_dd": None,
        "sharpe": None,
    }

    if len(trades):
        wins = (trades["NetPnL"] > 0).sum()
        k["win_rate"] = wins / len(trades)
        k["avg_trade"] = trades["NetPnL"].mean()

        # Build equity curve by exit time
        eq = trades.copy()
        eq["ExitTime"] = pd.to_datetime(eq["ExitTime"])  # ensure datetime
        eq = eq.sort_values("ExitTime")
        daily = eq.set_index("ExitTime")["NetPnL"].resample("1D").sum().fillna(0.0)
        cum = daily.cumsum()
        if len(cum):
            peak = cum.cummax()
            dd = cum - peak
            k["max_dd"] = float(dd.min())
            # Sharpe (daily); avoid division by zero
            if daily.std(ddof=1) > 0:
                k["sharpe"] = float((daily.mean() / daily.std(ddof=1)) * np.sqrt(252))
            else:
                k["sharpe"] = np.nan
        else:
            k["max_dd"] = np.nan
            k["sharpe"] = np.nan
    return k


def _plot_equity_curve(ax, trades: pd.DataFrame):
    if trades is None or trades.empty:
        ax.text(0.5, 0.5, "No trades to plot", ha="center", va="center")
        return
    df = trades.copy()
    df["ExitTime"] = pd.to_datetime(df["ExitTime"])  # ensure datetime
    daily = df.set_index("ExitTime")["NetPnL"].resample("1D").sum().fillna(0.0)
    equity = daily.cumsum()
    ax.plot(equity.index, equity.values)
    ax.set_title("Equity Curve (Daily NetPnL CumSum)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")


def _plot_pnl_hist(ax, trades: pd.DataFrame):
    if trades is None or trades.empty:
        ax.text(0.5, 0.5, "No trades to plot", ha="center", va="center")
        return
    ax.hist(trades["NetPnL"].astype(float).values, bins=50)
    ax.set_title("Distribution of Trade NetPnL")
    ax.set_xlabel("NetPnL per Trade ($)")
    ax.set_ylabel("Count")


def _plot_pnl_by_tod(ax, trades: pd.DataFrame):
    if trades is None or trades.empty or "TimeOfDay" not in trades.columns:
        ax.text(0.5, 0.5, "Time-of-day data unavailable", ha="center", va="center")
        return
    g = trades.groupby("TimeOfDay")["NetPnL"].sum().sort_index()
    g.plot(kind="bar", ax=ax)
    ax.set_title("PnL by Time of Day (Entry)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("NetPnL ($)")


def _plot_pnl_by_period(ax, trades: pd.DataFrame):
    if trades is None or trades.empty or "WF_Period" not in trades.columns:
        ax.text(0.5, 0.5, "Walk-forward period data unavailable", ha="center", va="center")
        return
    g = trades.groupby("WF_Period")["NetPnL"].sum()
    g.plot(kind="bar", ax=ax)
    ax.set_title("PnL by Walk-Forward Period")
    ax.set_xlabel("WF Period #")
    ax.set_ylabel("NetPnL ($)")


def _plot_side_attrib(ax, trades: pd.DataFrame):
    if trades is None or trades.empty or "Side" not in trades.columns:
        ax.text(0.5, 0.5, "Side breakdown unavailable", ha="center", va="center")
        return
    g = trades.groupby("Side")["NetPnL"].sum()
    g.plot(kind="bar", ax=ax)
    ax.set_title("PnL by Side")
    ax.set_xlabel("Side")
    ax.set_ylabel("NetPnL ($)")


def _save_kpi_csv(kpis: dict[str, Any], out_dir: str, title: str) -> str:
    df = pd.DataFrame([kpis])
    path = os.path.join(out_dir, "summary_kpis.csv")
    df.to_csv(path, index=False)
    return path


def _write_html(
    trades: pd.DataFrame,
    kpis: dict[str, Any],
    out_dir: str,
    title: str,
    images: dict[str, str],
    extras=None,
) -> str:
    html_path = os.path.join(out_dir, "report.html")

    def _row(label, val):
        if isinstance(val, float):
            if label in {"win_rate"}:
                val = f"{val*100:.2f}%"
            elif label in {"sharpe"}:
                val = f"{val:.2f}"
            else:
                val = _fmt_money(val)
        return f"<tr><td>{label}</td><td>{val}</td></tr>"

    rows = "".join(_row(k, v) for k, v in kpis.items())
    imgs = "".join(
        f"<h3>{name}</h3><img src='{os.path.basename(p)}' style='max-width:100%;'/>"
        for name, p in images.items()
    )
    extras_html = f"<pre>{json.dumps(extras, indent=2)}</pre>" if extras else ""

    html = f"""
    <html>
    <head><meta charset='utf-8'><title>{title}</title></head>
    <body>
      <h1>{title}</h1>
      <h2>Key Metrics</h2>
      <table border='1' cellpadding='6' cellspacing='0'>
        <tbody>
          {rows}
        </tbody>
      </table>
      <h2>Charts</h2>
      {imgs}
      <h2>Extras</h2>
      {extras_html}
    </body>
    </html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def generate_summary_report(
    trades: pd.DataFrame, out_dir: str, title: str, extras: dict[str, Any] | None = None
) -> dict[str, str]:
    """Create PDF + HTML reports for a given trades DataFrame.

    Returns dict of artifact paths.
    """
    _ensure_dir(out_dir)
    artifacts = {}

    # KPIs
    kpis = _kpis(trades if trades is not None else pd.DataFrame())
    artifacts["kpi_csv"] = _save_kpi_csv(kpis, out_dir, title)

    # Paths for images
    img_equity = os.path.join(out_dir, "equity_curve.png")
    img_hist = os.path.join(out_dir, "pnl_hist.png")
    img_tod = os.path.join(out_dir, "pnl_by_tod.png")
    img_wf = os.path.join(out_dir, "pnl_by_wf.png")
    img_side = os.path.join(out_dir, "pnl_by_side.png")

    # Generate charts
    # Equity
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    _plot_equity_curve(ax1, trades)
    fig1.tight_layout()
    fig1.savefig(img_equity, dpi=150)
    plt.close(fig1)

    # Hist
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    _plot_pnl_hist(ax2, trades)
    fig2.tight_layout()
    fig2.savefig(img_hist, dpi=150)
    plt.close(fig2)

    # PnL by Time of Day
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    _plot_pnl_by_tod(ax3, trades)
    fig3.tight_layout()
    fig3.savefig(img_tod, dpi=150)
    plt.close(fig3)

    # PnL by WF Period
    fig4, ax4 = plt.subplots(figsize=(9, 4))
    _plot_pnl_by_period(ax4, trades)
    fig4.tight_layout()
    fig4.savefig(img_wf, dpi=150)
    plt.close(fig4)

    # PnL by Side
    fig5, ax5 = plt.subplots(figsize=(9, 4))
    _plot_side_attrib(ax5, trades)
    fig5.tight_layout()
    fig5.savefig(img_side, dpi=150)
    plt.close(fig5)

    # PDF assembly
    pdf_path = os.path.join(out_dir, "report.pdf")
    with PdfPages(pdf_path) as pdf:
        for pth, ttl in [
            (img_equity, "Equity Curve"),
            (img_hist, "Trade NetPnL Distribution"),
            (img_tod, "PnL by Time of Day"),
            (img_wf, "PnL by Walk-Forward Period"),
            (img_side, "PnL by Side"),
        ]:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            img = plt.imread(pth)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(ttl)
            pdf.savefig(fig)
            plt.close(fig)

        # KPI table page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        lines = [
            f"Trades: {kpis['trades']}",
            f"Net PnL: {_fmt_money(kpis['net_pnl'])}",
            f"Gross PnL: {_fmt_money(kpis['gross_pnl'])}",
            f"Fees: {_fmt_money(kpis['fees'])}",
            (
                f"Win Rate: {kpis['win_rate']*100:.2f}%"
                if isinstance(kpis["win_rate"], (int, float))
                else "Win Rate: n/a"
            ),
            (
                f"Avg Trade: {_fmt_money(kpis['avg_trade'])}"
                if kpis["avg_trade"] is not None
                else "Avg Trade: n/a"
            ),
            (
                f"Max Drawdown: {_fmt_money(kpis['max_dd'])}"
                if kpis["max_dd"] is not None
                else "Max Drawdown: n/a"
            ),
            (
                f"Sharpe (daily): {kpis['sharpe']:.2f}"
                if isinstance(kpis["sharpe"], (int, float))
                else "Sharpe (daily): n/a"
            ),
        ]
        y = 0.95
        for L in lines:
            ax.text(0.05, y, L, fontsize=14)
            y -= 0.06
        ax.set_title("Key Performance Metrics", fontsize=18)
        pdf.savefig(fig)
        plt.close(fig)

    artifacts["pdf"] = pdf_path

    # HTML (simple)
    images = {
        "Equity Curve": img_equity,
        "Trade NetPnL Distribution": img_hist,
        "PnL by Time of Day": img_tod,
        "PnL by Walk-Forward Period": img_wf,
        "PnL by Side": img_side,
    }
    artifacts["html"] = _write_html(trades, kpis, out_dir, title, images, extras)

    print(f"✅ Report generated → {pdf_path}")
    print(f"✅ HTML summary → {artifacts['html']}")
    return artifacts
