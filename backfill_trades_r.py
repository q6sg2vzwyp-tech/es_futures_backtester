#!/usr/bin/env python3
import csv
from pathlib import Path

# --- CONFIG (match your bot) ---
RISK_TICKS = 12        # same as --risk-ticks
TICK_SIZE = 0.25       # same as --tick-size
MULTIPLIER = 50.0      # ES point value
TRADES_PATH = Path("results") / "trades.csv"
OUT_PATH = Path("results") / "trades_clean.csv"
# -------------------------------

def normalize_side(side: str) -> str:
    s = (side or "").upper().strip()
    if s in {"BUY", "LONG"}:
        return "LONG"
    if s in {"SELL", "SHORT"}:
        return "SHORT"
    return s or "?"


def main():
    if not TRADES_PATH.exists():
        print(f"Missing {TRADES_PATH}")
        return

    with TRADES_PATH.open("r", newline="", encoding="utf-8") as f_in, \
         OUT_PATH.open("w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or [
            "timestamp", "symbol", "side", "qty", "entry_px", "exit_px",
            "stop_px", "target_px", "pnl", "risk_usd", "R",
            "strategy", "arm", "reason", "notes",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            # Normalize side
            row["side"] = normalize_side(row.get("side", ""))

            # Parse pnl / risk / R if available
            pnl_str = row.get("pnl", "")
            R_str = row.get("R", "")
            risk_str = row.get("risk_usd", "")

            try:
                pnl = float(pnl_str)
            except Exception:
                writer.writerow(row)
                continue

            # If we already have a numeric R, leave it alone
            try:
                _ = float(R_str)
                writer.writerow(row)
                continue
            except Exception:
                pass

            # Try to compute risk_usd from stop distance if stop_px/entry_px present
            risk_usd = None
            try:
                entry_px = float(row.get("entry_px", "") or "nan")
                stop_px = float(row.get("stop_px", "") or "nan")
                if entry_px == entry_px and stop_px == stop_px:  # not NaN
                    pts = abs(entry_px - stop_px)
                    ru = pts * MULTIPLIER * abs(float(row.get("qty", 1) or 1))
                    if ru > 0:
                        risk_usd = ru
            except Exception:
                risk_usd = None

            # If still no risk_usd, fall back to 1R = RISK_TICKS * TICK_SIZE * MULTIPLIER
            if risk_usd is None or risk_usd <= 0:
                ru = RISK_TICKS * TICK_SIZE * MULTIPLIER * abs(float(row.get("qty", 1) or 1))
                risk_usd = ru if ru > 0 else None

            if risk_usd and risk_usd > 0:
                R_val = pnl / risk_usd
                row["risk_usd"] = f"{risk_usd:.2f}"
                row["R"] = f"{R_val:.6f}"
            else:
                # leave R blank if we truly can't compute
                row["R"] = row.get("R", "")

            writer.writerow(row)

    print(f"Wrote cleaned file to {OUT_PATH}")
    print("If everything looks good, you can replace trades.csv with trades_clean.csv")


if __name__ == "__main__":
    main()

