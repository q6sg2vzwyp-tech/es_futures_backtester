from pathlib import Path

import pandas as pd

# <<< set this to your latest session folder >>>
SESSION = r"results\ib_session_20250820_143127"

bars = pd.read_csv(Path(SESSION) / "bars_1m.csv")
bars["Time"] = pd.to_datetime(bars["Time"], utc=True, errors="coerce")
bars = bars.dropna(subset=["Time"]).set_index("Time")

# EMA cross reconstruction
close = bars["Close"]
bars["ema8"] = close.ewm(span=8, adjust=False).mean()
bars["ema20"] = close.ewm(span=20, adjust=False).mean()
up = (bars["ema8"].shift(1) <= bars["ema20"].shift(1)) & (bars["ema8"] > bars["ema20"])
dn = (bars["ema8"].shift(1) >= bars["ema20"].shift(1)) & (bars["ema8"] < bars["ema20"])
cands = bars.loc[up | dn].copy()
cands["side"] = ["BUY" if u else "SELL" for u in up[up | dn]]

# RTH filter (Chicago 08:30–15:00)
ct = cands.index.tz_convert("America/Chicago")
in_rth = ((ct.hour > 8) | ((ct.hour == 8) & (ct.minute >= 30))) & (ct.hour < 15)
cands_rth = cands[in_rth].copy()

# Compare to what the bot actually signaled (if any)
sigs = pd.read_csv(Path(SESSION) / "signals.csv")
sigs["Time"] = pd.to_datetime(sigs["Time"], utc=True, errors="coerce").dt.floor("min")
sigs = sigs[sigs["Signal"].isin(["BUY", "SELL"])]

cands_rth["Minute"] = cands_rth.index.floor("min")
missed = cands_rth.merge(sigs[["Time", "Signal"]], how="left", left_on="Minute", right_on="Time")
missed = missed[missed["Signal"].isna()]  # crosses the bot didn’t signal

# Save reports
cands_rth[["Open", "High", "Low", "Close", "ema8", "ema20", "side"]].to_csv(
    Path(SESSION) / "candidate_crosses.csv"
)
missed[["Minute", "Close", "ema8", "ema20", "side"]].to_csv(
    Path(SESSION) / "missed_crosses.csv", index=False
)

print("Candidate EMA crosses during RTH:", len(cands_rth))
print("Missed (no bot signal):", len(missed))
print("Wrote candidate_crosses.csv and missed_crosses.csv")
