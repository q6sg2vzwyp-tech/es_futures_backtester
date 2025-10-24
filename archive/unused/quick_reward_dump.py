# quick_reward_dump.py - skims logs fast and writes a simple CSV of rewards + a summary
import csv
import datetime
import os
import re
import sys
import logging
logger = logging.getLogger(__name__)

LOGDIR = sys.argv[1] if len(sys.argv) > 1 else r".\logs"
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 3  # lookback days
OUTDIR = os.path.join(LOGDIR, "reports")
os.makedirs(OUTDIR, exist_ok=True)
outcsv = os.path.join(OUTDIR, f"rewards_last{DAYS}d.csv")

now = datetime.datetime.now()
cutoff = now - datetime.timedelta(days=DAYS)
reward_re = re.compile(
    r"\[REWARD\]\s+arm=(\w+)\s+R=([-\d\.]+).*?slipR=([-\d\.]+).*?hrs=([-\d\.]+).*?maeR=([-\d\.]+).*?dailyR=([-\d\.]+)"
)
ts_re = re.compile(r"^\[(?:INFO|HB|\w+)\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")
entry_re = re.compile(r"\[INFO\].*Entered\s+(LONG|SHORT)\s+@\s+([0-9]+\.[0-9]+)")
exit_re = re.compile(r"\[INFO\].*\[STOP\]\s+.*?Modified stop|[REWARD]")

rows = []
summary = {}
scanned = 0
matched = 0
for root, _, files in os.walk(LOGDIR):
    for name in files:
        if not (
            name.startswith("bot_stdout_") or name.startswith("runtime_")
        ) and not name.endswith(".log"):
            continue
        fpath = os.path.join(root, name)
        try:
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
        except Exception as e:

            logger.debug('Swallowed exception in quick_reward_dump.py: %s', e)
 continue
        if mtime < cutoff:
            continue
        scanned += 1
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                last_ts = None
                last_entry = None
                for line in f:
                    mts = ts_re.search(line)
                    if mts:
                        last_ts = mts.group(1)
                    m = reward_re.search(line)
                    if m:
                        arm, R, slipR, hrs, maeR, dayR = m.groups()
                        matched += 1
                        R = float(R)
                        dayR = float(dayR)
                        rows.append(
                            [
                                last_ts or "",
                                arm,
                                R,
                                float(slipR),
                                float(hrs),
                                float(maeR),
                                dayR,
                                name,
                            ]
                        )
                        s = summary.setdefault(arm, {"N": 0, "sumR": 0.0})
                        s["N"] += 1
                        s["sumR"] += R
        except Exception as e:
            print(f"[WARN] Could not read {fpath}: {e}")

# write CSV
with open(outcsv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestamp", "arm", "R", "slipR", "hrs", "maeR", "dailyR", "file"])
    w.writerows(rows)

# print console summary
print(f"[OK] Scanned recent files: {scanned}  | Found reward lines: {matched}")
print("Per-arm summary:")
for arm, s in sorted(summary.items(), key=lambda x: -x[1]["sumR"]):
    avg = s["sumR"] / s["N"] if s["N"] else 0.0
    print(f"  {arm:10s}  N={s['N']:3d}  sumR={s['sumR']:+.2f}  avgR={avg:+.3f}")
print(f"[CSV] {outcsv}")
