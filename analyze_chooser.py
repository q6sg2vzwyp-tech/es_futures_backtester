#!/usr/bin/env python
# analyze_chooser.py — chart LinUCB learning over time (avgR and counts) from snapshots
import glob
import json
import os
import sys

import matplotlib.pyplot as plt

snap_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/chooser_snapshots"
files = sorted(glob.glob(os.path.join(snap_dir, "linucb_es_*.json")))
if not files:
    print("No snapshots found in", snap_dir)
    sys.exit(0)

dates = []
arms = set()
avgR = {}
counts = {}


def ensure(arm):
    if arm not in avgR:
        avgR[arm] = []
        counts[arm] = []


for f in files:
    with open(f, encoding="utf-8") as fh:
        state = json.load(fh)
    d = os.path.basename(f).split("_")[-1].split(".")[0]
    dates.append(d)
    # tolerate either flat dumps (counts/reward_sum at top) or nested under 'chooser'
    counts_map = state.get("counts", {}) or state.get("chooser", {}).get("counts", {})
    reward_sum = state.get("reward_sum", {}) or state.get("chooser", {}).get("reward_sum", {})
    for arm in set(list(counts_map.keys()) + list(reward_sum.keys())):
        arms.add(arm)
        ensure(arm)
        c = counts_map.get(arm, 0)
        rs = reward_sum.get(arm, 0.0)
        avg = (rs / c) if c else 0.0
        avgR[arm].append(avg)
        counts[arm].append(c)

# Plot avgR
plt.figure(figsize=(10, 5))
for arm in sorted(arms):
    plt.plot(dates, avgR[arm], label=f"{arm} avgR")
plt.xticks(rotation=45)
plt.ylabel("Average Reward (R)")
plt.title("Chooser learning — avgR by arm")
plt.legend()
plt.tight_layout()
plt.show()

# Plot counts
plt.figure(figsize=(10, 5))
for arm in sorted(arms):
    plt.plot(dates, counts[arm], label=f"{arm} count")
plt.xticks(rotation=45)
plt.ylabel("Trades per arm (cumulative)")
plt.title("Chooser learning — counts by arm")
plt.legend()
plt.tight_layout()
plt.show()
