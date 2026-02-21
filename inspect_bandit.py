#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_bandit.py

Quick snapshot of bandit arm performance from learn/bandit_state.json
so you can see which strategies are "making money".
"""

import os
import json
from learner_bandit import load_thompson

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BANDIT_PATH = os.path.join(BASE_DIR, "learn", "bandit_state.json")


def main() -> None:
    if not os.path.exists(BANDIT_PATH):
        print(f"bandit_state.json not found at:\n  {BANDIT_PATH}")
        print("Run the paper trader for a while so it can create it.")
        return

    bandit = load_thompson(BANDIT_PATH)
    if bandit is None:
        # Fall back to raw JSON
        with open(BANDIT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("bandit_state.json (raw):")
        print(json.dumps(data, indent=2, sort_keys=True))
        return

    mu = getattr(bandit, "mu", None)
    counts = getattr(bandit, "counts", None) or getattr(bandit, "n", None)

    if isinstance(mu, dict):
        rows = []
        for arm, m in mu.items():
            n = 0
            if isinstance(counts, dict):
                n = int(counts.get(arm, 0))
            rows.append((arm, float(m), n))

        rows.sort(key=lambda r: r[1], reverse=True)

        print("Arm performance snapshot (sorted by mean R)")
        print("------------------------------------------------")
        print(f"{'arm':20} {'mu_R':>10} {'trades':>10}")
        for arm, m, n in rows:
            print(f"{arm:20} {m:10.4f} {n:10d}")
    else:
        print("Bandit object doesn't expose a 'mu' dict; dumping __dict__ instead:")
        print(bandit.__dict__)


if __name__ == "__main__":
    main()

