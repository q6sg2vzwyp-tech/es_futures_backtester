from __future__ import annotations

# pt/indicators.py
#
# Pure math / indicator helpers extracted from paper_trader.py.
# Keep side-effect free.

from typing import List
import math


def ema(vals: List[float], span: int) -> float:
    if not vals:
        return float("nan")
    k = 2 / (span + 1)
    s = vals[0]
    for v in vals[1:]:
        s = v * k + s * (1 - k)
    return s


def atr(H: List[float], L: List[float], C: List[float], n: int = 14) -> float:
    if len(C) < n + 1:
        return float("nan")
    trs: List[float] = []
    for i in range(1, len(C)):
        hl = H[i] - L[i]
        hc = abs(H[i] - C[i - 1])
        lc = abs(L[i] - C[i - 1])
        trs.append(max(hl, hc, lc))
    if len(trs) < n:
        return float("nan")
    k = 2 / (n + 1)
    s = trs[-n]
    for v in trs[-n + 1:]:
        s = v * k + s * (1 - k)
    return s
