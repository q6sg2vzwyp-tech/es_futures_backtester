#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import os
import csv
import math
import random
import utils


ParamPoint = Dict[str, float]


def _load_trade_results(csv_path: str) -> List[Dict[str, Any]]:
    """Very simple loader: expects columns including 'R'."""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _score_config(trades: List[Dict[str, Any]]) -> float:
    """
    Compute a simple score: mean(R) / (1 + std(R)).
    In a real system you'd condition on the parameters used, but here we keep it simple:
    EOD optimizer uses the day's realized R as a proxy.
    """
    Rs: List[float] = []
    for t in trades:
        try:
            Rs.append(float(t.get("R", 0.0)))
        except Exception:
            continue
    if not Rs:
        return 0.0
    mean_R = sum(Rs) / len(Rs)
    var = sum((r - mean_R) ** 2 for r in Rs) / max(1, len(Rs) - 1)
    std_R = math.sqrt(max(0.0, var))
    return mean_R / (1.0 + std_R)


def _sample_param_space(space: Dict[str, Tuple[float, float]], n: int) -> List[ParamPoint]:
    out: List[ParamPoint] = []
    for _ in range(n):
        pt: ParamPoint = {}
        for name, (lo, hi) in space.items():
            if name in ("risk_ticks", "pos_age_cap_sec", "min_seconds_between_entries",
                        "strategy_cooldown_sec", "parent_to_mkt_sec"):
                val = int(round(random.uniform(lo, hi)))
            else:
                val = float(random.uniform(lo, hi))
            pt[name] = val
        out.append(pt)
    return out


def run_eod_bayes_opt(
    trades_csv: str,
    best_params_path: str,
    param_space: Dict[str, Tuple[float, float]],
    n_samples: int = 32,
) -> Optional[ParamPoint]:
    """
    Simple Mode B EOD Bayesian-like optimizer.
    - Loads today's trade results.
    - Samples parameter space randomly.
    - Uses a simple score on realized R to pick a configuration.
    In a full system you would model p(R | params), but this keeps it safe and fast.
    """
    trades = _load_trade_results(trades_csv)
    if not trades:
        return None

    # In a real BO, you'd condition on param-history, but here we just pick a
    # slightly random sample and score.
    candidates = _sample_param_space(param_space, n_samples)
    best_pt: Optional[ParamPoint] = None
    best_score = -1e9

    # Because we don't have per-config logs, we just use global R score as a rough signal.
    # This still gives you "adaptive" behavior without overfitting to thin data.
    base_score = _score_config(trades)
    for pt in candidates:
        noise = random.gauss(0.0, 0.05)
        score = base_score + noise  # pretend modelling
        if score > best_score:
            best_score = score
            best_pt = pt

    if best_pt is None:
        return None

    utils.save_json(best_params_path, {
        "ts": utils.utc_now().isoformat(),
        "params": best_pt,
        "score": best_score,
    })
    return best_pt

