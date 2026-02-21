#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tiny_learner.py

Tiny CLI tool to inspect the Thompson bandit state for ES Paper Trader.

Usage:
    python tiny_learner.py
    python tiny_learner.py --model-path path/to/bandit_state.json
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "learn", "bandit_state.json")


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _resolve_arms(bandit: Any) -> List[str]:
    """
    Try to extract a list of arm names from the bandit object.
    """
    for attr in ("arms", "_arms", "action_space"):
        v = _safe_getattr(bandit, attr, None)
        if isinstance(v, (list, tuple)):
            return list(v)
    # Fallback: if it has a 'stats' dict, use its keys
    stats = _safe_getattr(bandit, "stats", None)
    if isinstance(stats, dict):
        try:
            return list(stats.keys())
        except Exception:
            pass
    # Last resort: no arms
    return []


def _extract_per_arm_stats(
    bandit: Any,
    arms: List[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Try to build a per-arm view:
        {
            arm: {
                "pulls": int | None,
                "mean_R": float | None,
                "last_sample": float | None,
            },
            ...
        }
    We support a few likely layouts but never crash if the API is different.
    """
    result: Dict[str, Dict[str, Optional[float]]] = {}
    if not arms:
        return result

    # Common patterns: bandit.stats[arm] = { 'mu': ..., 'n': ... }
    stats_obj = _safe_getattr(bandit, "stats", None)

    # Common patterns: bandit.mu, bandit.n, bandit.last_sample, etc.
    mu_attr = None
    n_attr = None
    last_attr = None

    for name in ("mu", "means", "mean"):
        if hasattr(bandit, name):
            mu_attr = name
            break

    for name in ("n", "counts", "pulls"):
        if hasattr(bandit, name):
            n_attr = name
            break

    for name in ("last_sample", "last_draw", "theta"):
        if hasattr(bandit, name):
            last_attr = name
            break

    mu = _safe_getattr(bandit, mu_attr, None) if mu_attr else None
    n = _safe_getattr(bandit, n_attr, None) if n_attr else None
    last = _safe_getattr(bandit, last_attr, None) if last_attr else None

    def _lookup(container: Any, arm: str, idx: int) -> Optional[float]:
        if container is None:
            return None
        try:
            if isinstance(container, dict):
                return float(container.get(arm)) if arm in container else None
            if isinstance(container, (list, tuple)):
                if 0 <= idx < len(container):
                    return float(container[idx])
        except Exception:
            return None
        return None

    for idx, arm in enumerate(arms):
        pulls = None
        mean_R = None
        last_sample = None

        # First, try stats[arm]
        if isinstance(stats_obj, dict) and arm in stats_obj:
            try:
                s = stats_obj[arm]
                if isinstance(s, dict):
                    if pulls is None:
                        for k in ("n", "count", "pulls"):
                            if k in s:
                                pulls = int(s[k])
                    if mean_R is None:
                        for k in ("mu", "mean", "avg_R"):
                            if k in s:
                                mean_R = float(s[k])
                    if last_sample is None:
                        for k in ("last_sample", "theta", "draw"):
                            if k in s:
                                last_sample = float(s[k])
            except Exception:
                pass

        # Fallback to mu/n/last attributes
        if pulls is None:
            pulls = _lookup(n, arm, idx)
            if pulls is not None:
                pulls = int(pulls)
        if mean_R is None:
            mean_R = _lookup(mu, arm, idx)
        if last_sample is None:
            last_sample = _lookup(last, arm, idx)

        result[arm] = {
            "pulls": pulls,
            "mean_R": mean_R,
            "last_sample": last_sample,
        }

    return result


def _print_table(per_arm: Dict[str, Dict[str, Optional[float]]]) -> None:
    if not per_arm:
        print("No arm stats available.")
        return

    headers = ["Arm", "Pulls", "Mean R", "Last Sample"]
    col_widths = [max(len(h), 8) for h in headers]

    rows: List[List[str]] = []
    for arm, s in per_arm.items():
        pulls = s.get("pulls")
        mean_R = s.get("mean_R")
        last_sample = s.get("last_sample")

        pulls_str = "-" if pulls is None else str(pulls)
        mean_str = "-" if mean_R is None else f"{mean_R:.4f}"
        last_str = "-" if last_sample is None else f"{last_sample:.4f}"

        row = [arm, pulls_str, mean_str, last_str]
        rows.append(row)

        # update column widths
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # print header
    line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(line)
    print(sep)

    # print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Thompson bandit state for ES Paper Trader.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to bandit_state.json (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()

    model_path = args.model_path

    if not os.path.exists(model_path):
        print(f"[tiny_learner] No bandit state found at:\n  {model_path}")
        print("Run the paper trader long enough to close at least one trade, then try again.")
        sys.exit(1)

    try:
        from learner_bandit import load_thompson  # type: ignore
    except Exception as e:
        print(f"[tiny_learner] Failed to import learner_bandit.load_thompson: {e}")
        sys.exit(1)

    bandit = load_thompson(model_path)
    if bandit is None:
        print(f"[tiny_learner] load_thompson returned None for:\n  {model_path}")
        sys.exit(1)

    print(f"[tiny_learner] Loaded bandit from: {model_path}")
    print(f"Type: {type(bandit).__name__}")
    print()

    arms = _resolve_arms(bandit)
    if not arms:
        print("[tiny_learner] Could not detect arms list on bandit.")
        print("Raw bandit __dict__ below for debugging:\n")
        print(repr(getattr(bandit, "__dict__", bandit)))
        sys.exit(0)

    per_arm = _extract_per_arm_stats(bandit, arms)
    _print_table(per_arm)

    # Quick summary
    total_pulls = 0
    best_arm = None
    best_mean = None

    for arm, s in per_arm.items():
        pulls = s.get("pulls") or 0
        mean_R = s.get("mean_R")
        total_pulls += pulls
        if mean_R is not None:
            if best_mean is None or mean_R > best_mean:
                best_mean = mean_R
                best_arm = arm

    print()
    print(f"Total pulls (approx): {total_pulls}")
    if best_arm is not None and best_mean is not None:
        print(f"Best arm by mean R: {best_arm} (mean R ~ {best_mean:.4f})")


if __name__ == "__main__":
    main()

