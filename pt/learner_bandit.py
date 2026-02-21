#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
learner_bandit.py (v2.0)

Gaussian Thompson Sampling bandit for ES Paper Trader.

- Arms are strategy labels, e.g. ["trend_ema", "trend_sma", "breakout_atr"]
- reward is in R units (risk-adjusted PnL per trade)
- We maintain per-arm running mean + count and sample from a Gaussian
  posterior around the mean with variance that shrinks as count grows.

Public API used by paper_trader.py and strategy_core.py:

    from learner_bandit import ThompsonGaussian, save_thompson, load_thompson

    bandit = load_thompson(path)
    if bandit is None:
        bandit = ThompsonGaussian.new(DEFAULT_ARMS, gamma=0.01, sigma2=1.0)

    arm = bandit.choose()
    bandit.update(arm, reward)
    save_thompson(path, bandit)

New (v2.0):
    - Tracks per-arm EMA of R (ema_R) with configurable ema_alpha.
    - Provides summarize_for_heartbeat() to expose bandit_* fields to heartbeat.
    - Provides update_from_trade_row() helper for trades.csv rows.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import logging


@dataclass
class ThompsonGaussian:
    arms: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)    # n per arm
    mean: Dict[str, float] = field(default_factory=dict)    # lifetime mean R
    var: Dict[str, float] = field(default_factory=dict)     # running variance
    ema_R: Dict[str, float] = field(default_factory=dict)   # EMA of R

    # gamma controls how fast variance shrinks as we see more samples
    gamma: float = 0.01
    # sigma2 is the base observation noise variance
    sigma2: float = 1.0
    # ema_alpha controls EMA smoothing for R (0 < ema_alpha <= 1)
    ema_alpha: float = 0.1

    # ---------- construction ----------

    @classmethod
    def new(
        cls,
        arms: List[str],
        gamma: float = 0.01,
        sigma2: float = 1.0,
        ema_alpha: float = 0.1,
    ) -> "ThompsonGaussian":
        """
        Create a new ThompsonGaussian with a fixed set of arms.
        """
        arms = list(dict.fromkeys(arms))  # de-dup but keep order
        counts = {a: 0 for a in arms}
        mean = {a: 0.0 for a in arms}
        var = {a: 1.0 for a in arms}      # start with unit variance
        ema_R = {a: 0.0 for a in arms}
        return cls(
            arms=arms,
            counts=counts,
            mean=mean,
            var=var,
            ema_R=ema_R,
            gamma=gamma,
            sigma2=sigma2,
            ema_alpha=ema_alpha,
        )

    def ensure_arms(self, arms: List[str]) -> None:
        """
        Make sure all provided arms are present in internal state.
        Safe to call on every startup with DEFAULT_ARMS.
        """
        for a in arms:
            if a not in self.arms:
                self.arms.append(a)
            self.counts.setdefault(a, 0)
            self.mean.setdefault(a, 0.0)
            self.var.setdefault(a, 1.0)
            self.ema_R.setdefault(a, 0.0)

    # ---------- core TS logic ----------

    def _posterior_std(self, arm: str) -> float:
        """
        Posterior std-dev for an arm; shrinks as we get more observations.
        Very simple heuristic: sigma / sqrt(gamma + n)
        """
        n = max(0, self.counts.get(arm, 0))
        denom = max(1.0, self.gamma + float(n))
        return math.sqrt(self.sigma2 / denom)

    def sample_arms(self) -> Dict[str, float]:
        """
        Draw one sample from each armâ€™s posterior.
        """
        samples: Dict[str, float] = {}
        for a in self.arms:
            mu = float(self.mean.get(a, 0.0))
            std = max(1e-6, self._posterior_std(a))
            samples[a] = random.gauss(mu, std)
        return samples

    def choose(self) -> str:
        """
        Thompson Sampling choice: sample each arm, pick argmax.
        """
        if not self.arms:
            raise RuntimeError("ThompsonGaussian has no arms configured")
        samples = self.sample_arms()
        # choose arm with highest sample
        best_arm, _ = max(samples.items(), key=lambda kv: kv[1])
        return best_arm

    # ---------- update / observe ----------

    def observe(self, arm: str, reward: float) -> None:
        """
        Core learning update: one new reward R for the given arm.

        - arm: strategy label (e.g. 'trend_ema')
        - reward: R units for that completed trade
        """
        if arm is None:
            return
        arm = str(arm).strip()
        if not arm:
            return

        # Auto-register unknown arms if needed (should normally not happen)
        if arm not in self.arms:
            self.arms.append(arm)
            self.counts.setdefault(arm, 0)
            self.mean.setdefault(arm, 0.0)
            self.var.setdefault(arm, 1.0)
            self.ema_R.setdefault(arm, 0.0)

        n = self.counts.get(arm, 0)
        mu = float(self.mean.get(arm, 0.0))
        v = float(self.var.get(arm, 1.0))
        ema_prev = float(self.ema_R.get(arm, 0.0))

        # Standard running mean update
        n_new = n + 1
        mu_new = mu + (reward - mu) / float(n_new)

        # Simple (population) variance update; not critical but nice to track
        if n == 0:
            v_new = 1.0
        else:
            # Welford-style two-mean update
            v_new = ((n - 1) * v + (reward - mu) * (reward - mu_new)) / max(1, n)

        # EMA of R (more weight on recent trades)
        if n == 0:
            ema_new = reward
        else:
            alpha = float(self.ema_alpha)
            if alpha <= 0.0 or alpha > 1.0:
                alpha = 0.1
            ema_new = alpha * reward + (1.0 - alpha) * ema_prev

        self.counts[arm] = n_new
        self.mean[arm] = mu_new
        self.var[arm] = max(v_new, 1e-6)
        self.ema_R[arm] = ema_new

    def update(self, arm: str, reward: float) -> None:
        """
        Public update entry point used by paper_trader.py.

        Aliases to observe() so older code still works even if it used 'observe'
        and newer code uses 'update'.
        """
        self.observe(arm, reward)

    # ---------- serialization ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arms": self.arms,
            "counts": self.counts,
            "mean": self.mean,
            "var": self.var,
            "ema_R": self.ema_R,
            "gamma": self.gamma,
            "sigma2": self.sigma2,
            "ema_alpha": self.ema_alpha,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThompsonGaussian":
        arms = data.get("arms", []) or []
        counts = data.get("counts", {}) or {}
        mean = data.get("mean", {}) or {}
        var = data.get("var", {}) or {}
        ema_R = data.get("ema_R", {}) or {}
        gamma = float(data.get("gamma", 0.01))
        sigma2 = float(data.get("sigma2", 1.0))
        ema_alpha = float(data.get("ema_alpha", 0.1))

        # Normalize internal state
        arms = list(dict.fromkeys(arms))
        for a in arms:
            counts.setdefault(a, 0)
            mean.setdefault(a, 0.0)
            var.setdefault(a, 1.0)
            ema_R.setdefault(a, 0.0)

        return cls(
            arms=arms,
            counts=counts,
            mean=mean,
            var=var,
            ema_R=ema_R,
            gamma=gamma,
            sigma2=sigma2,
            ema_alpha=ema_alpha,
        )


# ---------- disk helpers ----------


def save_thompson(
    path: str,
    bandit: ThompsonGaussian,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Persist bandit state to JSON."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bandit.to_dict(), f, ensure_ascii=False, indent=2)
    except Exception as e:
        if logger is not None:
            logger.error(f"[bandit] failed to save Thompson state to {path}: {e}")


def load_thompson(
    path: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[ThompsonGaussian]:
    """Load bandit state from JSON; return None if missing or invalid."""
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        bandit = ThompsonGaussian.from_dict(data)
        return bandit
    except Exception as e:
        if logger is not None:
            logger.error(f"[bandit] failed to load Thompson state from {path}: {e}")
        return None


# ---------- extras for heartbeat / trades.csv ----------

def summarize_for_heartbeat(bandit: ThompsonGaussian) -> Dict[str, Any]:
    """
    Build a compact summary for the heartbeat:

    Returns dict like:
        {
            "bandit_best_arm": "trend_ema",
            "bandit_best_mean_R": 0.123,
            "bandit_arms": [
                {"arm": "trend_ema", "n": 10, "mean_R": 0.123, "ema_R": 0.110},
                ...
            ],
        }

    Arms with n == 0 will have mean_R = None and ema_R = None, so the dashboard
    can show "-" for them.
    """
    best_arm = None
    best_mean_R = None

    for arm in bandit.arms:
        n = bandit.counts.get(arm, 0)
        m = bandit.mean.get(arm, 0.0)
        if n > 0:
            if best_mean_R is None or m > best_mean_R:
                best_mean_R = m
                best_arm = arm

    arms_list: List[Dict[str, Any]] = []
    for arm in bandit.arms:
        n = int(bandit.counts.get(arm, 0))
        m = float(bandit.mean.get(arm, 0.0))
        ema = float(bandit.ema_R.get(arm, 0.0))
        if n == 0:
            mean_R_val: Optional[float] = None
            ema_R_val: Optional[float] = None
        else:
            mean_R_val = m
            ema_R_val = ema
        arms_list.append(
            {
                "arm": arm,
                "n": n,
                "mean_R": mean_R_val,
                "ema_R": ema_R_val,
            }
        )

    return {
        "bandit_best_arm": best_arm,
        "bandit_best_mean_R": best_mean_R,
        "bandit_arms": arms_list,
    }


def update_from_trade_row(
    bandit: ThompsonGaussian,
    row: Dict[str, Any],
) -> None:
    """
    Convenience helper: given a trades.csv row, update the bandit

    It looks for:
        - arm name in 'arm' / 'signal' / 'strategy'
        - R value in 'R' / 'r'
    """
    arm = (
        row.get("arm")
        or row.get("signal")
        or row.get("strategy")
        or ""
    )
    arm = str(arm).strip()
    if not arm:
        return

    raw_R = row.get("R") or row.get("r") or None
    if raw_R is None:
        return

    try:
        R_val = float(raw_R)
    except Exception:
        return

    bandit.update(arm, R_val)

