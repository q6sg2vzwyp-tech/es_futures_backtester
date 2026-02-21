#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bandit_core.py (v2.0)

Thompson-sampling bandit for ES Paper Trader.

Goals
-----
- Manage a fixed set of strategy "arms" (e.g. trend_ema, breakout_atr, etc).
- For each arm, track:
    * n         : number of trades with a valid R
    * mean_R    : simple average of R
    * ema_R     : exponential moving average of R (smooth / recent performance)
    * alpha,beta: Beta-Bernoulli parameters for Thompson sampling
- Provide:
    * select_arm()    : pick an arm to trade (Thompson sampling)
    * update(arm, R)  : update stats after a trade with per-trade R
    * summarize()     : dict suitable for exposing in heartbeat
    * save/load JSON  : for persistence across restarts

How to use (high level)
-----------------------
1) In paper_trader startup:

    from bandit_core import Bandit, DEFAULT_ARM_NAMES

    bandit = Bandit(arms=DEFAULT_ARM_NAMES)

2) When you need a signal:

    chosen_arm = bandit.select_arm()
    # pass chosen_arm into strategy_core / signal selection

3) When a trade closes and you know its per-trade R and arm name:

    bandit.update(arm_name, R)

4) To push into heartbeat:

    bandit_summary = bandit.summarize_for_heartbeat()
    heartbeat.update(bandit_summary)

5) Periodically save:

    bandit.save_json("run/bandit_state.json")

And on restart:

    bandit = Bandit.load_json("run/bandit_state.json",
                              arms=DEFAULT_ARM_NAMES)
"""

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable


# ---------------------------------------------------------------------------
# Default arm set (adjust names to match your strategy_core)
# ---------------------------------------------------------------------------

DEFAULT_ARM_NAMES: List[str] = [
    "trend_ema",
    "trend_sma",
    "breakout_atr",
    "pullback_vwap",
    "momentum_rsi",
    "range_fade",
    "trend_pullback",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_float(x, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Per-arm state
# ---------------------------------------------------------------------------

@dataclass
class ArmState:
    name: str
    n: int = 0                  # number of *valid R* observations
    mean_R: float = 0.0         # simple average over lifetime
    ema_R: float = 0.0          # exponential moving average of R
    alpha: float = 2.0          # Beta prior alpha (for Thompson sampling)
    beta: float = 2.0           # Beta prior beta
    # You can add fields like "last_ts" later if needed.

    def update(self, R: float, ema_alpha: float) -> None:
        """
        Update this arm with a new realized per-trade R.

        ema_alpha: smoothing factor in (0, 1]; higher = more weight on recent R.
        """
        if math.isnan(R):
            return

        # Update count
        self.n += 1

        # Update simple mean_R (online)
        if self.n == 1:
            self.mean_R = R
            self.ema_R = R
        else:
            self.mean_R += (R - self.mean_R) / float(self.n)
            # EMA
            self.ema_R = ema_alpha * R + (1.0 - ema_alpha) * self.ema_R

        # Update Beta parameters for Thompson sampling.
        # Here we treat "R > 0" as a "success", "R <= 0" as "failure".
        if R > 0.0:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def as_dict(self) -> Dict:
        d = asdict(self)
        # Ensure numeric fields are basic types (for JSON)
        d["n"] = int(self.n)
        d["mean_R"] = float(self.mean_R)
        d["ema_R"] = float(self.ema_R)
        d["alpha"] = float(self.alpha)
        d["beta"] = float(self.beta)
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "ArmState":
        return cls(
            name=data.get("name", ""),
            n=int(data.get("n", 0)),
            mean_R=float(data.get("mean_R", 0.0)),
            ema_R=float(data.get("ema_R", 0.0)),
            alpha=float(data.get("alpha", 2.0)),
            beta=float(data.get("beta", 2.0)),
        )


# ---------------------------------------------------------------------------
# Bandit
# ---------------------------------------------------------------------------

class Bandit:
    """
    Thompson-sampling bandit over a fixed set of arms.

    Typical workflow:
    - Initialize with a list of arm names.
    - Call select_arm() to choose which arm to trade.
    - After each trade, call update(arm_name, R) with the realized R.
    - Periodically (or at shutdown), save_json().
    - On startup, load_json() to restore state.
    """

    def __init__(
        self,
        arms: Iterable[str],
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        ema_alpha: float = 0.1,
        rng: Optional[random.Random] = None,
    ):
        self.ema_alpha = float(ema_alpha)
        self._rng = rng or random.Random()
        self._arms: Dict[str, ArmState] = {}

        for name in arms:
            name_str = str(name).strip()
            if not name_str:
                continue
            self._arms[name_str] = ArmState(
                name=name_str,
                n=0,
                mean_R=0.0,
                ema_R=0.0,
                alpha=prior_alpha,
                beta=prior_beta,
            )

    # ------------------------------
    # Core operations
    # ------------------------------

    @property
    def arm_names(self) -> List[str]:
        return list(self._arms.keys())

    def get_arm_state(self, name: str) -> Optional[ArmState]:
        return self._arms.get(name)

    def select_arm(self) -> Optional[str]:
        """
        Thompson sampling: for each arm, draw a sample from Beta(alpha, beta),
        pick the arm with the largest sample.

        Returns None if there are no arms.
        """
        if not self._arms:
            return None

        best_name = None
        best_sample = None

        for name, state in self._arms.items():
            # Draw from Beta(alpha, beta)
            a = max(state.alpha, 1e-6)
            b = max(state.beta, 1e-6)
            sample = self._rng.betavariate(a, b)
            if (best_sample is None) or (sample > best_sample):
                best_sample = sample
                best_name = name

        return best_name

    def update(self, arm_name: str, R: float) -> None:
        """
        Update the given arm with a realized per-trade R.
        If arm_name is unknown or R is None/NaN, this is a no-op.
        """
        if arm_name is None:
            return
        name = str(arm_name).strip()
        if not name:
            return

        state = self._arms.get(name)
        if state is None:
            # Optionally auto-create unknown arms. For now we do NOT,
            # but you could enable this if strategy_core may introduce new names.
            return

        R_val = _safe_float(R, None)
        if R_val is None:
            return

        state.update(R_val, self.ema_alpha)

    # ------------------------------
    # Bulk updates from trades.csv rows (optional helper)
    # ------------------------------

    def update_from_trade_row(self, row: Dict) -> None:
        """
        Given a trades.csv row (dict from csv.DictReader), update the bandit
        using the 'arm' (or 'signal'/'strategy') and 'R' columns.
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

        R_val = (
            row.get("R")
            or row.get("r")
            or None
        )
        self.update(arm, _safe_float(R_val, None))

    # ------------------------------
    # Summaries & heartbeat integration
    # ------------------------------

    def summarize(self) -> Dict[str, Dict]:
        """
        Returns a dict mapping arm_name -> ArmState.as_dict().

        Example:
        {
            "trend_ema": { "name": "trend_ema", "n": 10, "mean_R": 0.12, ... },
            ...
        }
        """
        return {name: st.as_dict() for name, st in self._arms.items()}

    def summarize_for_heartbeat(self) -> Dict[str, object]:
        """
        Returns a compact dict suitable to merge into the heartbeat, e.g.:

        {
            "bandit_best_arm": "trend_ema",
            "bandit_best_mean_R": 0.123,
            "bandit_arms": [
                {"arm": "trend_ema", "n": 10, "mean_R": 0.123, "ema_R": 0.100},
                ...
            ],
        }

        Arms with n == 0 will have mean_R = None and appear as "-" in the dashboard.
        """
        best_arm_name = None
        best_mean_R = None

        for name, st in self._arms.items():
            # If we have at least one R, consider it
            if st.n > 0:
                if (best_mean_R is None) or (st.mean_R > best_mean_R):
                    best_mean_R = st.mean_R
                    best_arm_name = name

        arms_list = []
        for name, st in self._arms.items():
            arms_list.append(
                {
                    "arm": st.name,
                    "n": st.n,
                    "mean_R": st.mean_R if st.n > 0 else None,
                    "ema_R": st.ema_R if st.n > 0 else None,
                    "alpha": st.alpha,
                    "beta": st.beta,
                }
            )

        summary = {
            "bandit_best_arm": best_arm_name,
            "bandit_best_mean_R": best_mean_R,
            "bandit_arms": arms_list,
        }
        return summary

    # ------------------------------
    # JSON persistence
    # ------------------------------

    def to_dict(self) -> Dict[str, object]:
        """
        Serialize full bandit state to a dict suitable for JSON.
        """
        return {
            "ema_alpha": self.ema_alpha,
            "arms": [st.as_dict() for st in self._arms.values()],
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, object],
        default_arms: Optional[Iterable[str]] = None,
    ) -> "Bandit":
        """
        Build Bandit from a previously-saved dict. Any arms that are missing
        in the saved data but present in default_arms will be created fresh.
        """
        ema_alpha = _safe_float(data.get("ema_alpha"), 0.1) or 0.1
        arms_data = data.get("arms", []) or []

        # First create an empty bandit with no arms; we'll populate manually.
        bandit = cls(arms=[], ema_alpha=ema_alpha)

        # Load existing arm states
        for arm_dict in arms_data:
            st = ArmState.from_dict(arm_dict)
            if st.name:
                bandit._arms[st.name] = st

        # Ensure default arms exist
        if default_arms:
            for name in default_arms:
                name_str = str(name).strip()
                if not name_str:
                    continue
                if name_str not in bandit._arms:
                    bandit._arms[name_str] = ArmState(name=name_str)

        return bandit

    def save_json(self, path: str) -> None:
        """
        Save bandit state as JSON to the given file path.
        """
        tmp_path = path + ".tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)

    @classmethod
    def load_json(
        cls,
        path: str,
        arms: Optional[Iterable[str]] = None,
        default_ema_alpha: float = 0.1,
    ) -> "Bandit":
        """
        Load bandit state from JSON. If the file does not exist or is invalid,
        returns a fresh Bandit(arms=arms).
        """
        if not path or not os.path.exists(path):
            return cls(arms=arms or DEFAULT_ARM_NAMES, ema_alpha=default_ema_alpha)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return cls(arms=arms or DEFAULT_ARM_NAMES, ema_alpha=default_ema_alpha)

        return cls.from_dict(data, default_arms=arms or DEFAULT_ARM_NAMES)

