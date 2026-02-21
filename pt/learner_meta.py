#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math
import utils


@dataclass
class MetaLearner:
    """Very simple meta-learner that tracks smoothed R and suggests an aggressiveness factor."""
    ema_R: float = 0.0
    alpha: float = 0.02  # smoothing factor

    def update(self, trade_R: float) -> None:
        self.ema_R = (1.0 - self.alpha) * self.ema_R + self.alpha * trade_R

    def aggressiveness_factor(self) -> float:
        """
        Map EMA of R into an aggressiveness multiplier in [0.5, 1.5].
        Positive EMA -> >1; negative -> <1.
        """
        x = max(-5.0, min(5.0, self.ema_R))
        return 1.0 + 0.1 * math.tanh(x)

