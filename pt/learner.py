from __future__ import annotations

import os
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Callable


# logger injection (so pt/ does not depend on paper_trader.py)
_LOG: Optional[Callable[..., None]] = None


def set_log(fn: Callable[..., None]):
    global _LOG
    _LOG = fn


def _log(tag: str, **fields):
    if _LOG:
        try:
            _LOG(tag, **fields)
            return
        except Exception:
            pass
    # fallback
    try:
        parts = [f"{k}={v}" for k, v in fields.items()]
        msg = f"[{tag}] " + " ".join(parts) if parts else f"[{tag}]"
        print(msg, flush=True)
    except Exception:
        pass


def mkdirs(p):
    if p:
        os.makedirs(p, exist_ok=True)


def learner_paths(base_dir: str, stem: str) -> str:
    mkdirs(base_dir)
    return os.path.join(base_dir, f"{stem}.json")


class ThompsonGaussian:
    def __init__(self, arms: List[str], decay_gamma: float, prior_mean=0.0, prior_var=0.25):
        self.arms = arms[:]
        self.gamma = float(decay_gamma)
        self.m = {a: float(prior_mean) for a in arms}
        self.s2 = {a: float(prior_var) for a in arms}
        self.w = {a: 1e-6 for a in arms}
        self.last_arm: Optional[str] = None

    def choose(self, cand_arms: List[str], sample: bool) -> Tuple[str, Dict[str, float]]:
        scores: Dict[str, float] = {}
        for a in cand_arms:
            std = math.sqrt(max(1e-6, self.s2[a] / (self.w[a] + 1.0)))
            scores[a] = random.gauss(self.m[a], std)
        m = max(scores.values()) if scores else 0.0
        exps = {a: math.exp(scores[a] - m) for a in cand_arms}
        s = sum(exps.values()) or 1.0
        probs = {a: exps[a] / s for a in cand_arms}
        choice = max(probs.items(), key=lambda kv: kv[1])[0]
        if sample:
            r, cum = random.random(), 0.0
            for a in cand_arms:
                cum += probs[a]
                if r <= cum:
                    choice = a
                    break
        return choice, probs

    def update(self, arm: str, reward_R: float):
        g = self.gamma
        w_old = self.w[arm]
        self.w[arm] = g * w_old + 1.0
        m_old = self.m[arm]
        m_new = m_old + (reward_R - m_old) / self.w[arm]
        s2_old = self.s2[arm]
        s2_new = g * s2_old + (reward_R - m_old) * (reward_R - m_new)
        self.m[arm] = m_new
        self.s2[arm] = max(1e-6, s2_new)
        self.last_arm = arm

    # ---- persistence ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "arms": self.arms,
            "gamma": self.gamma,
            "m": self.m,
            "s2": self.s2,
            "w": self.w,
            "last_arm": self.last_arm,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ThompsonGaussian":
        obj = ThompsonGaussian(list(d["arms"]), float(d["gamma"]))
        obj.m = {k: float(v) for k, v in d["m"].items()}
        obj.s2 = {k: float(v) for k, v in d["s2"].items()}
        obj.w = {k: float(v) for k, v in d["w"].items()}
        obj.last_arm = d.get("last_arm")
        return obj


def load_thompson(path: str) -> Optional[ThompsonGaussian]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return ThompsonGaussian.from_dict(d)
    except Exception as e:
        _log("learner_load_err", path=path, err=str(e))
    return None


def save_thompson(path: str, learner: ThompsonGaussian):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(learner.to_dict(), f)
        os.replace(tmp, path)
        _log("learner_saved", path=path)
    except Exception as e:
        _log("learner_save_err", path=path, err=str(e))
