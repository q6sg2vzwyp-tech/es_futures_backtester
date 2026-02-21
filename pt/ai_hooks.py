from __future__ import annotations

import os
import json
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple, Callable


def _default_utc_now_str() -> str:
    # Fallback if caller doesn't inject the project's utc_now_str()
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _default_log(tag: str, **fields):
    # Fallback if caller doesn't inject the project's log()
    try:
        parts = [f"{k}={v}" for k, v in fields.items()]
        msg = f"[{tag}] " + " ".join(parts) if parts else f"[{tag}]"
        print(msg, flush=True)
    except Exception:
        pass


class AIHooks:
    """
    Central place for AI-related behavior:
    - Journaling (post-trade analysis payloads)
    - Advisory (shadow decisions/do-not-trade suggestions)
    - Guardrails (extra veto reasons)
    For now everything is file-based and side-effect-only; no network calls.
    """

    def __init__(
        self,
        base_dir: str = r".\logs\ai",
        utc_now_str: Optional[Callable[[], str]] = None,
        logger: Optional[Callable[..., None]] = None,
    ):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.journal_path = os.path.join(self.base_dir, "trade_journal.jsonl")
        self.advice_path = os.path.join(self.base_dir, "advice_decisions.jsonl")
        self.guard_path = os.path.join(self.base_dir, "guardrails.jsonl")
        self._utc_now_str = utc_now_str or _default_utc_now_str
        self._log = logger or _default_log

    # --- utilities ---
    def _append_jsonl(self, path: str, obj: dict):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception as e:
            self._log("ai_log_err", path=path, err=str(e))

    # ---------- 1) Post-trade journal ----------
    def log_flat_cycle(
        self,
        context: Dict[str, Any],
        pnl_dollars: float,
        reward_R: float,
        param_arm: Optional[str],
        strat_arm: Optional[str],
    ):
        payload = {
            "ts": self._utc_now_str(),
            "evt": "ai_journal_flat_cycle",
            "pnl": float(round(pnl_dollars, 2)),
            "R": float(round(reward_R, 4)),
            "param_arm": param_arm,
            "strat_arm": strat_arm,
            "ctx": context,
        }
        self._append_jsonl(self.journal_path, payload)

    # ---------- 2) Advisory (shadow) ----------
    def advisory_decision(
        self,
        snapshot: Dict[str, Any],
        raw_candidate_arms: List[str],
        bandit_choice: Optional[str],
    ) -> Dict[str, Any]:
        """
        For now this is a dumb 'AI' – just echoes the bandit choice and logs it.
        Later you can replace this with a real model call.
        Returns an advice dict that can be used by guardrails or metrics.
        """
        advice = {
            "ts": self._utc_now_str(),
            "evt": "ai_advice",
            "cand_arms": raw_candidate_arms,
            "bandit_choice": bandit_choice,
            "recommended": bandit_choice,  # placeholder for real AI override
            "reason": "placeholder_advisor",
            "snapshot": snapshot,
        }
        self._append_jsonl(self.advice_path, advice)
        return advice

    # ---------- 3) Guardrails / veto ----------
    def guardrails_check(
        self,
        snapshot: Dict[str, Any],
        advice: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Returns (allowed, reason_if_blocked).
        Simple rule-based veto layer on top of bandit decisions.
        """

        risk = snapshot.get("risk", {}) or {}
        try:
            day_R = float(risk.get("day_R", 0.0))
        except Exception:
            day_R = 0.0
        trades_today = int(risk.get("trades_today", 0))
        consec_losses = int(risk.get("consec_losses", 0))

        # Some context variables if present
        atrp = snapshot.get("atrp", None)
        is_trend = bool(snapshot.get("is_trend", False))
        is_breakout = bool(snapshot.get("is_breakout", False))
        state = snapshot.get("state", "")

        def _block(reason: str) -> Tuple[bool, str]:
            payload = {
                "ts": self._utc_now_str(),
                "evt": "ai_guard_block",
                "reason": reason,
                "snapshot": snapshot,
            }
            if advice:
                payload["advice"] = advice
            self._append_jsonl(self.guard_path, payload)
            return False, reason

        # 1) Soft guardrail: if down more than 4R, pause trading for 15 minutes
        if day_R <= -25.0:
            now = dt.datetime.utcnow()
            pause_until = getattr(self, "_ai_pause_until", None)

            # If there is no pause yet, or the previous pause has expired, start a new one
            if (pause_until is None) or (now >= pause_until):
                pause_until = now + dt.timedelta(seconds=900)  # 900s = 15 minutes
                self._ai_pause_until = pause_until

            # While we're inside the pause window, block entries
            if now < pause_until:
                return _block(f"ai_guard:dayR_pause_until:{pause_until.isoformat()}")

        # If a pause window is active from earlier and hasn't expired yet,
        # keep blocking even if day_R has recovered a bit
        pause_until = getattr(self, "_ai_pause_until", None)
        if pause_until is not None and dt.datetime.utcnow() < pause_until:
            return _block(f"ai_guard:dayR_pause_until:{pause_until.isoformat()}")

        # 2) Soft stop: after 5+ consec losses and still red on the day, stand down
        if consec_losses >= 5 and day_R < 0.0:
            return _block("ai_guard:consec_losses_ge5_and_red_day")

        # 3) Very low volatility: block breakout entries if ATRP too small
        if is_breakout and (atrp is not None):
            try:
                atrp_val = float(atrp)
            except Exception:
                atrp_val = None
            if atrp_val is not None and atrp_val < 0.00003:
                return _block("ai_guard:breakout_in_too_low_atrp")

        # 4) If the top-level state already says caps/wait_rt/sleep, do nothing here
        #    (the outer rails will block trading anyway)
        if state in ("caps", "wait_rt", "sleep"):
            return True, ""

        # Otherwise, allow
        return True, ""
