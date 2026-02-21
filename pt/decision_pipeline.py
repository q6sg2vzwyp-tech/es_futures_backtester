# pt/decision_pipeline.py
# Extracted from paper_trader.py (decision + AI advisory/guardrails + placement mapping).
# Goal: behavior-preserving refactor step (no runtime logic changes intended).

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


def decide_and_maybe_place_entry(
    *,
    args: Any,
    log: Callable[..., Any],
    learner: Any,
    ai: Any,
    snapshot: Dict[str, Any],
    cand: List[str],
    close: float,
    last_bar_ts: Any,
    net_qty: int,
    place_bracket_fn: Callable[..., Any],
    # mapping inputs
    chosen_arm: Optional[str] = None,
    probs: Optional[Dict[str, float]] = None,
    fast: Optional[float] = None,
    slow: Optional[float] = None,
    c20_max: Optional[float] = None,
    # veto learning hook
    shadow_veto_learn_fn: Optional[Callable[..., Any]] = None,
    strat_path: Optional[str] = None,
    # optional: for logging
    state: Optional[str] = None,
    session_key: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Decide arm (bandit), run AI advisory + guardrails, and (if not shadow) place bracket.

    Returns: (chosen_arm or None, advice dict or None)

    Notes:
    - If chosen_arm is provided, bandit selection is skipped and probs default to {chosen_arm: 1.0} if not supplied.
    - This function intentionally mirrors monolith behavior; keep changes minimal.
    """
    if not cand:
        return None, None

    # 1) Bandit choice as before
    if chosen_arm is None:
        if len(cand) == 1:
            chosen = cand[0]
            probs_local = {chosen: 1.0}
        else:
            chosen, probs_local = learner.choose(cand, sample=(getattr(args, "learn_mode", "") != "shadow"))
            if getattr(args, "learn_mode", "") in ("shadow", "advisory"):
                log(
                    "learn_decision",
                    cand=cand,
                    probs={k: round(v, 3) for k, v in (probs_local or {}).items()},
                    chosen=chosen,
                )
    else:
        chosen = str(chosen_arm)
        probs_local = probs or {chosen: 1.0}

    # Ensure snapshot has bandit_probs updated (matches monolith)
    try:
        snapshot = dict(snapshot)
        snapshot["bandit_probs"] = {k: float(v) for k, v in (probs_local or {}).items()}
        if state is not None:
            snapshot["state"] = state
        if session_key is not None:
            snapshot["session_key"] = session_key
    except Exception:
        pass

    # 3) AI advisory (shadow only – no hard decisions yet)
    advice: Optional[Dict[str, Any]] = None
    try:
        advice = ai.advisory_decision(
            snapshot=snapshot,
            raw_candidate_arms=cand,
            bandit_choice=chosen,
        )
    except Exception as e:
        log("ai_advisory_err", err=str(e))

    # 4) AI guardrails veto (optional)
    allowed = True
    veto_reason = ""
    try:
        allowed, veto_reason = ai.guardrails_check(snapshot, advice)
    except Exception as e:
        log("ai_guard_err", err=str(e))

    if not allowed:
        log("ai_entry_veto", reason=veto_reason, chosen=chosen, cand=cand)
        if shadow_veto_learn_fn is not None:
            try:
                shadow_veto_learn_fn(
                    chosen_arm=str(chosen),
                    reason=f"ai_guard:{veto_reason}",
                    reward=0.0,
                    learner=learner,
                    strat_path=str(strat_path or ""),
                    args=args,
                )
            except Exception:
                pass
        return None, advice

    # Placement mapping (monolith mapping)
    go_long = True
    try:
        if chosen == "trend":
            # with-trend direction
            if fast is not None and slow is not None:
                go_long = bool(fast >= slow)
            else:
                go_long = True
        else:
            # breakout direction
            if c20_max is not None:
                go_long = bool(close >= float(c20_max))
            else:
                go_long = True
    except Exception:
        go_long = True

    # Only place in non-shadow
    if getattr(args, "learn_mode", "") != "shadow":
        place_bracket_fn(go_long, close, last_bar_ts, net_qty, signal_name=chosen)

    return chosen, advice
