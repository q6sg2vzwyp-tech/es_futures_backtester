#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pt_loop_promotion.py

Extracted from loop_core.py (v10): end-of-day promotion logic (shadow → real allowlist).

Behavior: identical to prior implementation (loop_core's helper).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    if not path or (not os.path.exists(path)):
        return []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _arm_meanR_from_rows(
    rows: List[Dict[str, Any]],
    arm_key_candidates: List[str],
    r_key_candidates: List[str],
) -> Dict[str, Tuple[int, float]]:
    per: Dict[str, List[float]] = {}
    for row in rows:
        arm = ""
        for k in arm_key_candidates:
            v = row.get(k)
            if v:
                arm = str(v).strip()
                break
        if not arm:
            continue

        r_val = None
        for rk in r_key_candidates:
            v = row.get(rk)
            if v is None:
                continue
            s = str(v).strip()
            if s in ("", "NA", "na", "None", "null", "-"):
                continue
            try:
                r_val = float(s)
            except Exception:
                r_val = None
            if r_val is not None:
                break

        if r_val is None:
            continue

        per.setdefault(arm, []).append(float(r_val))

    out: Dict[str, Tuple[int, float]] = {}
    for arm, rs in per.items():
        if rs:
            out[arm] = (len(rs), sum(rs) / float(len(rs)))
    return out


def maybe_promote_shadow_to_real(ctx: Dict[str, Any], now_ct: dt.datetime, logger) -> Optional[str]:
    args = ctx["args"]
    eod_state = ctx.get("eod_state", None)
    if eod_state is None:
        return None

    if not bool(getattr(args, "promote_shadow_to_real", False)):
        return None

    today = now_ct.date()
    if getattr(eod_state, "promoted_date", None) == today:
        return None

    shadow_csv = ctx.get("SHADOW_ROUNDTRIP_LOG", "")
    real_trades_csv = ctx.get("TRADE_LOG_CSV", "")
    real_arms_json = ctx.get("REAL_ARMS_JSON", "")

    min_shadow = int(getattr(args, "promote_min_shadow_trades", 30) or 30)
    min_real = int(getattr(args, "promote_min_real_trades", 10) or 10)
    thresh = float(getattr(args, "promote_meanR_threshold", 0.10) or 0.10)

    shadow_rows = _read_csv_rows(shadow_csv)
    real_rows = _read_csv_rows(real_trades_csv)

    shadow_stats = _arm_meanR_from_rows(
        shadow_rows,
        arm_key_candidates=["arm", "strategy", "signal"],
        r_key_candidates=["R", "r"],
    )
    real_stats = _arm_meanR_from_rows(
        real_rows,
        arm_key_candidates=["arm", "strategy", "signal"],
        r_key_candidates=["R", "r"],
    )

    real_arms = ctx.get("real_arms", None)
    if not isinstance(real_arms, list) or not real_arms:
        return None
    real_arms = [str(a).strip() for a in real_arms if str(a).strip()]
    if not real_arms:
        return None

    best_shadow_arm = None
    best_shadow_mean = None
    best_shadow_n = 0
    for arm, (n, meanR) in shadow_stats.items():
        if n < min_shadow:
            continue
        if best_shadow_mean is None or meanR > best_shadow_mean:
            best_shadow_arm = arm
            best_shadow_mean = float(meanR)
            best_shadow_n = int(n)

    if best_shadow_arm is None or best_shadow_mean is None:
        logger.info("[promote] no eligible shadow arm yet (need >=%d shadow trades per arm)", min_shadow)
        return None

    worst_real_arm = None
    worst_real_score = None
    worst_real_n = 0
    for arm in real_arms:
        n, meanR = real_stats.get(arm, (0, 0.0))
        score = -999.0 if n < min_real else float(meanR)
        if worst_real_score is None or score < worst_real_score:
            worst_real_arm = arm
            worst_real_score = score
            worst_real_n = int(n)

    if worst_real_arm is None or worst_real_score is None:
        return None

    if best_shadow_arm in real_arms:
        logger.info("[promote] best shadow arm already in real allowlist: %s", best_shadow_arm)
        try:
            eod_state.promoted_date = today
        except Exception:
            pass
        return None

    if best_shadow_mean < (float(worst_real_score) + thresh):
        logger.info(
            "[promote] no promote: best_shadow=%s (n=%d meanR=%.3f) vs worst_real=%s (n=%d score=%.3f) + thresh=%.3f",
            best_shadow_arm,
            best_shadow_n,
            best_shadow_mean,
            worst_real_arm,
            worst_real_n,
            float(worst_real_score),
            thresh,
        )
        try:
            eod_state.promoted_date = today
        except Exception:
            pass
        return None

    new_real_arms = [a for a in real_arms if a != worst_real_arm]
    new_real_arms.append(best_shadow_arm)

    try:
        if real_arms_json:
            payload = {
                "updated": now_ct.isoformat(timespec="seconds"),
                "real_arms": new_real_arms,
                "promoted_in": {
                    "added": best_shadow_arm,
                    "removed": worst_real_arm,
                    "best_shadow_n": best_shadow_n,
                    "best_shadow_meanR": best_shadow_mean,
                    "worst_real_n": worst_real_n,
                    "worst_real_score": float(worst_real_score),
                    "threshold": thresh,
                },
            }
            _atomic_write_json(real_arms_json, payload)

        ctx["real_arms"] = new_real_arms
        logger.warning(
            "[promote] UPDATED real_arms: removed=%s added=%s -> %s",
            worst_real_arm,
            best_shadow_arm,
            new_real_arms,
        )
    except Exception as e:
        logger.error("[promote] failed to write real arms json: %s", e)
        return None

    try:
        eod_state.promoted_date = today
    except Exception:
        pass

    return f"promoted:{best_shadow_arm}:replaced:{worst_real_arm}"

# Backward-compatible alias (older code may reference the private name)
_maybe_promote_shadow_to_real = maybe_promote_shadow_to_real
