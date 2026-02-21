# ----------------------------
# Option 2: gated EOD bayes
# ----------------------------
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EODState:
    """
    Tracks whether EOD bayes ran for the current CT date and throttles skip logs.
    """
    last_ran_date: Optional[dt.date] = None
    last_skip_log_ts: float = 0.0
    last_fail_ts: float = 0.0  # backoff after failure


@dataclass
class EODResult:
    """
    Standardized result payload for EOD bayes checks/runs.
    """
    ran: bool
    reason: str = ""
    eligible_trades: int = 0
    best_params: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.ran = bool(self.ran)
        self.reason = str(self.reason or "")
        self.eligible_trades = int(self.eligible_trades or 0)
        if self.best_params is None:
            self.best_params = {}


def _log_info(logger, msg: str, *args) -> None:
    try:
        if logger is not None:
            logger.info(msg, *args)
    except Exception:
        # Never let logging break trading loop
        return


def _log_error(logger, msg: str, *args) -> None:
    try:
        if logger is not None:
            logger.error(msg, *args)
    except Exception:
        return


def count_eligible_trades_for_bayes(trades_csv: str, ignore_reasons: List[str]) -> int:
    """
    Counts rows in trades.csv that are eligible for EOD bayes training.
    - Skips rows with exit_reason/reason in ignore_reasons
    - Skips rows with non-numeric pnl fields (best-effort)
    """
    try:
        if not trades_csv or not os.path.exists(trades_csv):
            return 0

        ignore_set = set([str(x).strip() for x in (ignore_reasons or []) if str(x).strip()])

        n = 0
        with open(trades_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                reason = str(row.get("reason", "") or row.get("exit_reason", "") or "").strip()
                if reason and reason in ignore_set:
                    continue

                pnl = row.get("pnl_usd", None)
                if pnl is None:
                    pnl = row.get("pnl", None)

                if pnl is not None:
                    try:
                        float(pnl)
                    except Exception:
                        continue

                n += 1

        return int(n)
    except Exception:
        return 0


def _run_optimizer_with_compatible_signature(
    *,
    run_eod_bayes_opt_filtered,
    bayes_train_csv: str,
    best_params_path: str,
    param_space: dict,
    logger,
) -> Optional[dict]:
    """
    Calls run_eod_bayes_opt_filtered with a few historical signature variants.
    Returns best_params dict or None.

    This function intentionally tries multiple call signatures to support
    older implementations. TypeError means "wrong signature", so we try
    the next variant.
    """
    # Variant 1: train_csv=
    try:
        return run_eod_bayes_opt_filtered(
            train_csv=bayes_train_csv,
            best_params_path=best_params_path,
            param_space=param_space,
            logger=logger,
        )
    except TypeError:
        # wrong signature; try next
        ...

    # Variant 2: trades_csv=
    try:
        return run_eod_bayes_opt_filtered(
            trades_csv=bayes_train_csv,
            best_params_path=best_params_path,
            param_space=param_space,
            logger=logger,
        )
    except TypeError:
        ...

    # Variant 3: best_params_file=
    try:
        return run_eod_bayes_opt_filtered(
            trades_csv=bayes_train_csv,
            best_params_file=best_params_path,
            param_space=param_space,
            logger=logger,
        )
    except TypeError:
        ...

    # Variant 4: positional fallback
    return run_eod_bayes_opt_filtered(bayes_train_csv, best_params_path, param_space, logger)


def _ensure_state(state: Optional[EODState]) -> EODState:
    if isinstance(state, EODState):
        return state
    return EODState()


def _persist_best_params(best_params: dict, best_params_path: str, logger) -> None:
    if not best_params_path:
        return
    try:
        d = os.path.dirname(best_params_path)
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception:
        # best effort only
        pass

    try:
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as e:
        _log_error(logger, "[eod_core] failed writing best params json: %s", e)


def maybe_run_eod_bayes(
    *,
    now_ct,
    trade_end,
    eod_time,
    state: Optional[EODState],
    trades_csv: str,
    bayes_train_csv: str,
    best_params_path: str,
    param_space: dict,
    ignore_reasons: List[str],
    build_bayes_training_set,
    run_eod_bayes_opt_filtered,
    logger,
    skip_weekends: bool = False,
    fail_backoff_sec: float = 300.0,
) -> EODResult:
    """
    Canonical EOD bayes runner:
    - Runs at most once per CT date (state.last_ran_date)
    - Only runs if now_time >= trade_end and now_time >= eod_time
    - Builds a filtered training set
    - Runs bayes optimization and persists best params
    - Backoff after failures to avoid loop spam
    """
    state = _ensure_state(state)

    # Backoff if we recently failed (prevents log spam)
    now_ts = time.time()
    last_fail_ts = float(getattr(state, "last_fail_ts", 0.0) or 0.0)
    if (now_ts - last_fail_ts) < float(fail_backoff_sec or 300.0):
        return EODResult(ran=False, reason="recent_failure_backoff")

    try:
        if skip_weekends and int(getattr(now_ct, "weekday", lambda: 0)()) >= 5:
            return EODResult(ran=False, reason="weekend_skip")

        now_time = now_ct.time()
        should_check = (now_time >= eod_time) and (now_time >= trade_end)
        if not should_check:
            return EODResult(ran=False, reason="not_eod_window")

        today = now_ct.date()
        if getattr(state, "last_ran_date", None) == today:
            return EODResult(ran=False, reason="already_ran_today")

        eligible_n = count_eligible_trades_for_bayes(trades_csv, ignore_reasons)

        # Build training set (filtered)
        try:
            build_bayes_training_set(
                src_csv=trades_csv,
                dst_csv=bayes_train_csv,
                ignore_reasons=ignore_reasons,
                logger=logger,
            )
        except TypeError:
            # Backward-compatible: older signature may not accept ignore_reasons
            build_bayes_training_set(
                src_csv=trades_csv,
                dst_csv=bayes_train_csv,
                logger=logger,
            )

        # Run optimizer (support multiple historical signatures)
        try:
            best_params = _run_optimizer_with_compatible_signature(
                run_eod_bayes_opt_filtered=run_eod_bayes_opt_filtered,
                bayes_train_csv=bayes_train_csv,
                best_params_path=best_params_path,
                param_space=param_space,
                logger=logger,
            )
        except TypeError as e:
            _log_error(logger, "[eod_core] run_eod_bayes_opt_filtered signature mismatch: %s", e)
            state.last_fail_ts = time.time()
            return EODResult(ran=False, reason="optimizer_signature_mismatch", eligible_trades=eligible_n)

        if not isinstance(best_params, dict):
            state.last_fail_ts = time.time()
            return EODResult(ran=False, reason="optimizer_return_not_dict", eligible_trades=eligible_n)

        # Persist best params (belt + suspenders)
        _persist_best_params(best_params, best_params_path, logger)

        # Mark ran
        state.last_ran_date = today
        state.last_skip_log_ts = 0.0
        state.last_fail_ts = 0.0

        _log_info(
            logger,
            "[eod_core] EOD bayes complete (eligible_trades=%d) best_params_saved=%s",
            eligible_n,
            best_params_path,
        )

        return EODResult(ran=True, reason="ran", eligible_trades=eligible_n, best_params=best_params or {})

    except Exception as e:
        state.last_fail_ts = time.time()
        _log_error(logger, "[eod_core] maybe_run_eod_bayes failed: %s", e)
        return EODResult(ran=False, reason="exception")


def maybe_run_eod_bayes_gated(
    *,
    now_ct,
    trade_end,
    eod_time,
    state: Optional[EODState],
    trades_csv: str,
    bayes_train_csv: str,
    best_params_path: str,
    param_space: dict,
    ignore_reasons: List[str],
    min_trades: int,
    build_bayes_training_set,
    run_eod_bayes_opt_filtered,
    logger,
    skip_log_throttle_sec: float = 60.0,
    skip_weekends: bool = False,
    fail_backoff_sec: float = 300.0,
) -> EODResult:
    """
    Wrapper around maybe_run_eod_bayes() that gates execution on:
    - now_time >= eod_time and now_time >= trade_end
    - eligible_trades >= min_trades
    Also throttles the skip log to avoid spamming every loop tick.
    """
    state = _ensure_state(state)

    try:
        now_time = now_ct.time()
        should_check = (now_time >= eod_time) and (now_time >= trade_end)

        # If we aren't in the post-EOD window, delegate (it will no-op).
        if not should_check:
            return maybe_run_eod_bayes(
                now_ct=now_ct,
                trade_end=trade_end,
                eod_time=eod_time,
                state=state,
                trades_csv=trades_csv,
                bayes_train_csv=bayes_train_csv,
                best_params_path=best_params_path,
                param_space=param_space,
                ignore_reasons=ignore_reasons,
                build_bayes_training_set=build_bayes_training_set,
                run_eod_bayes_opt_filtered=run_eod_bayes_opt_filtered,
                logger=logger,
                skip_weekends=skip_weekends,
                fail_backoff_sec=fail_backoff_sec,
            )

        eligible_n = count_eligible_trades_for_bayes(trades_csv, ignore_reasons)

        if eligible_n < int(min_trades or 0):
            now_ts = time.time()
            last_ts = float(getattr(state, "last_skip_log_ts", 0.0) or 0.0)
            if (now_ts - last_ts) >= float(skip_log_throttle_sec or 60.0):
                _log_info(
                    logger,
                    "[eod_core] skipping EOD bayes: eligible_trades=%d < eod_min_trades=%d",
                    eligible_n,
                    int(min_trades or 0),
                )
                state.last_skip_log_ts = now_ts

            return EODResult(ran=False, reason="min_trades_gate", eligible_trades=eligible_n)

        # Gate passed; run the normal EOD bayes
        return maybe_run_eod_bayes(
            now_ct=now_ct,
            trade_end=trade_end,
            eod_time=eod_time,
            state=state,
            trades_csv=trades_csv,
            bayes_train_csv=bayes_train_csv,
            best_params_path=best_params_path,
            param_space=param_space,
            ignore_reasons=ignore_reasons,
            build_bayes_training_set=build_bayes_training_set,
            run_eod_bayes_opt_filtered=run_eod_bayes_opt_filtered,
            logger=logger,
            skip_weekends=skip_weekends,
            fail_backoff_sec=fail_backoff_sec,
        )

    except Exception as e:
        try:
            state.last_fail_ts = time.time()
        except Exception:
            pass
        _log_error(logger, "[eod_core] maybe_run_eod_bayes_gated failed: %s", e)
        return EODResult(ran=False, reason="exception")
