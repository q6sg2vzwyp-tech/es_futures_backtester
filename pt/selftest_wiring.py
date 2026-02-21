#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
selftest_wiring.py

Hard check that paper_trader bootstrap + loop_core are correctly wired to
required modules and required callables exist.

This does NOT connect to IB.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from typing import Any, Dict


def require_attr(mod, name: str):
    if not hasattr(mod, name):
        raise AssertionError(f"{mod.__name__} missing required attribute: {name}")
    return getattr(mod, name)


def require_callable(mod, name: str):
    obj = require_attr(mod, name)
    if not callable(obj):
        raise AssertionError(f"{mod.__name__}.{name} exists but is not callable")
    return obj


def show_sig(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "(signature unavailable)"


def main() -> None:
    print("=== Importing core modules ===")

    paper_trader = importlib.import_module("paper_trader")
    loop_core = importlib.import_module("loop_core")

    # These are imported/used by paper_trader bootstrap
    ib_core = importlib.import_module("ib_core")
    hb_core = importlib.import_module("hb_core")
    bayes_core = importlib.import_module("bayes_core")
    startup_core = importlib.import_module("startup_core")
    margin_core = importlib.import_module("margin_core")
    pt_utils = importlib.import_module("pt_utils")
    state_core = importlib.import_module("state_core")
    day_policy_core = importlib.import_module("day_policy_core")
    learner_bandit = importlib.import_module("learner_bandit")
    learner_meta = importlib.import_module("learner_meta")
    risk_core = importlib.import_module("risk_core")
    strategy_core = importlib.import_module("strategy_core")

    # Required symbols (bootstrap)
    print("=== Checking required callables (bootstrap) ===")
    require_callable(ib_core, "connect_ib")
    require_callable(ib_core, "resolve_contract")

    require_callable(hb_core, "build_and_write_heartbeat")

    require_callable(bayes_core, "maybe_apply_bayes_best")
    require_callable(bayes_core, "build_bayes_training_set")
    require_callable(bayes_core, "run_eod_bayes_opt_filtered")

    require_callable(startup_core, "maybe_daily_restart")
    require_callable(startup_core, "has_hedge_protection")
    require_callable(startup_core, "attach_startup_protection")

    require_callable(margin_core, "MarginManager")

    require_callable(pt_utils, "build_bandit_hb_fields")
    require_callable(pt_utils, "recompute_intraday_from_trades")

    require_callable(state_core, "load_runtime_state")

    require_callable(day_policy_core, "DayPolicyState")

    require_callable(learner_bandit, "load_thompson")
    require_callable(learner_bandit, "save_thompson")

    require_attr(strategy_core, "BarBuffer")
    require_attr(strategy_core, "DEFAULT_ARMS")

    require_callable(risk_core, "default_week_state")
    require_callable(risk_core, "roll_week_if_needed")

    # loop_core entrypoint + its dependencies
    print("=== Checking loop_core entrypoint + required imports ===")
    run_iter = require_callable(loop_core, "run_loop_iteration")
    print(f"loop_core.run_loop_iteration sig: {show_sig(run_iter)}")

    # Hard checks for loop_core imports used in your version
    gate_core = importlib.import_module("gate_core")
    session_core = importlib.import_module("session_core")
    equity_core = importlib.import_module("equity_core")
    pnl_core = importlib.import_module("pnl_core")
    position_core = importlib.import_module("position_core")
    eod_core = importlib.import_module("eod_core")

    require_callable(gate_core, "compute_gate")
    require_callable(session_core, "reset_daily_flags")
    require_callable(session_core, "reset_caps_for_new_session")
    require_callable(equity_core, "update_equity_and_hwm")
    require_callable(pnl_core, "snapshot_es_pnl_and_orders")
    require_callable(position_core, "compute_position")
    require_callable(eod_core, "maybe_run_eod_bayes_gated")

    # order_core is central—confirm critical callables exist
    order_core = importlib.import_module("order_core")
    for name in [
        "flatten_all",
        "guard_naked_position",
        "reconcile_orphans",
        "place_protected_entry",
    ]:
        require_callable(order_core, name)

    print("=== OK: all required modules imported and required symbols exist ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
