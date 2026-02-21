#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
es_self_test.py

Lightweight pre-flight check for ES Paper Trader.

What it does:
- Verifies that all core modules import successfully.
- Verifies that key functions/classes exist (e.g. snapshot_es_pnl_and_orders).
- Verifies that paper_trader.py itself imports cleanly and exposes main().
- Checks that key directories (logs, run, results, learn) exist or can be created.

Usage (from es_futures_backtester folder):

    .venv/Scripts/python.exe es_self_test.py

Exit code:
- 0  => all checks passed
- 1  => one or more checks failed
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import List, Tuple

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)


def print_header(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def check_import(module_name: str):
    try:
        mod = __import__(module_name)
        print(f"[OK]   import {module_name}")
        return True, mod
    except Exception as e:
        print(f"[FAIL] import {module_name} -> {e}")
        traceback.print_exc(limit=1)
        return False, None


def check_attrs(module, module_name: str, attrs: List[str]) -> bool:
    ok = True
    for attr in attrs:
        if not hasattr(module, attr):
            print(f"[FAIL] {module_name} missing attribute: {attr}")
            ok = False
        else:
            print(f"[OK]   {module_name}.{attr}")
    return ok


def ensure_dirs(dirs: List[str]) -> bool:
    ok = True
    for d in dirs:
        path = os.path.join(BASE_DIR, d)
        try:
            os.makedirs(path, exist_ok=True)
            print(f"[OK]   directory ready: {path}")
        except Exception as e:
            print(f"[FAIL] cannot create/access directory: {path} -> {e}")
            ok = False
    return ok


def main() -> int:
    print_header("ES Paper Trader - Module Wiring Self Test")

    all_ok = True

    # --- 1) Core module imports --------------------------------------------
    modules_to_check: List[Tuple[str, List[str]]] = [
        ("utils", ["setup_logger", "ensure_dir"]),
        ("risk_core", ["DayRisk", "default_week_state", "roll_week_if_needed"]),
        ("learner_bandit", ["ThompsonGaussian", "save_thompson", "load_thompson"]),
        ("learner_meta", ["MetaLearner"]),
        ("order_core", ["place_protected_entry", "reconcile_protective_orders", "reconcile_orphans"]),
        ("strategy_core", ["BarBuffer", "DEFAULT_ARMS", "build_signal_and_bands"]),
        ("ib_core", ["connect_ib", "resolve_contract"]),
        ("position_core", ["compute_position", "dynamic_contracts"]),
        ("bayes_core", ["build_bayes_training_set", "maybe_apply_bayes_best", "run_eod_bayes_opt_filtered"]),
        ("pnl_core", ["snapshot_es_pnl_and_orders"]),
        ("startup_core", ["maybe_daily_restart", "has_hedge_protection", "attach_startup_protection"]),
        ("gate_core", ["compute_gate"]),
        ("session_core", ["reset_daily_flags", "reset_caps_for_new_session"]),
        ("equity_core", ["update_equity_and_hwm"]),
        ("trade_bridge", ["handle_realized_pnl_event"]),
        ("margin_core", ["MarginManager", "MarginSnap"]),
    ]

    print_header("Step 1: Import modules + check key symbols")

    for mod_name, attrs in modules_to_check:
        ok_import, mod = check_import(mod_name)
        if not ok_import:
            all_ok = False
            continue
        if attrs:
            if not check_attrs(mod, mod_name, attrs):
                all_ok = False

    # --- 2) Check paper_trader.py itself -----------------------------------
    print_header("Step 2: Import paper_trader module")

    try:
        import paper_trader  # type: ignore
        print("[OK]   import paper_trader")
        if hasattr(paper_trader, "main"):
            print("[OK]   paper_trader.main is present")
        else:
            print("[FAIL] paper_trader.main is missing")
            all_ok = False
    except Exception as e:
        print(f"[FAIL] import paper_trader -> {e}")
        traceback.print_exc(limit=1)
        all_ok = False

    # --- 3) Check filesystem scaffolding -----------------------------------
    print_header("Step 3: Check required directories")

    dirs_ok = ensure_dirs([
        "logs",
        "logs/watchdog",
        "logs/child",
        "logs/learn",
        "run",
        "results",
        "data",
        "data/state",
        "learn",
    ])
    if not dirs_ok:
        all_ok = False

    # --- 4) Summary ---------------------------------------------------------
    print_header("Self test summary")

    if all_ok:
        print("[PASS] All module + filesystem checks look good.")
        print("       You are clear to launch watchdog / paper_trader.")
        return 0
    else:
        print("[FAIL] One or more checks failed. Fix above issues before trading.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
