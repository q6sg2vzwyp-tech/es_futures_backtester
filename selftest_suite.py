#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
selftest_suite.py

Module & wiring smoke tests for ES Paper Trader project.

What it does:
- Imports your top-level modules (repo root .py files only)
- Scans for placeholder markers ("TODO", "placeholder", "hook placeholder", etc.)
- Verifies critical functions/classes exist in key modules (light contract)
- Optional: IB connect and auto-roll/contract resolve (NO orders)
- Optional: checks run artifacts exist (heartbeat, trades.csv) after a short live run

Usage examples:
  python selftest_suite.py --strict --no-ib
  python selftest_suite.py --strict --ib --ib-host 127.0.0.1 --ib-port 4002 --ib-client-id 111
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EXCLUDE_DIR_PATTERNS = [
    os.sep + ".venv" + os.sep,
    os.sep + "__pycache__" + os.sep,
    os.sep + ".git" + os.sep,
    os.sep + "archive" + os.sep,
    os.sep + "tools" + os.sep,
    os.sep + "profiles" + os.sep,
    os.sep + "_archive_broken_IGNORE" + os.sep,
]

PLACEHOLDER_PATTERNS = [
    r"\bTODO\b",
    r"\bFIXME\b",
    r"hook placeholder",
    r"\bplaceholder\b",
    r"raise\s+NotImplementedError",
]

REQUIRED_SYMBOLS = {
    "paper_trader": ["main"],
    "utils": ["ct_now", "in_time_window"],
    "order_core": ["place_protected_entry"],
    "gate_core": ["compute_gate"],
    "day_policy_core": ["_caps_for_state"],
    "hb_core": [
        "build_and_write_heartbeat",
        "build_heartbeat_payload",
        "emit_hb_snapshot",
        "hb_update_entry_and_unreal",
    ],
    "bandit_core": ["Bandit", "ArmState", "DEFAULT_ARM_NAMES"],
    "eod_core": ["maybe_run_eod_bayes", "maybe_run_eod_bayes_gated", "EODState", "EODResult"],
    "margin_core": ["MarginManager"],
}


@dataclass
class TestResult:
    name: str
    ok: bool
    detail: str = ""


def is_excluded(path: str) -> bool:
    p = os.path.abspath(path)

    lower = p.lower()

    # Exclude installed packages
    if (os.sep + "site-packages" + os.sep) in lower:
        return True

    # Exclude any venv-ish folder: .venv, venv, .venv313, etc.
    if re.search(rf"{re.escape(os.sep)}\.?venv\d*{re.escape(os.sep)}", lower):
        return True

    for pat in EXCLUDE_DIR_PATTERNS:
        if pat in p:
            return True
    return False


def find_project_py_files() -> List[str]:
    out: List[str] = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # prune excluded dirs
        pruned = []
        for d in dirs:
            full = os.path.join(root, d)
            if is_excluded(full):
                pruned.append(d)
        for d in pruned:
            dirs.remove(d)

        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(root, fn)
            if is_excluded(full):
                continue
            out.append(full)
    return sorted(out)


def placeholder_scan(strict: bool) -> TestResult:
    files = find_project_py_files()
    hits: List[Tuple[str, int, str]] = []
    regexes = [re.compile(pat, flags=re.IGNORECASE | re.MULTILINE) for pat in PLACEHOLDER_PATTERNS]

    for f in files:
        rel = os.path.relpath(f, PROJECT_ROOT).replace("\\", "/")
        # Do not scan this file; otherwise it will flag its own documentation text.
        if rel == "selftest_suite.py":
            continue

        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fp:
                txt = fp.read()
        except Exception as e:
            return TestResult("placeholder_scan", False, f"Failed reading {rel}: {e}")

        for i, line in enumerate(txt.splitlines(), start=1):
            for rx in regexes:
                if rx.search(line):
                    hits.append((rel, i, line.strip()))
                    break

    if hits:
        msg_lines = ["Placeholder markers found:"]
        for rel, i, line in hits[:80]:
            msg_lines.append(f"  {rel}:{i}: {line}")
        if len(hits) > 80:
            msg_lines.append(f"  ...and {len(hits)-80} more")
        detail = "\n".join(msg_lines)
        return TestResult("placeholder_scan", (not strict), detail if strict else ("WARN\n" + detail))

    return TestResult("placeholder_scan", True, "No placeholder markers detected.")


def import_all_modules() -> TestResult:
    # Import only top-level modules in the repo root.
    mods = []
    for fn in os.listdir(PROJECT_ROOT):
        if not fn.endswith(".py"):
            continue
        if fn == "selftest_suite.py":
            continue
        mods.append(fn[:-3])

    failures = []
    for m in sorted(mods):
        try:
            importlib.import_module(m)
        except Exception as e:
            failures.append((m, repr(e), traceback.format_exc(limit=12)))

    if failures:
        msg = ["Module import failures:"]
        for m, e, tb in failures:
            msg.append(f"\n--- {m} ---\n{e}\n{tb}")
        return TestResult("import_all_modules", False, "\n".join(msg))

    return TestResult("import_all_modules", True, f"Imported {len(mods)} top-level modules successfully.")


def required_symbols_check() -> TestResult:
    missing: List[str] = []
    for mod_name, syms in REQUIRED_SYMBOLS.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            missing.append(f"{mod_name}: import failed: {e}")
            continue
        for s in syms:
            if not hasattr(mod, s):
                missing.append(f"{mod_name}.{s} missing")

    if missing:
        return TestResult("required_symbols_check", False, "Missing required symbols:\n" + "\n".join(missing))
    return TestResult("required_symbols_check", True, "All required symbols present.")


def ib_connect_and_resolve_contract(
    host: str,
    port: int,
    client_id: int,
    symbol: str,
    exchange: str,
    currency: str,
) -> TestResult:
    """
    Connect to IBKR via ib_insync and resolve front month contract in the same manner as paper_trader.
    NO orders.
    """
    try:
        from ib_insync import IB
    except Exception as e:
        return TestResult("ib_connect", False, f"ib_insync import failed: {e}")

    auto_roll_fn = None
    try:
        import auto_roll_core  # adjust if your module name differs
        auto_roll_fn = getattr(auto_roll_core, "resolve_front_month", None)
    except Exception:
        auto_roll_fn = None

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=8)
    except Exception as e:
        return TestResult("ib_connect", False, f"IB connect failed: {e}")

    try:
        if auto_roll_fn is not None:
            c = auto_roll_fn(ib=ib, symbol=symbol, exchange=exchange, currency=currency)
            detail = f"Resolved via auto_roll_core.resolve_front_month -> {c}"
        else:
            from ib_insync import Future
            base = Future(symbol=symbol, exchange=exchange, currency=currency)
            cds = ib.reqContractDetails(base)
            if not cds:
                return TestResult("ib_resolve_contract", False, "reqContractDetails returned 0 results")
            cds_sorted = sorted(cds, key=lambda x: x.contract.lastTradeDateOrContractMonth or "")
            c = cds_sorted[0].contract
            detail = f"Resolved via fallback reqContractDetails -> {c.localSymbol} exp={c.lastTradeDateOrContractMonth} conId={c.conId}"

        return TestResult("ib_connect_and_resolve_contract", True, detail)
    except Exception as e:
        return TestResult(
            "ib_connect_and_resolve_contract",
            False,
            f"Contract resolve failed: {e}\n{traceback.format_exc(limit=8)}",
        )
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def artifact_check() -> TestResult:
    hb = os.path.join(PROJECT_ROOT, "run", "heartbeat.txt")
    trades = os.path.join(PROJECT_ROOT, "results", "trades.csv")
    missing = []
    if not os.path.exists(hb):
        missing.append("run/heartbeat.txt")
    if not os.path.exists(trades):
        missing.append("results/trades.csv")

    if missing:
        return TestResult("artifact_check", False, "Missing artifacts:\n" + "\n".join(missing))
    return TestResult("artifact_check", True, "Artifacts present: heartbeat.txt and trades.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="Fail if ANY placeholder markers are found.")
    ap.add_argument("--no-ib", action="store_true", help="Skip IB connection tests.")
    ap.add_argument("--ib", action="store_true", help="Enable IB connection tests.")
    ap.add_argument("--ib-host", default="127.0.0.1")
    ap.add_argument("--ib-port", type=int, default=4002)
    ap.add_argument("--ib-client-id", type=int, default=111)

    ap.add_argument("--contract", default="ES")
    ap.add_argument("--exchange", default="CME")
    ap.add_argument("--currency", default="USD")

    ap.add_argument("--check-artifacts", action="store_true")
    args = ap.parse_args()

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    results: List[TestResult] = []

    results.append(placeholder_scan(strict=args.strict))
    results.append(import_all_modules())
    results.append(required_symbols_check())

    do_ib = args.ib and (not args.no_ib)
    if do_ib:
        results.append(
            ib_connect_and_resolve_contract(
                host=args.ib_host,
                port=args.ib_port,
                client_id=args.ib_client_id,
                symbol=args.contract,
                exchange=args.exchange,
                currency=args.currency,
            )
        )

    if args.check_artifacts:
        results.append(artifact_check())

    ok_all = True
    print("\n================ SELFTEST REPORT ================\n")
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"[{status}] {r.name}")
        if r.detail:
            print(r.detail)
            print()
        if not r.ok:
            ok_all = False

    print("=================================================\n")
    if not ok_all:
        print("SELFTEST: FAILED")
        return 2
    print("SELFTEST: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
