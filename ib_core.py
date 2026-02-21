#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ib_core.py

IB connection + ES contract resolution (including auto-roll front month).

Patches (2025-12-28):
- Add explicit connect timeout + RequestTimeout
- Add small retry/backoff to avoid intermittent Gateway handshake stalls
- Log connected status after connect
- Remove duplicate "Resolving ES front-month..." log in resolve_contract (keep only inside resolve_es_front_month)
"""

import datetime as dt
import time
from typing import Optional, List, Tuple

from ib_insync import IB, Future, Contract


# ---------- Connect ----------

def connect_ib(args, logger) -> IB:
    """
    Connect to IB Gateway / TWS using ib_insync.

    Why this exists:
    - Gateway can occasionally accept TCP but stall apiStart briefly after login/restart.
    - Explicit timeout + small retry/backoff reduces intermittent connect flaps.
    """
    ib = IB()

    # Tolerant defaults (override via args if you add those CLI flags later)
    ib.RequestTimeout = int(getattr(args, "ib_request_timeout", 20) or 20)
    connect_timeout = int(getattr(args, "ib_connect_timeout", 20) or 20)
    max_attempts = int(getattr(args, "ib_connect_attempts", 3) or 3)

    last_err: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(
                f"Connecting to IB {args.ib_host}:{args.ib_port} clientId={args.ib_client_id} "
                f"(timeout={connect_timeout}s, attempt {attempt}/{max_attempts})..."
            )
            ib.connect(
                args.ib_host,
                args.ib_port,
                clientId=args.ib_client_id,
                timeout=connect_timeout,
            )
            logger.info(f"IB connected={ib.isConnected()}")
            return ib
        except Exception as e:
            last_err = e
            logger.warning(f"IB connect attempt {attempt} failed: {e}")
            # Exponential-ish backoff: 2s, 4s, 6s...
            time.sleep(2 * attempt)

    # If we get here, all attempts failed
    assert last_err is not None
    raise last_err


# ---------- ES front-month auto-roll ----------

def resolve_es_front_month(ib: IB, args, logger) -> Contract:
    """
    Auto-resolve ES *front-month* using contractDetails.
    - Try both GLOBEX and CME, because IB configs differ.
    - Pick nearest non-expired quarterly ES (H/M/U/Z).
    """
    logger.info("Resolving ES front-month contract via auto-roll (contractDetails)...")

    base_candidates = [
        # Try CME first because we know it works from test_es_contracts.py
        Future(symbol="ES", exchange="CME", currency="USD"),
        Future(
            symbol="ES",
            exchange=getattr(args, "exchange", None) or "GLOBEX",
            currency=getattr(args, "currency", None) or "USD",
        ),
    ]

    all_cds = []
    for base in base_candidates:
        try:
            logger.info(
                f"[auto-roll] reqContractDetails: symbol={base.symbol}, "
                f"exchange={base.exchange}, currency={base.currency}"
            )
            cds = ib.reqContractDetails(base)
            logger.info(f"[auto-roll] got {len(cds)} contractDetails for {base.exchange}")
        except Exception as e:
            logger.warning(f"[auto-roll] reqContractDetails failed for {base}: {e}")
            cds = []

        if cds:
            all_cds = cds
            break

    if not all_cds:
        raise ValueError(
            "No ES futures contractDetails returned from IB for ES/CME or ES/GLOBEX. "
            "This usually means: (1) CME futures are not enabled on the account, or "
            "(2) the account has no permission to request ES contracts. "
            "Please verify in TWS/Gateway that you can add an ES futures contract manually."
        )

    today = dt.date.today()
    quarterly_months = {3, 6, 9, 12}

    def parse_expiry(s: str) -> Optional[dt.date]:
        if not s:
            return None
        try:
            if len(s) == 6:  # YYYYMM
                year = int(s[0:4])
                month = int(s[4:6])
                day = 1
            elif len(s) == 8:  # YYYYMMDD
                year = int(s[0:4])
                month = int(s[4:6])
                day = int(s[6:8])
            else:
                return None
            return dt.date(year, month, day)
        except Exception:
            return None

    candidates: List[Tuple[dt.date, Contract]] = []
    for cd in all_cds:
        c = cd.contract
        if c.symbol != "ES":
            continue
        expiry = parse_expiry(c.lastTradeDateOrContractMonth)
        if not expiry:
            continue
        if expiry <= today:
            continue
        if expiry.month not in quarterly_months:
            continue
        candidates.append((expiry, c))

    if not candidates:
        raise ValueError(
            "ES contractDetails returned but no suitable non-expired quarterly "
            "front-month was found. Check IB's contract description for ES."
        )

    candidates.sort(key=lambda x: x[0])
    front_expiry, front_con = candidates[0]
    logger.info(
        f"[auto-roll] selected ES front-month: {front_con.localSymbol} "
        f"(expiry={front_expiry}) | conId={front_con.conId}"
    )

    qualified = ib.qualifyContracts(front_con)
    if not qualified:
        raise ValueError(f"qualifyContracts() failed for selected ES contract {front_con}")

    return qualified[0]


# ---------- Top-level resolver used by paper_trader ----------

def resolve_contract(ib: IB, args, logger) -> Contract:
    """
    Resolve the trading contract.

    Modes:
      - Auto-roll ES front month:
          --local-symbol auto
      - Explicit ES future by local symbol:
          --local-symbol ESH6, ESZ5, etc.
    """
    lsym = (args.local_symbol or "").strip()
    if not lsym:
        raise ValueError(
            "Missing --local-symbol (use 'auto' or explicit ES local like 'ESH6')."
        )

    # 1) Auto-roll mode: front-month ES via contractDetails
    if lsym.lower() in ("auto", "front", "roll", "es_auto"):
        # NOTE: log happens inside resolve_es_front_month; keep it single-source to reduce noise.
        return resolve_es_front_month(ib, args, logger)

    # 2) Explicit ES local-symbol (e.g., ESH6, ESZ5)
    sym = lsym.upper()
    if sym.startswith("ES") and len(sym) >= 4:
        logger.info("Resolving ES contract from localSymbol=%s (explicit mode)", sym)

        con_local = Future(
            localSymbol=sym,
            exchange=args.exchange,
            currency=args.currency,
        )
        qualified = ib.qualifyContracts(con_local)
        if qualified:
            logger.info("Resolved ES via localSymbol=%s", sym)
            return qualified[0]

        # Fallback: derive lastTradeDateOrContractMonth from month code/year digit
        month_code = sym[2]
        year_code = sym[3]

        month_map = {
            "H": "03",  # March
            "M": "06",  # June
            "U": "09",  # September
            "Z": "12",  # December
        }
        if month_code not in month_map:
            raise ValueError(f"Invalid ES month code in local symbol: {sym}")

        contract_month = "202" + year_code + month_map[month_code]

        logger.info(
            "LocalSymbol %s did not qualify; trying explicit month %s",
            sym,
            contract_month,
        )

        con_full = Future(
            symbol="ES",
            lastTradeDateOrContractMonth=contract_month,
            exchange=args.exchange,
            currency=args.currency,
            multiplier="50",
        )
        qualified2 = ib.qualifyContracts(con_full)
        if not qualified2:
            raise ValueError(
                f"Could not resolve ES contract from localSymbol={sym} "
                f"or month={contract_month}. Check CME futures permissions / data."
            )
        logger.info("Resolved ES via explicit month %s", contract_month)
        return qualified2[0]

    # 3) Unsupported symbol path (for now we only handle ES)
    raise ValueError(
        f"Unsupported --local-symbol value: {lsym!r}. "
        "Use 'auto' or an ES local symbol like 'ESH6'."
    )
