from __future__ import annotations

import datetime as dt
from typing import List, Optional, Tuple

# 3rd party
from ib_insync import IB, Future, Contract

# NOTE:
# - No top-level execution.
# - Keep behavior identical to paper_trader.py.
# - Any helper functions referenced (e.g., parse_ib_date, log) are expected
#   to exist in the caller module (paper_trader.py) and be passed in if needed.


def qualify_local_symbol(ib: IB, local_symbol: str, exchange: str = "CME") -> Contract:
    cds = ib.reqContractDetails(Future(localSymbol=local_symbol, exchange=exchange))
    if not cds:
        raise RuntimeError(f"Local symbol {local_symbol} not found on {exchange}")
    con = cds[0].contract
    ib.qualifyContracts(con)
    return con


def _pick_by_expiry(cds, parse_ib_date) -> Contract:
    """
    Fallback: choose earliest expiry from reqContractDetails results.
    """
    best = None
    best_date = None
    for cd in cds:
        d = parse_ib_date(cd.contract.lastTradeDateOrContractMonth)
        if not d:
            continue
        if best is None or d < best_date:
            best = cd.contract
            best_date = d
    if best is None:
        raise RuntimeError("Could not resolve front contract from contractDetails")
    return best


def _daily_volume_for_contract(ib: IB, con: Contract, log_fn=None) -> float:
    """
    Use recent daily TRADES volume as a liquidity proxy.
    """
    try:
        bars = ib.reqHistoricalData(
            con,
            endDateTime="",
            durationStr="3 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
            keepUpToDate=False,
        )
        if not bars:
            return 0.0
        vol = getattr(bars[-1], "volume", None)
        return float(vol) if vol is not None else 0.0
    except Exception as e:
        if log_fn:
            log_fn("roll_vol_err", conId=getattr(con, "conId", None), err=str(e))
        return 0.0


def _oi_for_contract(ib: IB, con: Contract, log_fn=None) -> float:
    """
    Try to get futures open interest via generic tick 588.
    """
    tkr = None
    try:
        tkr = ib.reqMktData(con, genericTickList="588", snapshot=True, regulatorySnapshot=False)
        ib.sleep(1.0)
        oi = getattr(tkr, "openInterest", None)
        if oi is None:
            if log_fn:
                log_fn(
                    "roll_oi_missing",
                    conId=getattr(con, "conId", None),
                    localSymbol=getattr(con, "localSymbol", None),
                )
            return 0.0
        try:
            v = float(oi)
            if v != v:  # NaN
                if log_fn:
                    log_fn(
                        "roll_oi_nan",
                        conId=getattr(con, "conId", None),
                        localSymbol=getattr(con, "localSymbol", None),
                    )
                return 0.0
            return v
        except Exception:
            return 0.0
    except Exception as e:
        if log_fn:
            log_fn(
                "roll_oi_err",
                conId=getattr(con, "conId", None),
                localSymbol=getattr(con, "localSymbol", None),
                err=str(e),
            )
        return 0.0
    finally:
        try:
            if tkr is not None:
                ib.cancelMktData(con)
        except Exception:
            pass


def _liquidity_metrics_for_contract(ib: IB, con: Contract, log_fn=None) -> Tuple[float, float]:
    """
    Return (daily_volume, open_interest) for this futures contract.
    """
    vol = _daily_volume_for_contract(ib, con, log_fn=log_fn)
    oi = _oi_for_contract(ib, con, log_fn=log_fn)

    # Normalize weird values
    if vol < 0:
        vol = 0.0
    if oi < 0:
        oi = 0.0

    if log_fn:
        log_fn(
            "roll_liquidity_metrics",
            localSymbol=getattr(con, "localSymbol", None),
            conId=getattr(con, "conId", None),
            dailyVol=vol,
            openInterest=oi,
        )
    return vol, oi


def _pick_by_liquidity(
    ib: IB,
    cds,
    parse_ib_date,
    log_fn=None,
    max_candidates: int = 3,
) -> Contract:
    """
    Among the nearest expiries, choose the one with best liquidity using
    score = 0.7 * norm_volume + 0.3 * norm_oi

    Fallback to earliest expiry if we cannot compute any metrics.
    """
    # Extract (contract, expiry date)
    contracts: List[Tuple[Contract, dt.date]] = []
    for cd in cds:
        d = parse_ib_date(cd.contract.lastTradeDateOrContractMonth)
        if not d:
            continue
        contracts.append((cd.contract, d))

    if not contracts:
        raise RuntimeError("No contracts with parsable expiry; cannot roll by liquidity")

    # Sort by expiry ascending and take the nearest N
    contracts.sort(key=lambda x: x[1])
    candidates = contracts[: max(1, max_candidates)]

    # Measure liquidity
    scored: List[Tuple[Contract, float, float, dt.date]] = []
    max_vol = 0.0
    max_oi = 0.0

    for con, d in candidates:
        vol, oi = _liquidity_metrics_for_contract(ib, con, log_fn=log_fn)
        max_vol = max(max_vol, vol)
        max_oi = max(max_oi, oi)
        scored.append((con, vol, oi, d))

    if not scored:
        return _pick_by_expiry(cds, parse_ib_date=parse_ib_date)

    best_con = None
    best_score = -1.0
    best_exp = None
    best_vol = 0.0
    best_oi = 0.0

    for con, vol, oi, d in scored:
        # Very basic sanity veto: if both vol and oi are zero, skip this contract.
        if vol <= 0 and oi <= 0:
            continue

        nv = (vol / max_vol) if max_vol > 0 else 0.0
        no = (oi / max_oi) if max_oi > 0 else 0.0
        score = 0.7 * nv + 0.3 * no

        if log_fn:
            log_fn(
                "roll_candidate",
                localSymbol=getattr(con, "localSymbol", None),
                conId=getattr(con, "conId", None),
                expiry=str(d),
                dailyVol=vol,
                openInterest=oi,
                score=score,
            )

        if score > best_score:
            best_score = score
            best_con = con
            best_exp = d
            best_vol = vol
            best_oi = oi

    if best_con is None:
        best = _pick_by_expiry(cds, parse_ib_date=parse_ib_date)
        if log_fn:
            log_fn(
                "roll_choice",
                mode="expiry_fallback",
                localSymbol=getattr(best, "localSymbol", None),
                conId=getattr(best, "conId", None),
                expiry=str(parse_ib_date(best.lastTradeDateOrContractMonth)),
            )
        return best

    if log_fn:
        log_fn(
            "roll_choice",
            mode="liquidity",
            localSymbol=getattr(best_con, "localSymbol", None),
            conId=getattr(best_con, "conId", None),
            expiry=str(best_exp),
            dailyVol=best_vol,
            openInterest=best_oi,
        )
    return best_con


def mk_contract(ib: IB, args, *, parse_ib_date=None, log_fn=None) -> Contract:
    # Dependency injection hooks (default to local helpers)
    if parse_ib_date is None:
        parse_ib_date = _parse_ib_date
    if log_fn is None:
        def log_fn(*_a, **_k):
            return

    """
    Contract resolution:

    - If --local-symbol is provided: use that exact contract (no rolling).
    - Else, use --symbol and exchange=CME, and either:
        * default earliest-expiry selection, or
        * if --roll-by-volume: pick by liquidity (volume + OI) among nearest expiries.
    """

    # 1) Exact localSymbol (e.g. ESH6)
    if getattr(args, "local_symbol", None):
        con = qualify_local_symbol(ib, args.local_symbol, "CME")
        print(
            f"[CONTRACT] Using {con.localSymbol} conId={con.conId} "
            f"exp={con.lastTradeDateOrContractMonth} (fixed local symbol)"
        )
        return con

    # 2) Symbol-based lookup (e.g. ES) -> get all listed futures on CME.
    base_symbol = getattr(args, "symbol", None) or "ES"
    cds = ib.reqContractDetails(Future(symbol=base_symbol, exchange="CME", currency="USD"))
    if not cds:
        raise RuntimeError(f"Symbol {base_symbol} not found on CME; supply --local-symbol")

    # 3) Decide by expiry or liquidity, depending on flag.
    use_liquidity_roll = bool(getattr(args, "roll_by_volume", False))

    if use_liquidity_roll:
        con = _pick_by_liquidity(ib, cds, parse_ib_date=parse_ib_date, log_fn=log_fn)
        mode = "liquidity"
    else:
        con = _pick_by_expiry(cds, parse_ib_date=parse_ib_date)
        mode = "expiry"

    ib.qualifyContracts(con)
    print(
        f"[CONTRACT] Using {con.localSymbol} conId={con.conId} "
        f"exp={con.lastTradeDateOrContractMonth} (mode={mode})"
    )
    return con


def contract_multiplier(ib: IB, con: Contract) -> float:
    try:
        cds = ib.reqContractDetails(con)
        mul = cds[0].contract.multiplier
        m = float(mul) if mul is not None else 1.0
        return m if m > 0 else 1.0
    except Exception:
        return 1.0
