# position_core.py
import math
from ib_insync import IB, Contract

def compute_position(ib: IB, con: Contract) -> int:
    net = 0
    for p in ib.positions():
        if p.contract.conId == con.conId:
            net += p.position
    return net

def dynamic_contracts(
    equity: float,
    risk_pct: float,
    risk_ticks: int,
    tick_size: float,
    multiplier: float,
    max_contracts: int,
) -> int:
    per_contract_risk = risk_ticks * tick_size * multiplier
    if per_contract_risk <= 0:
        return 1
    raw = equity * risk_pct / per_contract_risk
    return max(1, min(max_contracts, int(math.floor(raw))))

