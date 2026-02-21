from __future__ import annotations

# pt_singleton.py (compat shim)
#
# paper_trader.py historically imported acquire_or_exit from this module.
# The single source of truth now lives in pt/singleton.py.

from pt.singleton import acquire_or_exit, acquire_paper_trader_lock, lock_path

__all__ = ["acquire_or_exit", "acquire_paper_trader_lock", "lock_path"]
