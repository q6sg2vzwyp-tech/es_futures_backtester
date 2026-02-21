from __future__ import annotations

# singleton_win32.py (compat shim)
#
# Legacy callers may import this. Delegate to pt/singleton.py.

from pt.singleton import acquire_or_exit, acquire_paper_trader_lock, lock_path

__all__ = ["acquire_or_exit", "acquire_paper_trader_lock", "lock_path"]
