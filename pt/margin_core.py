#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
margin_core.py

Simple risk-based "soft margin" clamp for ES only (for now).

Used by paper_trader.py:

- We build a MarginSnap each loop with:
    * product="ES"
    * per_contract_init â‰ˆ 1R risk per contract
    * available_funds = equity - |net| * per_contract_init
    * net_liq = equity

- MarginManager.update_snapshot(snap) stores the latest snapshot.

- MarginManager.clamp_entry_size(...) takes:
    * desired_qty_delta (e.g. +1 or -2)
    * current_net_qty (e.g. 0, +1, -1)
    * per_contract_init

  and returns a CLAMPED delta so that:

    abs(net_after) <= max_contracts_soft

- heartbeat_fields() exposes diagnostics for the heartbeat dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any


@dataclass
class MarginSnap:
    """
    Lightweight snapshot used by MarginManager.

    product:
        Symbol / key (e.g. "ES").

    per_contract_init:
        Dollar risk per contract (we approximate this with 1R:
        risk_ticks * tick_size * multiplier).

    available_funds:
        "Headroom" dollars after current position risk is accounted for.

    net_liq:
        Approximate account net liquidation value / equity.
    """
    product: str = "ES"
    per_contract_init: float = 0.0
    available_funds: float = 0.0
    net_liq: float = 0.0


class MarginManager:
    """
    Very simple per-product soft-margin manager.

    Key ideas:
    ----------
    - We DO NOT try to be a broker. We just ensure the *position size*
      stays within a reasonable fraction of net_liq.

    - max_usage_soft:
        Target max *risk* as a fraction of net_liq, used for clamping.
        (e.g. 0.80 â†’ at most ~80% of net_liq in risk collateral.)

    - max_usage_hard:
        Reserved for future use; we keep it but currently clamp with
        max_usage_soft only.
    """

    def __init__(
        self,
        *,
        logger=None,
        max_usage_soft: float = 0.80,
        max_usage_hard: float = 0.95,
    ) -> None:
        self.logger = logger
        self.max_usage_soft = float(max_usage_soft)
        self.max_usage_hard = float(max_usage_hard)

        # Latest snapshots keyed by product symbol
        self._snap_by_product: Dict[str, MarginSnap] = {}

        # Last clamp calculus, for heartbeat diagnostics
        self._last_calc: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------
    def _log(self, level: str, msg: str) -> None:
        if self.logger is None:
            return
        fn = getattr(self.logger, level, None)
        if fn is None:
            return
        fn(msg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_snapshot(self, snap: MarginSnap) -> None:
        """
        Store the latest margin snapshot for a product.
        """
        self._snap_by_product[snap.product] = snap

    def clamp_entry_size(
        self,
        *,
        product: str,
        desired_qty_delta: int,
        current_net_qty: int,
        per_contract_init: float,
    ) -> int:
        """
        Clamp the change in position (desired_qty_delta) to respect a
        soft risk limit based on per_contract_init and net_liq.

        Returns:
            clamped_qty_delta (int)

        Example:
            current_net_qty = 0
            desired_qty_delta = +5
            per_contract_init = 1,000
            net_liq = 100,000
            max_usage_soft = 0.80

            => max_risk_dollars = 80,000
               max_contracts_soft = 80,000 / 1,000 = 80

            So we allow +5 (no clamp).

        If we had a tiny net_liq, this could clamp it down to +1 or 0.
        """
        try:
            desired_qty_delta = int(desired_qty_delta)
            current_net_qty = int(current_net_qty)
        except Exception:
            # If something weird is passed in, fail open but log
            self._log(
                "warning",
                f"[margin_core] non-int desired/current qty: "
                f"desired={desired_qty_delta}, current={current_net_qty}",
            )
            return int(desired_qty_delta)

        snap: Optional[MarginSnap] = self._snap_by_product.get(product)

        # If we don't have a snapshot or bogus params, just passthrough.
        if snap is None or per_contract_init <= 0 or snap.net_liq <= 0:
            self._last_calc = {
                "product": product,
                "mode": "passthrough_no_snapshot",
                "desired_delta": desired_qty_delta,
                "current_net": current_net_qty,
            }
            return desired_qty_delta

        # Soft max risk (dollars) allowed for *this product*.
        max_risk_dollars = snap.net_liq * self.max_usage_soft

        # Max contracts allowed by soft risk limit.
        # If per_contract_init is huge, this may be 0 â€“ that's okay.
        max_contracts_soft = int(max_risk_dollars // per_contract_init)
        if max_contracts_soft < 0:
            max_contracts_soft = 0

        # Proposed new net position if we just applied desired_delta.
        proposed_net = current_net_qty + desired_qty_delta

        # Clamp absolute net contracts to +/- max_contracts_soft.
        if max_contracts_soft == 0:
            clamped_net = 0
        else:
            clamped_net = max(
                -max_contracts_soft,
                min(proposed_net, max_contracts_soft),
            )

        clamped_delta = clamped_net - current_net_qty

        # Max additional contracts allowed from the *current* net.
        max_additional = max(0, max_contracts_soft - abs(current_net_qty))

        # Approximate current usage ratio (0..1) using current_net only.
        current_risk_dollars = abs(current_net_qty) * per_contract_init
        usage_ratio = 0.0
        if snap.net_liq > 0:
            usage_ratio = current_risk_dollars / (snap.net_liq * self.max_usage_soft)

        # Store diagnostics for heartbeat_fields()
        self._last_calc = {
            "product": product,
            "mode": "clamp",
            "desired_delta": desired_qty_delta,
            "current_net": current_net_qty,
            "proposed_net": proposed_net,
            "clamped_net": clamped_net,
            "clamped_delta": clamped_delta,
            "max_contracts_soft": max_contracts_soft,
            "max_additional": max_additional,
            "usage_ratio": usage_ratio,
            "per_contract_init": per_contract_init,
            "net_liq": snap.net_liq,
            "available_funds": snap.available_funds,
        }

        # Log only when we actually clamp something.
        if clamped_delta != desired_qty_delta:
            self._log(
                "info",
                "[margin_core] clamp_entry_size: "
                f"product={product} desired_delta={desired_qty_delta} "
                f"current_net={current_net_qty} proposed_net={proposed_net} "
                f"-> clamped_net={clamped_net} (max_soft={max_contracts_soft})",
            )

        return clamped_delta

    def heartbeat_fields(self) -> Dict[str, Any]:
        """
        Fields to splice into the heartbeat payload.

        We keep names prefixed with 'margin_' to avoid collisions.
        """
        if not self._snap_by_product:
            return {
                "margin_enabled": False,
            }

        # Prefer ES if present; otherwise take any snapshot
        snap = self._snap_by_product.get("ES") or next(
            iter(self._snap_by_product.values())
        )

        fields: Dict[str, Any] = {
            "margin_enabled": True,
            "margin_product": snap.product,
            "margin_net_liq": float(snap.net_liq),
            "margin_per_contract_init": float(snap.per_contract_init),
            "margin_available_funds": float(snap.available_funds),
        }

        # Merge in last clamp diagnostics if available
        if self._last_calc:
            fields.update(
                {
                    "margin_mode": self._last_calc.get("mode"),
                    "margin_max_contracts_soft": self._last_calc.get(
                        "max_contracts_soft"
                    ),
                    "margin_max_additional": self._last_calc.get("max_additional"),
                    "margin_usage_ratio": self._last_calc.get("usage_ratio"),
                }
            )

        return fields

