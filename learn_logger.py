import csv
import json
import math
import os
import time


def reward_R_from_pnl(pnl_dollars, risk_ticks, tick_size, qty, multiplier=50.0):
    risk_dollars = max(1e-9, float(risk_ticks) * float(tick_size) * float(multiplier) * float(qty))
    return float(pnl_dollars) / risk_dollars


class LearnLogger:
    def __init__(self, enable, symbol, state_path, log_dir):
        self.enable = bool(enable)
        self.symbol = symbol
        self.state_path = state_path
        self.log_dir = log_dir
        self._last = {}
        if not self.enable:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
        self._state = self._load_state()
        self._csv = os.path.join(self.log_dir, f"learn_{self.symbol}.csv")
        if not os.path.exists(self._csv):
            with open(self._csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        "ts",
                        "symbol",
                        "arm",
                        "side",
                        "qty",
                        "entry",
                        "exit",
                        "pnl$",
                        "risk$",
                        "R",
                    ]
                )

    def _load_state(self):
        try:
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"arms": {}, "total": 0}

    def _save_state(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False)
        os.replace(tmp, self.state_path)

    # --- logging hooks you call from your strategy ---
    def record_decision(self, arm, meta=None):
        if not self.enable:
            return
        self._last = {"arm": str(arm), "t": time.time(), "meta": meta or {}}

    def on_enter(self, side, entry_price, qty):
        if not self.enable:
            return
        self._last.update({"side": str(side), "entry": float(entry_price), "qty": float(qty)})

    def on_exit(
        self,
        exit_price,
        fees=0.0,
        risk_ticks=1,
        tick_size=0.25,
        qty=None,
        multiplier=50.0,
    ):
        if not self.enable:
            return
        d = dict(self._last)
        d["exit"] = float(exit_price)
        if qty is not None:
            d["qty"] = float(qty)
        qty = float(d.get("qty", 1.0))
        side = str(d.get("side", "BUY")).upper()
        entry = float(d.get("entry", d["exit"]))
        pnl = (d["exit"] - entry) * float(multiplier) * qty
        if side.startswith("S"):  # short
            pnl = -pnl
        pnl -= float(fees)

        risk_dollars = max(1e-9, float(risk_ticks) * float(tick_size) * float(multiplier) * qty)
        R = pnl / risk_dollars

        arm = d.get("arm", "unknown")
        arms = self._state.setdefault("arms", {})
        st = arms.setdefault(arm, {"n": 0, "value": 0.0})
        st["n"] += 1
        st["value"] += (R - st["value"]) / st["n"]
        self._state["total"] = self._state.get("total", 0) + 1
        self._save_state()

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self._csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    ts,
                    self.symbol,
                    arm,
                    side,
                    qty,
                    entry,
                    d["exit"],
                    pnl,
                    risk_dollars,
                    R,
                ]
            )

    # Optional: simple UCB chooser you can use to pick an arm
    def choose_arm_ucb(self, enabled_arms, c=1.2):
        if not self.enable:
            return enabled_arms[0]
        arms = self._state.setdefault("arms", {})
        total = max(1, self._state.get("total", 0))
        best, best_score = None, -1e9
        for a in enabled_arms:
            st = arms.get(a, {"n": 0, "value": 0.0})
            if st["n"] == 0:
                return a  # force exploration
            ucb = st["value"] + c * math.sqrt(math.log(total) / st["n"])
            if ucb > best_score:
                best, best_score = a, ucb
        return best or enabled_arms[0]
