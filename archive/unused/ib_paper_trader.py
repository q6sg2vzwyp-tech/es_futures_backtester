import argparse
import datetime as dt
import logging

import pandas as pd
from ib_insync import *


# --------------------
# Helpers
# --------------------
def ema(values, period):
    return pd.Series(values).ewm(span=period, adjust=False).mean().iloc[-1]


def rsi(values, period=14):
    series = pd.Series(values)
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down
    return 100 - (100 / (1 + rs.iloc[-1]))


# --------------------
# Trader
# --------------------
class Trader:
    def __init__(self, ib, contract, args):
        self.ib = ib
        self.contract = contract
        self.args = args
        self.position = 0
        self.order_refs = {}
        self.bars = []

    def log(self, msg):
        print(msg)
        logging.info(msg)

    def handle_bar(self, bar):
        self.bars.append(bar)
        if len(self.bars) < max(self.args.fast, self.args.slow, self.args.rsi) + 1:
            return

        closes = [b.close for b in self.bars]
        ema_f = ema(closes, self.args.fast)
        ema_s = ema(closes, self.args.slow)
        rsi_val = rsi(closes, self.args.rsi)

        sig = None
        if ema_f > ema_s and rsi_val > 55:
            sig = "BUY"
        elif ema_f < ema_s and rsi_val < 45:
            sig = "SELL"

        self.log(f"[SIG ] {bar.time} ema_f={ema_f:.2f} ema_s={ema_s:.2f} rsi={rsi_val:.1f} → {sig}")

        if not self.args.place_orders or sig is None:
            return

        if sig == "BUY" and self.position <= 0:
            self.enter("BUY", bar.close)
        elif sig == "SELL" and self.position >= 0:
            self.enter("SELL", bar.close)

        # flat-at close
        if self.args.flat_at:
            flat_time = dt.datetime.strptime(self.args.flat_at, "%H:%M").time()
            if bar.time.time() >= flat_time and self.position != 0:
                self.log("[Flat] Closing all at flat-at time")
                self.exit_all(bar.close)

    def enter(self, side, price):
        qty = self.args.qty
        action = "BUY" if side == "BUY" else "SELL"
        limit_price = price + (
            self.args.entry_limit_offset if action == "BUY" else -self.args.entry_limit_offset
        )

        entry = LimitOrder(action, qty, limit_price, tif=self.args.tif)
        self.log(f"[ORDER] Placing {side} limit {qty}@{limit_price}")
        trade = self.ib.placeOrder(self.contract, entry)

        # fallback to market if not filled
        self.ib.sleep(self.args.entry_timeout_s)
        if not trade.isDone():
            self.log("[ORDER] Timeout, switching to market")
            self.ib.cancelOrder(entry)
            mkt = MarketOrder(action, qty, tif=self.args.tif)
            trade = self.ib.placeOrder(self.contract, mkt)

        self.position = qty if action == "BUY" else -qty
        self.order_refs["entry"] = trade

        # place bracket if enabled
        if self.args.bracket:
            tp = limit_price + (
                self.args.tp_ticks * 0.25 if action == "BUY" else -self.args.tp_ticks * 0.25
            )
            sl = limit_price - (
                self.args.sl_ticks * 0.25 if action == "BUY" else -self.args.sl_ticks * 0.25
            )
            self.log(f"[BRKT] TP={tp} SL={sl}")

            take_profit = LimitOrder("SELL" if action == "BUY" else "BUY", qty, tp, tif="GTC")
            stop_loss = StopOrder("SELL" if action == "BUY" else "BUY", qty, sl, tif="GTC")

            oco = [take_profit, stop_loss]
            for o in oco:
                o.parentId = trade.order.orderId
                self.ib.placeOrder(self.contract, o)
            self.order_refs["tp"] = take_profit
            self.order_refs["sl"] = stop_loss

    def exit_all(self, price):
        if self.position == 0:
            return
        action = "SELL" if self.position > 0 else "BUY"
        mkt = MarketOrder(action, abs(self.position))
        self.log(f"[EXIT] Closing {self.position} @ MKT {price}")
        self.ib.placeOrder(self.contract, mkt)
        self.position = 0
        self.order_refs.clear()


# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--clientId", type=int, default=9)
    parser.add_argument("--use-delayed", action="store_true")
    parser.add_argument("--fast", type=int, default=8)
    parser.add_argument("--slow", type=int, default=20)
    parser.add_argument("--rsi", type=int, default=14)
    parser.add_argument("--no-rsi", action="store_true")
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument("--place-orders", action="store_true")
    parser.add_argument("--bracket", action="store_true")
    parser.add_argument("--tp-ticks", type=int, default=12)
    parser.add_argument("--sl-ticks", type=int, default=8)
    parser.add_argument("--trail-ticks", type=int, default=0)
    parser.add_argument("--entry-limit-offset", type=float, default=2)
    parser.add_argument("--entry-timeout-s", type=int, default=15)
    parser.add_argument("--tif", default="DAY")
    parser.add_argument("--rth-only", action="store_true")
    parser.add_argument("--min-atr-ticks", type=float, default=0)
    parser.add_argument("--max-spread-ticks", type=float, default=999)
    parser.add_argument("--flat-at", default=None)
    parser.add_argument("--pyramid-max", type=int, default=0)
    parser.add_argument("--pyramid-step-atr", type=float, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(filename="trades.log", level=logging.INFO, format="%(asctime)s %(message)s")

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.clientId)

    if args.use - delayed:
        ib.reqMarketDataType(4)
        print("Market data type: DELAYED (4)")
    else:
        ib.reqMarketDataType(1)
        print("Market data type: LIVE (1)")

    today = dt.datetime.now().strftime("%Y%m%d")
    contract = Future("ES", today, "CME")
    ib.qualifyContracts(contract)
    print(f"Using contract: {contract.localSymbol}  ({contract.conId})")

    trader = Trader(ib, contract, args)

    def on_bar(bar):
        trader.handle_bar(bar)

    bars = ib.reqRealTimeBars(contract, 5, "TRADES", True)
    bars.updateEvent += on_bar

    print("▶ Streaming ES real-time bars (aggregating to 1m)… Ctrl+C to stop")
    ib.run()


if __name__ == "__main__":
    main()
