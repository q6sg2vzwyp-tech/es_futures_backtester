from ib_insync import *

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=1)

# If you lack real-time ES permissions, request delayed data (works in paper)
ib.reqMarketDataType(4)  # 1=real, 3=delayed, 4=delayed-frozen

# Resolve front-month ES
es = Future(symbol="ES", lastTradeDateOrContractMonth="", exchange="CME", currency="USD")
cds = ib.reqContractDetails(es)
if not cds:
    print("No contract details for ES—check exchange/permissions.")
    ib.disconnect()
    raise SystemExit(1)
es = cds[0].contract

# Request a streaming quote
tkr = ib.reqMktData(es, "", False, False)
ib.sleep(3)  # give TWS time to populate fields

# Show something useful even if last is None (common on delayed)
last = tkr.last if tkr.last is not None else tkr.marketPrice()
print("ES bid/ask/last:", tkr.bid, tkr.ask, last)

ib.disconnect()
