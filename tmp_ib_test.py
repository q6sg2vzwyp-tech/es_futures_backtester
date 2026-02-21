from ib_insync import IB
ib = IB()
print("Connecting...")
ib.connect("127.0.0.1", 4002, clientId=7777, timeout=5)
print("Connected:", ib.isConnected())
ib.disconnect()
print("Done")
