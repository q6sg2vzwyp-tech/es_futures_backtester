import io, re

p = r".\paper_trader.py"
data = open(p, "rb").read()
nul = data.count(b"\x00")

# 1) Remove NULs
if nul:
    data = data.replace(b"\x00", b"")

# 2) Normalize newlines to \n
data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

# 3) Decode; replace undecodable bytes
text = data.decode("utf-8", "replace")

# 4) Strip other control chars except \t and \n
text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)

# 5) Write back clean UTF-8 with \n newlines
with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print(f"nul_removed={nul}")
